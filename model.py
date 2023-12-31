import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eta=1.0):
        ctx.eta = eta
        # See https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4 on why we need to use "x.view_as(x)"
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.eta
        return output, None


def grad_reverse(x, eta=1.0) -> torch.Tensor:
    return GradReverse.apply(x, eta)


class Classifier(nn.Module):
    """
    MLP classifier using GELU activations and optionaly skip connections and batch or layer normalization.

    Parameters
    ----------
    in_features : int
        number of input features

    out_features : int
        number of output features

    num_layers : int
        number of layers in the MLP

    layer_norm : bool
        whether to use layer normalization

    hidden_size : int | None
        size of hidden layers. If None, hidden size is equal to the input size

    skip_every : int | None
        number of layers between skip connections. If None, no skip connections

    temperature : float
        the features are normalized to unit length and then divided by this value
    """

    def __init__(self, in_features, out_features, num_layers=1, layer_norm=True, hidden_size=None, skip_every=None, temperature=0.05):
        super().__init__()

        self.num_layers = num_layers
        self.skip_every = skip_every if skip_every is not None else num_layers + 1
        self.layer_norm = layer_norm
        self.temperature = temperature

        self.mlp = nn.ModuleList()
        if num_layers == 1:
            self.mlp.append(nn.Linear(in_features, out_features))
        else:
            if hidden_size is None:
                hidden_size = in_features
            self.mlp.append(nn.Linear(in_features, hidden_size))
            self.mlp.append(nn.GELU())
            if layer_norm:
                self.mlp.append(nn.LayerNorm(hidden_size))
            for _ in range(num_layers - 2):
                self.mlp.append(nn.Linear(hidden_size, hidden_size))
                self.mlp.append(nn.GELU())
                if layer_norm:
                    self.mlp.append(nn.LayerNorm(hidden_size))
            self.mlp.append(nn.Linear(hidden_size, out_features))

    def forward(self, x, reverse=False):
        # (batch_size, in_features)
        x = self.get_features(x, reverse=reverse)
        # (batch_size, hidden_size)
        x = self.get_predictions(x)
        # (batch_size, out_features)
        return x

    def get_features(self, x, reverse=False):
        # (batch_size, in_features)
        if self.num_layers > 1:
            x = self.mlp[0](x)
            skip = x
            for i, layer in enumerate(self.mlp[1:-1]):
                x = layer(x)
                if (i - 1) % ((2 + int(self.layer_norm)) * self.skip_every) == 0:
                    x = x + skip
                    skip = x
        # (batch_size, hidden_size)
        if reverse:
            x = grad_reverse(x)
        return F.normalize(x, dim=1) / self.temperature

    def get_predictions(self, x):
        # (batch_size, hidden_size)
        x = self.mlp[-1](x)
        # (batch_size, out_features)
        return x


class ProtoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.center = None
        self.label = None
        self.num_classes = num_classes

    def init(self, model, t_loader):
        model.eval()
        t_pred, t_feat = [], []
        with torch.no_grad():
            for x, _ in t_loader:
                x = x.cuda().float()
                t_feat.append(model.get_features(x))
                t_pred.append(model.get_predictions(t_feat[-1]))
        t_pred, t_feat = torch.vstack(t_pred), torch.vstack(t_feat)
        label = t_pred.argmax(dim=1)
        center = torch.nan_to_num(torch.vstack([t_feat[label == i].mean(dim=0) for i in range(self.num_classes)]))
        invalid_idx = center.sum(dim=1) == 0
        if invalid_idx.any() and self.label is not None:
            logging.warning("Invalid center(s) found, using old center(s) instead")
            old_center = torch.nan_to_num(torch.vstack([t_feat[self.label == i].mean(dim=0) for i in range(self.num_classes)]))
            center[invalid_idx] = old_center[invalid_idx]
        else:
            self.label = label
        self.center = center.requires_grad_(False)
        model.train()

    @torch.no_grad()
    def forward(self, x, temperature=1.0):
        assert self.center is not None, 'ProtoClassifier not initialized'
        return F.softmax(-torch.cdist(x, self.center) * temperature, dim=1)


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, layer_norm=True, hidden_size=None, skip_every=None, temperature=0.05):
        super().__init__()

        self.classifier = Classifier(in_features, out_features, num_layers, layer_norm, hidden_size, skip_every, temperature)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, reverse=False):
        return self.classifier(x, reverse=reverse)

    def get_features(self, x, reverse=False):
        return self.classifier.get_features(x, reverse=reverse)

    def get_predictions(self, f):
        return self.classifier.get_predictions(f)

    def base_loss(self, x, y):
        return self.criterion(self.forward(x), y).mean()

    def feature_base_loss(self, f, y):
        return self.criterion(self.get_predictions(f), y).mean()

    def mme_loss(self, x, lamda=0.1):
        out = self.forward(x, reverse=True)
        out = F.softmax(out, dim=1)
        return lamda * torch.mean(torch.sum(out * torch.log(out + 1e-7), dim=1))

    def sla_loss(self, f, y1, y2, alpha):
        out = self.get_predictions(f)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(dim=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()
