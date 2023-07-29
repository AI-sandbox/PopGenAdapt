import logging
import time

import torch
import torch.nn.functional as F
import torchmetrics
import wandb

from data import DataLoaders
from model import ProtoClassifier, Model


class LearningRateScheduler:
    """Gamma learning rate scheduler"""

    def __init__(self, optimizer, num_iters, step=0):
        self.optimizer = optimizer
        self.iter = step
        self.num_iters = num_iters
        self.base = self.optimizer.param_groups[-1]["lr"]

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base * ((1 + 0.0001 * self.iter) ** (-0.75))
        self.iter += 1

    def refresh(self):
        self.iter = 0

    def get_lr(self):
        return self.optimizer.param_groups[-1]["lr"]


class BaseTrainer:
    def __init__(self, model: Model, data_loaders: DataLoaders, lr=1e-3, num_iters=10000, **kwargs):
        self.model = model
        self.data_loaders = data_loaders
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.lr_scheduler = LearningRateScheduler(self.optimizer, num_iters)
        self.num_iters = num_iters

    def get_source_loss(self, step, sx, sy):
        return self.model.base_loss(sx, sy)

    def get_target_loss(self, step, tx, ty):
        return self.model.base_loss(tx, ty)

    def evaluation_step(self, loader, num_classes=2):
        assert num_classes >= 2
        if num_classes == 2:
            metric = torchmetrics.classification.BinaryAUROC()
        else:
            metric = torchmetrics.classification.MulticlassAUROC(num_classes=num_classes)
        with torch.no_grad():
            for x, y in loader:
                x, y = x.float().cuda(), y.long().cuda()
                out = self.model(x)
                if num_classes == 2:
                    y_pred = F.sigmoid(out[:, 1])
                else:
                    y_pred = F.softmax(out, dim=1)
                metric.update(y_pred, y)
        auc = metric.compute()
        return auc

    def evaluate(self):
        self.model.eval()
        logging.info("Validation")
        val_auc = self.evaluation_step(self.data_loaders.target_labeled_validation)
        logging.info("Test")
        test_auc = self.evaluation_step(self.data_loaders.target_labeled_test)
        return val_auc, test_auc

    def training_step(self, step, sx, sy, tx, ty, ux):
        self.optimizer.zero_grad()
        s_loss = self.get_source_loss(step, sx, sy)
        t_loss = self.get_target_loss(step, tx, ty)

        loss = (s_loss + t_loss) / 2
        loss.backward()
        self.optimizer.step()

        return s_loss.item(), t_loss.item(), 0

    def train(self, eval_interval, early_stop=None):
        best_val_auc = 0.0
        iterations_without_improvement = 0
        iter_train_data_loaders = iter(self.data_loaders)
        self.model.train()
        start_time = time.time()
        logging.info("Training started")
        for step in range(1, self.num_iters + 1):
            logging.info(f"Step {step}")
            (sx, sy), (tx, ty), ux = next(iter_train_data_loaders)
            # with torch.autograd.detect_anomaly():  # for debugging the next line
            s_loss, t_loss, u_loss = self.training_step(step, sx, sy, tx, ty, ux)
            self.lr_scheduler.step()

            wandb.log({
                's_loss': s_loss,
                't_loss': t_loss,
                'u_loss': u_loss,
                'lr': self.lr_scheduler.get_lr(),
                'time': (time.time() - start_time) / 60,
            }, step=step)

            if step % eval_interval == 0 or step == self.num_iters:
                logging.info("Evaluating")
                val_auc, test_auc = self.evaluate()
                wandb.log({
                    'val_auc': val_auc,
                    'test_auc': test_auc,
                }, step=step)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    iterations_without_improvement = 0
                    if wandb.run:
                        wandb.run.summary['best_val_auc'] = val_auc
                        wandb.run.summary['best_test_auc'] = test_auc
                else:
                    iterations_without_improvement += eval_interval
                    if early_stop is not None and iterations_without_improvement >= early_stop:
                        logging.info("Early stopping")
                        break


class UnlabeledTrainer(BaseTrainer):
    def __init__(self, model: Model, data_loaders, lr=1e-3, num_iters=10000, unlabeled_method='mme', **kwargs):
        super().__init__(model, data_loaders, lr, num_iters)
        self.unlabeled_method = unlabeled_method

    def unlabeled_training_step(self, step, ux):
        self.optimizer.zero_grad()
        u_loss = self.model.unlabeled_loss(step, ux, method=self.unlabeled_method)
        u_loss.backward()
        self.optimizer.step()

        return u_loss.item()

    def training_step(self, step, sx, sy, tx, ty, ux):
        s_loss, t_loss, _ = super().training_step(step, sx, sy, tx, ty, ux)
        u_loss = self.unlabeled_training_step(step, ux)
        return s_loss, t_loss, u_loss


def get_sla_trainer(Trainer):
    class SLATrainer(Trainer):
        def __init__(self, num_classes, warmup, temperature, alpha, update_interval, **trainer_kwargs):
            super().__init__(**trainer_kwargs)
            self.ppc = ProtoClassifier(num_classes)

            self.warmup = warmup
            self.temperature = temperature
            self.alpha = alpha
            self.update_interval = update_interval

        def get_source_loss(self, step, sx, sy):
            sf = self.model.get_features(sx)
            if step > self.warmup:
                sy2 = self.ppc(sf.detach(), self.temperature)
                s_loss = self.model.sla_loss(sf, sy, sy2, self.alpha)
            else:
                s_loss = self.model.feature_base_loss(sf, sy)
            return s_loss

        def ppc_update(self, step):
            if (step == self.warmup) or (step > self.warmup and step % self.update_interval == 0):
                if step == self.warmup:
                    self.lr_scheduler.refresh()
                self.ppc.init(self.model, self.data_loaders.target_unlabeled_train)

        def training_step(self, step, sx, sy, tx, ty, ux):
            s_loss, t_loss, u_loss = super().training_step(step, sx, sy, tx, ty, ux)
            self.ppc_update(step)
            return s_loss, t_loss, u_loss

    return SLATrainer


def get_trainer(method, sla):
    Trainer = None
    if method == "base":
        Trainer = BaseTrainer
    else:  # "ent" or "mme"
        Trainer = UnlabeledTrainer

    if sla:
        Trainer = get_sla_trainer(Trainer)

    return Trainer
