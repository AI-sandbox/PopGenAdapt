import argparse

import wandb

from data import DataLoaders
from model import Model
from trainer import get_trainer


def parse_args():
    parser = argparse.ArgumentParser(prog='PopGenAdapt',
                                     description='Semi-Supervised Domain Adaptation for Genotype-to-Phenotype Prediction in Underrepresented Populations')

    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the dataset JSON file.')
    
    # Model arguments
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of layers in the MLP model.')
    parser.add_argument('--normalization', type=str, choices=['batch', 'layer'], default='layer',
                        help='Normalization to use in the MLP model.')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden size of the MLP model. If None, hidden size is equal to the input size.')
    parser.add_argument('--skip-every', type=int, default=2,
                        help='Number of layers between skip connections in the MLP model. If None, no skip connections.')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Temperature to use in the MLP model.')

    # SSDA arguments
    parser.add_argument('--ssda-method', type=str, choices=['base', 'ent', 'mme'], default='base',
                        help='Semi-supervised domain adaptation method to use.')
    parser.add_argument('--sla', action='store_true',
                        help='Whether to use source label adaptation.')

    # SLA arguments
    parser.add_argument('--sla-warmup', type=int, default=2500,
                        help='Number of iterations to warmup the source label adaptation.')
    parser.add_argument('--sla-temperature', type=float, default=0.6,
                        help='Temperature to use in the source label adaptation.')
    parser.add_argument('--sla-alpha', type=float, default=0.3,
                        help='Alpha to use in the source label adaptation.')
    parser.add_argument('--sla-update-interval', type=int, default=500,
                        help='Number of iterations between source label adaptation updates.')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for data loading and training / evaluation.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate for training.')
    parser.add_argument('--num-iters', type=int, default=15000,
                        help='Number of iterations to train for.')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Number of iterations between evaluations.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    wandb.init(project='PopGenAdapt', name=args.ssda_method + ("-sla" if args.sla else ""), config=vars(args))
    print("Loading data...", flush=True)
    data_loaders = DataLoaders(args.data, args.batch_size)
    in_features, out_features = data_loaders.in_features, data_loaders.out_features
    print("Building model...", flush=True)
    model = Model(in_features=in_features, out_features=out_features, num_layers=args.num_layers,
                  normalization=args.normalization, hidden_size=args.hidden_size, skip_every=args.skip_every,
                  temperature=args.temperature).cuda()
    print("Starting training...", flush=True)
    trainer = get_trainer(args.ssda_method, args.sla)(model=model, data_loaders=data_loaders, lr=args.lr, num_iters=args.num_iters,
                                                      unlabeled_method=args.ssda_method,  # for UnlabeledTrainer
                                                      num_classes=out_features, warmup=args.sla_warmup, temperature=args.sla_temperature, alpha=args.sla_alpha, update_interval=args.sla_update_interval)  # for SLATrainer
    trainer.train(eval_interval=args.eval_interval)
    print("Done!", flush=True)
