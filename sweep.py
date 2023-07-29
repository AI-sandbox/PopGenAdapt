import argparse

import wandb


def parse_args():
    parser = argparse.ArgumentParser(prog='python sweep.py',
                                     description='Intialize a wandb hyperparameter sweep for PopGenAdapt for a given dataset and SSDA method.')

    parser.add_argument('--data', type=str, required=True,
                        help='Path to the dataset JSON file.')
    parser.add_argument('--ssda_method', type=str, choices=['base', 'ent', 'mme'], default='base',
                        help='Semi-supervised domain adaptation method to use.')
    parser.add_argument('--sla', action='store_true',
                        help='Whether to use source label adaptation.')
    parser.add_argument('--suffix', type=str, default='',  # useful for different phenotypes
                        help='Suffix to append to the project name.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    project_name = f"PopGenAdapt-{args.ssda_method}{('-sla' if args.sla else '')}{'-' + args.suffix if args.suffix else ''}"

    sweep_config = {
        'program': 'main.py',
        'method': 'random',
        'metric': {
            'name': 'best_val_auc',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
        },
        'command': [
            '${env}',
            'python3',
            '${program}',
            '--verbose',
            '--project',
            project_name,
            '--seed',
            '42',
            '--data',
            args.data,
            '--ssda_method',
            args.ssda_method,
        ] + ([] if not args.sla else ['--sla']) + ['${args}']
    }

    if args.sla:
        sweep_config['parameters'].update({
            'sla_warmup': {
                'values': [100, 500, 1000, 2000, 5000]
            },
            'sla_temperature': {
                'min': 0.0,
                'max': 1.0,
                'distribution': 'uniform'
            },
            'sla_alpha': {
                'min': 0.0,
                'max': 1.0,
                'distribution': 'uniform'
            },
            'sla_update_interval': {
                'values': [5, 10, 100, 500, 1000, 2000, 5000]
            }
        })

    sweep_id = wandb.sweep(sweep_config,
                           project=project_name)
    wandb.agent(sweep_id)
