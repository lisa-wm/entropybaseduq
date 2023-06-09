"""Evaluate experiments."""

import os
from itertools import combinations
from typing import List

import click
import pandas as pd
import torch

from udecomp.plot_data_generator import PlotDataGenerator
from udecomp.utils import save_as_json, compute_eval_metrics


@click.command()
@click.option('-ex', '--experiment', required=True)
@click.option('-pl', '--prob-learner', required=True)
@click.option('-ds', '--dataset-id', required=True)
def main(experiment: str, prob_learner: str, dataset_id: str, root: str = 'results'):
    """Eval experiments."""
    # Compute performance & uncertainty metrics for all seeds/options
    result_path = os.path.join(root, experiment, prob_learner, dataset_id)
    random_seeds = os.listdir(result_path)
    for rs in random_seeds:
        exp_options = [
            f for f in os.listdir(os.path.join(result_path, f'{rs}')) if '.' not in f
        ]
        logfiles: List = []
        for op in exp_options:
            this_path = os.path.join(result_path, f'{rs}', f'{op}')
            logfile_name = os.path.join(this_path, f'logfile_{op}.json')
            pred_files = [
                os.path.join(this_path, f) for f in os.listdir(this_path)
                if 'scores' in f and '.pt' in f
            ]
            if len(pred_files) == 0:
                logfiles.append(logfile_name)
                save_as_json({}, logfile_name)
                continue
            pred_all = []
            for pf in pred_files:
                pred_all.append(
                    torch.tensor(torch.load(pf, map_location=torch.device('cpu')))
                )
            pred_all = torch.stack(tuple(pred_all))
            targets = torch.load(
                os.path.join(this_path, 'targets.pt'), map_location=torch.device('cpu')
            )
            metrics = compute_eval_metrics(pred_all, torch.tensor(targets), this_path)
            logfiles.append(logfile_name)
            save_as_json(metrics, logfile_name)
            tu, eu, au = [], [], []
            all_combos = list(combinations(range(pred_all.shape[0]), r=5))
            for idx in all_combos:
                pred_5 = pred_all[idx, ...]
                metrics_5 = compute_eval_metrics(pred_5, torch.tensor(targets))
                tu.append(metrics_5['tu'])
                eu.append(metrics_5['eu'])
                au.append(metrics_5['au'])
            pd.DataFrame(tu).to_csv(os.path.join(this_path, 'tu_five.csv'))
            pd.DataFrame(eu).to_csv(os.path.join(this_path, 'eu_five.csv'))
            pd.DataFrame(au).to_csv(os.path.join(this_path, 'au_five.csv'))
        pdg = PlotDataGenerator(save_dir=os.path.join(result_path, f'{rs}'))
        pdg.make_plot_data(
            result_files=sorted(logfiles),
            cases=sorted(exp_options),
            random_seed=rs,
            prob_learner=prob_learner,
            dataset_id=dataset_id,
            filename=f'uvs{experiment}',
        )


if __name__ == '__main__':
    main()
