"""Utility functions."""

import math
import os.path
from copy import deepcopy
from typing import Any, Optional, List, Dict

import pandas as pd
import torch
from torch.utils.data import Subset
from torchmetrics.functional import accuracy, calibration_error

from udecomp.baselearner import NNLearner
from udecomp.evaluator import Evaluator
from udecomp.probabilistic_extensions import DeepEnsemble, LaplaceApproximator
from udecomp.utils import seed_everything, save_as_json, comp_discr_entropy


def run_experiment(
    result_path: str,
    model: torch.nn.Module,
    hyperparams: Dict,
    ensemble_size: int,
    epochs: int,
    data_train: Any,
    backbone: str,
    seed: int,
    num_classes: int = 10,
    prob_learner: str = 'ensemble',
    log_wandb: bool = False,
    train_idx: Optional[List] = None,
    testloader: Optional[torch.utils.data.DataLoader] = None,
) -> None:
    """Run experiment with given inputs."""
    seed_everything(seed)
    save_as_json(
        dict({'ensemble_size': ensemble_size}, **hyperparams),
        os.path.join(result_path, 'hyperparams.json')
    )
    learner = NNLearner(
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        model=model,
        hyperparams=hyperparams,
        dataset=Subset(data_train, train_idx) if train_idx is not None else data_train,
        regression=True if backbone == 'mlp' else False,
        prop_val=0.1,
        num_classes=num_classes,
        seed=seed,
    )
    if prob_learner == 'ensemble':
        prob_learner = DeepEnsemble(
            deepcopy(learner),
            ensemble_size=ensemble_size,
            seed=seed,
            log_wandb=log_wandb,
            patience=hyperparams['patience'],
            pretraining=backbone in ['resnet50', 'densenet121', 'efficientnetb7']
        )
    elif prob_learner == 'laplace':
        prob_learner = LaplaceApproximator(
            deepcopy(learner),
            ensemble_size=ensemble_size,
            seed=seed,
            log_wandb=log_wandb,
            patience=hyperparams['patience'],
        )
    else:
        raise ValueError('Unknown probabilistic learner')
    prob_learner.train(
        run_prefix=result_path, epochs=epochs if epochs is not None else 5
    )
    if testloader is not None:
        evaluator = Evaluator(result_path=result_path)
        print('---> Evaluating results...')
        evaluator.evaluate(learners=[prob_learner], dataloaders=[testloader])


def compute_eval_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    result_path: Optional[str] = None,
    incl_perf: bool = True,
) -> Dict:
    """Compute performance and uncertainty metrics."""
    is_classification = preds.shape[-1] > 1
    metric_dict: Dict = {}
    if torch.nanmean(preds).isnan():
        return metric_dict
    else:
        if is_classification:
            entropy_upper_bound = math.log2((preds.nanmean(0).shape[1]))
            tu = comp_discr_entropy(preds.nanmean(0)) / entropy_upper_bound
            au = comp_discr_entropy(preds).nanmean(0) / entropy_upper_bound
            eu = tu - au
            if result_path is not None:
                pd.DataFrame(tu).to_csv(os.path.join(result_path, 'tu_full.csv'))
                pd.DataFrame(au).to_csv(os.path.join(result_path, 'au_full.csv'))
                pd.DataFrame(eu).to_csv(os.path.join(result_path, 'eu_full.csv'))

            if incl_perf:
                acc = accuracy(
                    preds.nanmean(0),
                    targets.long(),
                    average='weighted',
                    num_classes=preds.shape[-1],
                )
                expce = calibration_error(
                    preds.nanmean(0), targets.long(), norm='l1', n_bins=10
                )
                maxce = calibration_error(
                    preds.nanmean(0), targets.long(), norm='max', n_bins=10
                )
                metric_dict.update(
                    {
                        'acc': format(float(acc), '.4f'),
                        'expce': format(float(expce), '.4f'),
                        'maxce': format(float(maxce), '.4f'),
                    }
                )
        else:  # re-implement regression if necessary
            return metric_dict
        metric_dict.update(
            {
                'tu': format(float(tu.mean(0)), '.4f'),
                'eu': format(float(eu.mean(0)), '.4f'),
                'au': format(float(au.mean(0)), '.4f'),
            }
        )
        return metric_dict