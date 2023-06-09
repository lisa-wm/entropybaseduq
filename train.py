"""Perform experiments."""

import os
import random
from copy import deepcopy
from typing import (
    Dict,
    Final,
    List,
    Optional,
)

import click
import numpy as np
import torch.cuda
from torch.utils.data import DataLoader, Subset

from udecomp.utils import seed_everything
from udecomp.baselearner import NNLearner
from udecomp.datasets import DatasetFactory
from udecomp.evaluator import Evaluator
from udecomp.models import ModelFactory
from udecomp.probabilistic_extensions import DeepEnsemble, LaplaceApproximator
from utils_train import run_experiment

SAMPLE_SIZES: Final[List] = [0.01, 0.02, 0.05, 0.1, 0.5, 1]
RESOLUTIONS: Final[List] = [0.5, 0.1, 0.05]


@click.command()
@click.option('-ex', '--experiment', required=True)
@click.option('-bb', '--backbone', required=True, default='resnet50')
@click.option('-fz', '--freeze', required=True, default=False)
@click.option('-ds', '--dataset-id', required=True, default='mnist')
@click.option('-bs', '--batch-size', required=True, default=64)
@click.option('-wd', '--weight-decay', required=True, default=0.0)
@click.option('-lr', '--learning-rate', required=True, default=0.001)
@click.option('-op', '--optimizer', required=True, default='rmsprop')
@click.option('-ls', '--lr-scheduler', required=True, default='plateau')
@click.option('-mo', '--momentum', required=False)
@click.option('-pl', '--prob-learner', required=True, default='ensemble')
@click.option('-pt', '--patience', required=True, default=5)
@click.option('-es', '--ensemble-size', required=False)
@click.option('-ep', '--epochs', required=False)
@click.option('-wb', '--log-wandb', required=True, default=False)
@click.option('-rs', '--random-seed', required=True)
def main(
    experiment: str,
    backbone: str,
    freeze: bool,
    dataset_id: str,
    batch_size: int,
    learning_rate: float,
    optimizer: str,
    lr_scheduler: str,
    prob_learner: str,
    patience: int,
    weight_decay: float,
    random_seed: int,
    ensemble_size: Optional[int] = None,
    epochs: Optional[int] = None,
    momentum: Optional[float] = None,
    log_wandb: bool = False,
):
    """Run experiments."""
    random_seed = int(random_seed)
    seed_everything(random_seed)
    # Fetch data
    data_train = DatasetFactory.get(dataset_id, train=True)
    data_test = DatasetFactory.get(dataset_id, train=False)
    testloader = DataLoader(data_test, batch_size=32, num_workers=1, shuffle=False)
    # Get model
    if backbone == 'mlp':
        model = ModelFactory.get(
            backbone, num_classes=1, input_size=data_train.x.shape[-1]
        )
    else:
        model = ModelFactory.get(
            backbone, num_classes=len(data_train.classes), freeze=bool(freeze)
        )
    # Store hyperparameters from CLI
    hyperparams: Dict = {
        'prob_learner': prob_learner,
        'learning_rate': float(learning_rate),
        'batch_size': int(batch_size),
        'weight_decay': float(weight_decay),
        'patience': int(patience),
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'momentum': float(momentum) or 0.,
    }

    if experiment == 'samples':
        for idx, i in enumerate(SAMPLE_SIZES):
            size = str(int(100 * i))
            result_path = os.path.join(
                'results',
                experiment,
                prob_learner,
                dataset_id,
                f'{random_seed}',
                f'{size}',
            )
            os.makedirs(result_path, exist_ok=True)
            # Sample data according to current target size
            torch.manual_seed(random_seed + idx)
            train_idx = torch.from_numpy(
                np.random.choice(
                    range(len(data_train)),
                    size=int(i * len(data_train)),
                    replace=False,
                )
            )
            run_experiment(
                result_path=result_path,
                model=model,
                prob_learner=prob_learner,
                hyperparams=hyperparams,
                ensemble_size=int(ensemble_size),
                epochs=int(epochs),
                data_train=data_train,
                train_idx=list(train_idx),
                testloader=testloader,
                backbone=backbone,
                seed=random_seed + idx,
                log_wandb=log_wandb,
                num_classes=len(data_train.classes),
            )

    elif experiment == 'resolution':
        for res in RESOLUTIONS:
            res_int = int(100 * res)
            result_path = os.path.join(
                'results',
                experiment,
                prob_learner,
                dataset_id,
                f'{random_seed}',
                f'{res_int}',
            )
            os.makedirs(result_path, exist_ok=True)
            data_train = DatasetFactory.get(dataset_id, train=True, resolution=res)
            data_test = DatasetFactory.get(dataset_id, train=False, resolution=res)
            testloader = DataLoader(
                data_test, batch_size=32, num_workers=1, shuffle=False
            )
            run_experiment(
                result_path=result_path,
                model=model,
                prob_learner=prob_learner,
                hyperparams=hyperparams,
                ensemble_size=int(ensemble_size),
                epochs=int(epochs),
                data_train=data_train,
                testloader=testloader,
                backbone=backbone,
                seed=random_seed,
                log_wandb=log_wandb,
                num_classes=len(data_train.classes),
            )

    elif experiment == 'synthetic':
        n_train: Final[int] = 60000
        n_test: Final[int] = 10000
        # Train on synthetic rectangles
        data_train = DatasetFactory.get(
            'shapes',
            train=True,
            min_ratio=1.,
            max_ratio=200.,
            input_size=28,
            n_images=n_train,
        )
        learner = NNLearner(
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            model=ModelFactory.get(backbone, num_classes=2, freeze=False),
            hyperparams=hyperparams,
            dataset=data_train,
            prop_val=0.1,
            num_classes=2,
        )
        if prob_learner == 'ensemble':
            pl = DeepEnsemble(
                deepcopy(learner),
                ensemble_size=int(ensemble_size),
                seed=random_seed,
                log_wandb=log_wandb,
                patience=hyperparams['patience'],
                pretraining=backbone in [
                    'resnet50', 'densenet121', 'efficientnetb7'
                ]
            )
        elif prob_learner == 'laplace':
            pl = LaplaceApproximator(
                deepcopy(learner),
                ensemble_size=ensemble_size,
                seed=random_seed,
                log_wandb=log_wandb,
                patience=hyperparams['patience'],
            )
        else:
            raise ValueError('Unknown probabilistic learner')
        pl.train(
            run_prefix=f'{experiment}_{backbone}_{dataset_id}_{random_seed}',
            epochs=int(epochs) if epochs is not None else 5,
        )
        # Evaluate on rectangles with varying quadratic-ness
        for ratio in [1., 2., 5.]:
            ratio_int = int(ratio)
            result_path = os.path.join(
                'results',
                f'{experiment}_au',
                prob_learner,
                dataset_id,
                f'{random_seed}',
                f'{ratio_int}',
            )
            os.makedirs(result_path, exist_ok=True)
            data_test = DatasetFactory.get(
                'shapes',
                train=True,
                min_ratio=ratio,
                max_ratio=ratio,
                n_images=n_test,
                input_size=28,
            )
            testloader = DataLoader(
                data_test, batch_size=32, num_workers=1, shuffle=False
            )
            evaluator = Evaluator(result_path=result_path)
            print('---> Evaluating results...')
            evaluator.evaluate(learners=[pl], dataloaders=[testloader])
        # Evaluate on IND (rectangles) vs OOD (polygons)
        data_test_ind = DatasetFactory.get(
            'shapes',
            train=False,
            min_ratio=1.,
            max_ratio=200.,
            n_images=n_test,
            input_size=28,
        )
        data_test_ood = DatasetFactory.get(
            'polygons',
            train=False,
            min_vertices=3,
            max_vertices=5,
            n_images=n_test,
            input_size=28
        )
        for m, dataset in zip(['ind', 'ood'], [data_test_ind, data_test_ood]):
            result_path = os.path.join(
                'results',
                f'{experiment}_ood',
                prob_learner,
                dataset_id,
                f'{random_seed}',
                f'{m}',
            )
            os.makedirs(result_path, exist_ok=True)
            testloader = DataLoader(
                dataset, batch_size=32, num_workers=1, shuffle=False
            )
            evaluator = Evaluator(result_path=result_path)
            print('---> Evaluating results...')
            evaluator.evaluate(learners=[pl], dataloaders=[testloader])

    elif experiment == 'ood':
        os.makedirs(os.path.join('results', experiment), exist_ok=True)
        # Define list of classes to act as OOD
        ood_classes: List = [0]
        for ood in ood_classes:
            # For the OOD training, remove the OOD class from the training indices;
            # keep all classes for InD but resize to size of OOD training set
            train_idx_ood = [
                idx for idx, label in enumerate(data_train.targets) if label != ood
            ]
            train_idx_all = random.sample(
                range(len(data_train)), len(train_idx_ood)
            )
            test_idx = [
                idx for idx, label in enumerate(data_test.targets) if label == ood
            ]
            testloader = DataLoader(
                Subset(data_test, test_idx),
                batch_size=int(batch_size),
                num_workers=1,
                shuffle=False,
            )
            for m, idx in zip(['ood', 'all'], [train_idx_ood, train_idx_all]):
                result_path = os.path.join(
                    'results',
                    experiment,
                    prob_learner,
                    dataset_id,
                    f'{random_seed}',
                    f'{m}',
                )
                os.makedirs(result_path, exist_ok=True)
                run_experiment(
                    result_path=result_path,
                    model=model,
                    prob_learner=prob_learner,
                    hyperparams=hyperparams,
                    ensemble_size=int(ensemble_size),
                    epochs=int(epochs),
                    data_train=data_train,
                    train_idx=idx,
                    testloader=testloader,
                    backbone=backbone,
                    seed=random_seed,
                    log_wandb=log_wandb,
                    num_classes=len(data_train.classes),
                )


if __name__ == '__main__':
    main()
