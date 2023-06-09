"""Conduct additional experiments on toy datasets."""

import os
from collections import namedtuple
from typing import Final, List, Any, Dict
import shutil

import click
import pandas as pd
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from udecomp.utils import save_as_json, compute_eval_metrics

N_SAMPLES: Final[int] = 7000
TRAIN_SPLIT: Final[float] = 6/7
CLASSES_DEFAULT: Final[int] = 4
FEATURES_DEFAULT: Final[int] = 2
SEP_DEFAULT: Final[float] = 1.
MEMBERS_DEFAULT: Final[int] = 10
MEMBERS_DEFAULT_MLP: Final[int] = 10
DEPTH_DEFAULT: Final[int] = 10
DEPTH_DEFAULT_MLP: Final[int] = 10

"""0 is just random, 5 is full seperation, but with two clusters
per class instead of 1, so that may influence the results (experiment with this)"""
DISTS: Final[List] = [0.05, 0.25, 0.5, 1.0, 2.5, 5.0]
NOISES: Final[List] = [0.01, 0.1, 0.25, 0.5, 0.75]
MEMBERS: Final[List] = [3, 5, 10, 20, 50]
DEPTHS: Final[List] = [1, 3, 5, 7, 10, 20]

EXPERIMENTS: Final[Dict] = {
    'data_situation': {'distance': DISTS, 'noise': NOISES},
    'learner_situation': {
        'members': MEMBERS, 'complexity': DEPTHS, 'bootstrap': [True, False],
    }
}


@click.command()
@click.option('-ex', '--experiment', required=True)
@click.option('-pl', '--ensemble-learner', required=True, default='rf')
@click.option('-rs', '--random-seed', required=True)
def main(experiment: str, ensemble_learner: str, random_seed: int):
    """Run experiments."""

    ClassificationDataset = namedtuple('ClassificationDataset', ['dsid', 'data'])
    EnsembleLearner = namedtuple(
        'EnsembleLearner', ['elid', 'type', 'learner']
    )
    seed: Final[int] = int(random_seed)

    dataset_args = {
        'n_samples': N_SAMPLES,
        'n_features': FEATURES_DEFAULT,
        'n_informative': FEATURES_DEFAULT,
        'n_redundant': 0,
        'n_repeated': 0,
        'n_clusters_per_class': 1,
        'class_sep': SEP_DEFAULT,
        'random_state': seed,
    }
    if ensemble_learner == 'rf':
        learner_args = {'random_state': seed, 'criterion': 'entropy'}
    else:
        learner_args = {
            'solver': 'sgd',
            'max_iter': 100,
            'learning_rate_init': 0.1,
            'early_stopping': True,
        }

    rs_dir = os.path.join('results', experiment, ensemble_learner, str(random_seed))
    if os.path.exists(rs_dir):
        shutil.rmtree(rs_dir)

    if experiment in EXPERIMENTS['data_situation'].keys():
        for case in EXPERIMENTS['data_situation'][experiment]:
            case_str = str(case) if isinstance(case, int) else str(int(100 * case))
            result_path = os.path.join(
                'results',
                experiment,
                ensemble_learner,
                str(random_seed),
                case_str
            )
            os.makedirs(result_path, exist_ok=True)
            if experiment == 'distance':
                dataset_args['class_sep'] = case
            dataset = ClassificationDataset(
                experiment,
                make_classification(
                    n_classes=case if experiment == 'classes' else CLASSES_DEFAULT,
                    flip_y=case if experiment == 'noise' else 0.01,
                    **dataset_args
                )
            )
            x, y = dataset.data
            dt = pd.DataFrame(
                {
                    'x_1': x[:, 0],
                    'x_2': x[:, 1],
                    'y': y,
                    'experiment': experiment,
                    'case': case,
                    'rs': seed,
                }
            )
            dt.to_csv(os.path.join(result_path, 'dataset.csv'))
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=TRAIN_SPLIT, random_state=seed
            )
            if ensemble_learner == 'rf':
                el = EnsembleLearner(
                    elid=f'{experiment}_rf',
                    type='rf',
                    learner=RandomForestClassifier(
                        max_depth=DEPTH_DEFAULT, **learner_args
                    )
                )
                clf = el.learner
                clf.fit(x_train, y_train)
                preds = [bl.predict_proba(x_test) for bl in clf.estimators_]
            elif ensemble_learner == 'mlp':
                np.random.seed = seed
                el = EnsembleLearner(
                    elid=f'{experiment}_mlp',
                    type='mlp',
                    learner=[
                        MLPClassifier(
                            hidden_layer_sizes=(DEPTH_DEFAULT_MLP,), **learner_args
                        )
                        for _ in range(MEMBERS_DEFAULT)
                    ]
                )
                clf = el.learner
                idx_bs = [
                    np.random.choice(
                        range(len(x_train)), size=len(x_train), replace=True
                    )
                    for _ in range(MEMBERS_DEFAULT)
                ]
                x_train_bs = [x_train[i, ...] for i in idx_bs]
                y_train_bs = [y_train[i, ...] for i in idx_bs]
                clf_trained: List[Any] = [
                    bl.fit(x, y) for bl, x, y in zip(clf, x_train_bs, y_train_bs)
                ]
                preds = [bl.predict_proba(x_test) for bl in clf_trained]
            else:
                raise ValueError('unknown learner')
            preds = np.array(preds)
            torch.save(y_test, os.path.join(result_path, 'targets.pt'))
            for m in range(preds.shape[0]):
                pred_bl = preds[m].round(4)
                torch.save(pred_bl, os.path.join(result_path, f'scores_{m}.pt'))
            metrics = compute_eval_metrics(torch.tensor(preds), torch.tensor(y_test))
            save_as_json(metrics, os.path.join(result_path, f'logfile.json'))

    elif experiment in EXPERIMENTS['learner_situation'].keys():
        for case in EXPERIMENTS['learner_situation'][experiment]:
            case_str = str(case)
            result_path = os.path.join(
                'results',
                experiment,
                ensemble_learner,
                str(random_seed),
                case_str
            )
            os.makedirs(result_path, exist_ok=True)
            dataset = ClassificationDataset(
                experiment,
                make_classification(
                    n_classes=CLASSES_DEFAULT,
                    flip_y=0.01,
                    **dataset_args
                )
            )
            x, y = dataset.data
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=TRAIN_SPLIT, random_state=seed
            )
            if ensemble_learner == 'rf':
                el = EnsembleLearner(
                    elid=f'{experiment}_rf',
                    type='rf',
                    learner=RandomForestClassifier(
                        n_estimators=case
                        if experiment == 'members' else MEMBERS_DEFAULT,
                        max_depth=case if experiment == 'complexity' else DEPTH_DEFAULT,
                        **learner_args
                    )
                )
                clf = el.learner
                clf.fit(x_train, y_train)
                preds = [bl.predict_proba(x_test) for bl in clf.estimators_]
            elif ensemble_learner == 'mlp':
                n_members = case if experiment == 'members' else MEMBERS_DEFAULT_MLP
                np.random.seed(seed)
                el = EnsembleLearner(
                    elid=f'{experiment}_mlp',
                    type='mlp',
                    learner=[
                        MLPClassifier(
                            hidden_layer_sizes=(
                                case
                                if experiment == 'complexity' else DEPTH_DEFAULT_MLP,
                            ),
                            **learner_args
                        )
                        for _ in range(n_members)
                    ]
                )
                clf = el.learner
                np.random.seed(seed)  # perform manual bootstrapping
                idx_bs = [
                    np.random.choice(
                        range(len(x_train)), size=len(x_train), replace=True
                    )
                    for _ in range(MEMBERS_DEFAULT)
                ]
                x_train_bs = [x_train[i, ...] for i in idx_bs]
                y_train_bs = [y_train[i, ...] for i in idx_bs]
                clf_trained: List[Any] = [
                    bl.fit(x, y) for bl, x, y in zip(clf, x_train_bs, y_train_bs)
                ]
                preds = [bl.predict_proba(x_test) for bl in clf_trained]
            else:
                raise ValueError('unknown learner')

            preds = np.array(preds)
            torch.save(y_test, os.path.join(result_path, 'targets.pt'))
            for m in range(preds.shape[0]):
                pred_bl = preds[m].round(4)
                torch.save(pred_bl, os.path.join(result_path, f'scores_{m}.pt'))
            metrics = compute_eval_metrics(torch.tensor(preds), torch.tensor(y_test))
            save_as_json(metrics, os.path.join(result_path, f'logfile.json'))

    else:
        raise ValueError('Unknown experiment')


if __name__ == '__main__':
    main()