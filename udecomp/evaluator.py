"""Evaluation protocol for experiments."""

import itertools
import os.path
from typing import (
    List,
    Optional,
)

import pandas as pd
import torch
from torch.utils.data import DataLoader
from udecomp.probabilistic_extensions import ProbabilisticPredictor


class Evaluator:
    """Evaluating model predictions on suitable metrics."""

    def __init__(self, result_path: str) -> None:
        """Instantiate evaluator."""
        self.result_path = result_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def evaluate(
        self,
        learners: List[ProbabilisticPredictor],
        dataloaders: List[DataLoader],
        ensemble_sizes: Optional[List[int]] = None,
    ) -> None:
        """Evaluate learners on datasets w.r.t. several metrics."""
        ensemble_sizes = ensemble_sizes or [pp.get_ensemble_size() for pp in learners]
        for (idx, learner), loader in itertools.product(
            enumerate(learners), dataloaders
        ):
            targets = torch.empty(0, device=self.device)
            for _, y in loader:
                targets = torch.concat((targets, y.to(self.device)))
            torch.save(targets, os.path.join(self.result_path, 'targets.pt'))
            preds, _ = learner.predict(
                loader, avg=False, n_members=ensemble_sizes[idx]
            )
            for m in range(preds.shape[0]):
                pred_bl = preds[m].cpu().numpy().round(4)
                torch.save(pred_bl, os.path.join(self.result_path, f'scores_{m}.pt'))
                pred_bl_df = pd.DataFrame(pred_bl)
                pred_bl_df.to_csv(os.path.join(self.result_path, f'scores_{m}.csv'))