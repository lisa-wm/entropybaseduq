"""Wrappers to make existing (deterministic) NN Bayesian."""
import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    List,
    Optional,
    Tuple,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import wandb
from laplace import Laplace, LLLaplace
from torch import nn
from torch.nn.functional import softmax
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader

from udecomp.baselearner import NNLearner


class ProbabilisticPredictor(ABC):
    """Abstract class for probabilistic prediction."""

    @abstractmethod
    def get_id(self) -> str:
        """Return predictor OD."""
        pass

    @abstractmethod
    def get_ensemble_size(self) -> int:
        """Return ensemble size."""
        pass

    @abstractmethod
    def train(self, run_prefix: str, epochs: int) -> None:
        """Train predictor."""
        pass

    @abstractmethod
    def predict(
        self,
        dataloader: DataLoader,
        n_members: Optional[int] = None,
        avg: bool = False,
    ) -> torch.Tensor:
        """Predict on given data."""
        pass

    @abstractmethod
    def get_models(self) -> List[nn.Module]:
        """Return trained models."""
        pass

    def get_sigma(self) -> torch.Tensor:
        """Return sigmas."""
        pass


class DeepEnsemble(ProbabilisticPredictor):
    """Ensemble wrapper for NN learners."""

    def __init__(
        self,
        base_learner: NNLearner,
        pretraining: bool,
        ensemble_size: int,
        patience: int,
        ckpt: Optional[str] = None,
        log_wandb: bool = True,
        seed: int = 1,
    ) -> None:
        """Instantiate ensemble."""
        self.base_learner_type = base_learner.get_model_id()
        self.pretraining = pretraining
        self.num_classes = base_learner.get_num_classes()
        self.regression = base_learner.regression
        self.ensemble_size = ensemble_size
        self.ensemble = []
        for _ in range(self.ensemble_size):
            self.ensemble.append(deepcopy(base_learner))
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.ckpt = ckpt
        self.log_wandb = log_wandb
        self.seed = seed
        self.patience = patience

        self._init(pretraining)

    def _init(self, pretraining: bool):
        """Initialize NN weights."""
        for idx, bl in enumerate(self.ensemble):
            bl.set_seed(self.seed + idx)

    def get_id(self) -> str:
        """Return predictor ID."""
        return 'de'

    def get_ensemble_size(self) -> int:
        """Return ensemble size."""
        return self.ensemble_size

    def get_models(self) -> List[nn.Module]:
        """Return models."""
        return [bl.get_model() for bl in self.ensemble]

    def get_sigma(self) -> torch.Tensor:
        """Return sigmas."""
        if not self.regression:
            raise ValueError('Only available for regression.')
        sigmas = torch.empty(0, device=self.device)
        for m in self.get_models():
            sigmas = torch.cat((sigmas, m.sigma.detach().to(self.device)))
        return sigmas

    def train(self, run_prefix: str, epochs: int) -> None:
        """Train ensemble."""
        print('---> Training ensemble...')
        lr = self.ensemble[0].hyperparams.get('learning_rate')
        bs = self.ensemble[0].hyperparams.get('batch_size_train')
        bb = self.ensemble[0].get_model_id()

        for idx, bl in enumerate(self.ensemble):
            if self.log_wandb:
                wandb.init(
                    project='udecomp',
                    tags=[f'lr_{lr}', f'bs_{bs}', f'bb_{bb}', f'pt_{self.patience}']
                )
                logger = pl.loggers.WandbLogger(name=f'{run_prefix}_{idx}')
            else:
                logger = None
            trainer = pl.Trainer(
                max_epochs=epochs,
                logger=logger,
                gpus=torch.cuda.device_count(),
                num_sanity_val_steps=0,
                deterministic=True,
                precision=16,
                callbacks=[
                    EarlyStopping(
                        'loss_val', patience=self.patience, check_finite=False
                    )
                ],
                enable_checkpointing=False,
            )
            print(f'---> Training ensemble member {idx + 1}...')
            trainer.fit(bl)
            if self.ckpt is not None:
                trainer.save_checkpoint(os.path.join(self.ckpt, f'trainer_{idx}.ckpt'))
            if self.log_wandb:
                wandb.finish()  # necessary for each run to be actually logged

    def predict(
        self,
        dataloader: DataLoader,
        n_members: Optional[int] = None,
        avg: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with ensemble."""
        print('---> Computing ensemble prediction...')

        if n_members is not None:
            members_idx = random.sample(range(self.ensemble_size), n_members)
        else:
            members_idx = [idx for idx, _ in enumerate(self.ensemble)]

        ensemble_prediction = []
        if self.regression:
            sigmas = self.get_sigma()
        else:
            sigmas = torch.empty(0, device=self.device)

        for idx, bl in enumerate(self.ensemble):
            if idx not in members_idx:
                continue
            bl_predictions = torch.empty(0, device=self.device)
            for x, _ in dataloader:
                with torch.no_grad():
                    bl.get_model().eval()
                    if not self.regression:
                        bl.get_model().eval()
                        prediction = softmax(bl(x), dim=1).to(self.device)
                    else:
                        prediction = bl(x.float()).to(self.device)
                bl_predictions = torch.cat((bl_predictions, prediction), dim=0)
            ensemble_prediction.append(bl_predictions)
        ensemble_prediction = torch.stack(tuple(ensemble_prediction))
        if avg:
            ensemble_prediction = torch.mean(ensemble_prediction, dim=0)

        return ensemble_prediction, sigmas


class LaplaceApproximator(ProbabilisticPredictor):
    """LA wrapper for NN learners."""

    def __init__(
        self,
        base_learner: NNLearner,
        ensemble_size: int,
        patience: int,
        ckpt: Optional[str] = None,
        log_wandb: bool = True,
        seed: int = 1,
    ) -> None:
        """Instantiate Laplace approximator."""
        self.base_learner_type = base_learner.get_model_id()
        self.bl = base_learner
        self.num_classes = base_learner.get_num_classes()
        self.regression = base_learner.regression
        self.ensemble_size = ensemble_size
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.ckpt = ckpt
        self.log_wandb = log_wandb
        self.sigma = None
        self.seed = seed,
        self.patience = patience

        self.laplace_learner = Laplace(
            model=self.bl.model,
            likelihood='classification' if not self.regression else 'regression',
        )

    def get_id(self) -> str:
        """Return predictor ID."""
        return 'la'

    def get_ensemble_size(self) -> int:
        """Return ensemble size."""
        return self.ensemble_size

    def get_models(self) -> List[nn.Module]:
        """Return models."""
        return [self.bl.get_model()]

    def get_sigma(self) -> torch.Tensor:
        """Return sigmas."""
        if self.sigma is not None:
            return self.sigma
        else:
            raise ValueError

    def train(self, run_prefix: str, epochs: int) -> None:
        """Train Laplace approximator."""
        print('---> Training base model...')
        lr = self.bl.hyperparams.get('learning_rate')
        bs = self.bl.hyperparams.get('batch_size_train')
        bb = self.bl.get_model_id()
        if self.log_wandb:
            wandb.init(
                project='udecomp',
                tags=['la', f'lr_{lr}', f'bs_{bs}', f'bb_{bb}', f'pt_{self.patience}']
            )
            logger = pl.loggers.WandbLogger(name=f'{run_prefix}')
        else:
            logger = None
        trainer = pl.Trainer(
            max_epochs=epochs,
            logger=logger,
            gpus=torch.cuda.device_count(),
            num_sanity_val_steps=0,
            deterministic=True,
            precision=16,
            enable_checkpointing=False,
            callbacks=[EarlyStopping('loss_val', patience=self.patience)],
        )
        if self.ckpt is not None:
            trainer.save_checkpoint(os.path.join(self.ckpt, 'trainer.ckpt'))
        trainer.fit(self.bl)
        print('---> Training Laplace approximator...')
        model = self.bl.get_model()
        if self.regression:
            self.sigma = model.sigma.detach()
            model.sigma = None
        if self.bl.get_model_id() == 'convnet':
            weight_set = 'all'
        else:
            weight_set = 'last_layer'
        self.laplace_learner = Laplace(
            model=model,
            likelihood='classification' if not self.regression else 'regression',
            subset_of_weights=weight_set,
            hessian_structure='kron',
        )
        self.laplace_learner.fit(self.bl.train_dataloader())
        self.laplace_learner.optimize_prior_precision(
            pred_type='nn' if not self.regression else 'glm', method='marglik'
        )
        if self.log_wandb:
            wandb.finish()  # necessary for each run to be actually logged

    def predict(
        self,
        dataloader: DataLoader,
        n_members: Optional[int] = None,
        avg: bool = False,
        seed: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with Laplace approximator."""
        print('---> Drawing from Laplace approximate posterior...')

        model = self.bl.get_model()
        ensemble_prediction = []
        ensemble_sigma = torch.empty(0)
        torch.manual_seed(seed)

        for sample in self.laplace_learner.sample(int(self.ensemble_size)):
            bl_predictions = torch.empty(0, device=self.device)
            if isinstance(self.laplace_learner, LLLaplace):
                vector_to_parameters(sample, model.classifier.parameters())
            else:
                vector_to_parameters(sample, model.parameters())
            for x, _ in dataloader:
                with torch.no_grad():
                    model.eval()
                    if not self.regression:
                        prediction = softmax(model(x.float()), dim=1).to(self.device)
                    else:
                        prediction = model(x.float()).to(self.device)
                bl_predictions = torch.cat((bl_predictions, prediction), dim=0)
            ensemble_prediction.append(bl_predictions)
        ensemble_prediction = torch.stack(tuple(ensemble_prediction))
        if self.regression:
            ensemble_sigma = torch.var(ensemble_prediction, dim=1)
        if avg:
            ensemble_prediction = torch.mean(ensemble_prediction, dim=0)
            if self.regression:
                ensemble_sigma = torch.mean(ensemble_sigma, dim=0)

        return ensemble_prediction, ensemble_sigma

