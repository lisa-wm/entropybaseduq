"""Training routine for experiments."""

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from torch.utils.data import DataLoader, random_split

from udecomp.utils import init_weights


class NNLearner(pl.LightningModule):
    """Vanilla network training."""

    def __init__(
        self,
        device: str,
        model: nn.Module,
        hyperparams: Dict,
        dataset: Any,
        test_dataset: Optional[Any] = None,
        prop_val: float = 0.3,
        member: int = 0,
        seed: int = 1,
        regression: bool = False,
        num_classes: int = 10,
    ) -> None:
        """Set up learner object."""
        super().__init__()
        self._device = device
        self.member = member
        self.model = model.to(self._device)
        self.seed = seed
        self.regression = regression
        self.num_classes = num_classes

        self.prop_val = prop_val
        val_set_size = int(len(dataset) * self.prop_val)
        train_set_size = len(dataset) - val_set_size
        seed = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(
            dataset, [train_set_size, val_set_size], generator=seed
        )
        self.data_train = train_set
        self.data_val = val_set
        self.data_test = test_dataset

        self.optimizer = None
        self.scheduler = None
        self.hyperparams = hyperparams

        if not self.regression:
            if self.num_classes == 2:
                self.lossfun_train = nn.CrossEntropyLoss()
                self.lossfun_val = nn.CrossEntropyLoss()
            else:
                self.lossfun_train = nn.BCEWithLogitsLoss(reduction='none')
                self.lossfun_val = nn.BCEWithLogitsLoss(reduction='none')
            self.perf_train = torchmetrics.Accuracy(
                num_classes=self.num_classes,
                average='weighted',
                subset_accuracy=True,
                prefix='train',
            )
            self.perf_val = torchmetrics.Accuracy(
                num_classes=self.num_classes,
                average='weighted',
                subset_accuracy=True,
                prefix='val',
            )
            self.rmse_train, self.rmse_val = None, None
        else:
            self.lossfun_train = nn.GaussianNLLLoss()
            self.lossfun_val = nn.GaussianNLLLoss()
            self.perf_train = torchmetrics.MeanSquaredError(squared=False)
            self.perf_val = torchmetrics.MeanSquaredError(squared=False)
            self.acc_train, self.acc_val = None, None

        self.loss_train = torchmetrics.MeanMetric(prefix='train')
        self.loss_val = torchmetrics.MeanMetric(prefix='val')

    def on_fit_start(self) -> None:
        """Set global seed."""
        pl.seed_everything(seed=self.seed)

    def train_dataloader(self) -> DataLoader:
        """Set up data loader for training."""
        return DataLoader(
            self.data_train,
            batch_size=self.hyperparams.get('batch_size_train') or 32,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self) -> DataLoader:
        """Set up data loader for validation."""
        return DataLoader(
            self.data_val,
            batch_size=self.hyperparams.get('batch_size_test') or 32,
            num_workers=1,
        )

    def configure_optimizers(self) -> Dict:
        """Set up optimization-related objects."""
        if self.hyperparams.get('optimizer') == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.hyperparams.get('learning_rate') or 0.01,
                weight_decay=self.hyperparams.get('weight_decay') or 0.,
            )
        elif self.hyperparams.get('optimizer') == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hyperparams.get('learning_rate') or 0.01,
                weight_decay=self.hyperparams.get('weight_decay') or 0.,
                momentum=self.hyperparams.get('momentum') or 0.,
            )
        else:
            raise ValueError('Unknown optimizer')
        if self.hyperparams.get('lr_scheduler') == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=self.hyperparams.get('patience'),
                threshold=0.0001,
                threshold_mode='abs',
            )
        elif self.hyperparams.get('lr_scheduler') == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=10,
            )
        else:
            raise ValueError('Unknown scheduler')
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {'scheduler': self.scheduler, 'monitor': 'loss_val'},
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define standard forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Define training routine."""
        x, y = batch
        preds, targets, loss = self._compute_loss(x, y)
        self.loss_train.update(loss.detach())
        self.perf_train.update(preds, targets.long())

        return loss

    def training_epoch_end(self, outputs) -> None:
        """Collect metrics after each training step."""
        self.log('loss_train', self.loss_train.compute())
        self.log('perf_train', self.perf_train.compute())
        self.loss_train.reset()
        self.perf_train.reset()

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Define validation routine."""
        x, y = batch
        preds, targets, loss = self._compute_loss(x, y)
        self.loss_val.update(loss.detach())
        self.perf_val.update(preds, targets.long())

        return loss

    def validation_epoch_end(self, outputs) -> None:
        """Collect metrics after each validation step."""
        self.log('loss_val', self.loss_val.compute())
        self.log('perf_val', self.perf_val.compute())
        self.loss_val.reset()
        self.perf_val.reset()

    def init_model_weights(self, pretraining: bool, seed: Optional[int] = None) -> None:
        """Initialize model weights."""
        if seed is not None:
            torch.manual_seed(seed)
        self.model.apply(lambda m: init_weights(m, pretraining=pretraining))

    def get_model_id(self) -> str:
        """Get model identifier as handed to model factory."""
        return self.model.model_id

    def get_model(self) -> nn.Module:
        """Get model backbone."""
        return self.model

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_classes

    def set_member_number(self, number: int) -> None:
        """Assign member number in ensemble."""
        self.member = number

    def set_seed(self, seed: int) -> None:
        """Set seed to custom value."""
        self.seed = seed

    def _expand_targets(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Expand scalar class labels to one-hot encoding."""
        targets = torch.zeros_like(x, device=self._device)
        for row in range(targets.shape[0]):
            targets[row, y[row]] = 1
        return targets

    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> Tuple:
        """Compute loss function."""
        preds = self.model(x.float()) if self.regression else self.model(x)
        targets = y if self.regression else self._expand_targets(preds, y)
        if self.regression:
            var = (
                    self.get_model().sigma *
                    torch.ones(x.shape[0], 1, device=self._device)
            )
            loss = self.lossfun_train(preds, targets, var)
        else:
            loss = self.lossfun_train(preds, targets)
        if len(loss.shape) > 1:
            loss = loss.sum(1).mean()
        return preds, targets, loss
