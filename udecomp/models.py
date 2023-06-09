"""Models for experiments."""

from typing import (
    Dict,
    Final,
    List,
    Optional,
)

import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    EfficientNet_B7_Weights,
    densenet121,
    resnet50,
    efficientnet_b7,
)

AVAILABLE_MODELS: Final[Dict] = {
    'resnet50': {'weights': ResNet50_Weights.IMAGENET1K_V1, 'in_last': 2048},
    'densenet121': {'weights': DenseNet121_Weights.IMAGENET1K_V1, 'in_last': 1024},
    'efficientnetb7': {
        'weights': EfficientNet_B7_Weights.IMAGENET1K_V1, 'in_last': 2560  # 1536
    },
    'mlp': {'hidden_sizes': [32, 32, 32]},
    'convnet': {},
    'smallconvnet': {}
}


class ModelFactory:
    """Factory for creating torch model objects."""

    @staticmethod
    def get(
        model_id: str,
        num_classes: int,
        freeze: bool = False,
        input_size: Optional[int] = None,
    ) -> nn.Module:
        """Return an initialized model instance from an identifier."""
        if model_id in ['resnet50', 'densenet121', 'efficientnetb7']:
            clf = nn.Sequential(
                nn.Linear(
                    AVAILABLE_MODELS[model_id]['in_last'], num_classes
                )
            )
            if model_id == 'resnet50':
                model = resnet50(weights=AVAILABLE_MODELS[model_id]['weights'])
                model.fc = clf
            elif model_id == 'efficientnetb7':
                model = efficientnet_b7(weights=AVAILABLE_MODELS[model_id]['weights'])
                model.classifier = clf
            elif model_id == 'densenet121':
                model = densenet121(weights=AVAILABLE_MODELS[model_id]['weights'])
                model.classifier = clf
            else:
                raise ValueError(f'Model `{model_id}` not available.')
            if freeze:
                for child in list(model.children())[: -len(clf)]:
                    for param in child.parameters():
                        param.requires_grad = False
        elif model_id == 'convnet':
            model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3)),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=(3, 3)),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(800, 64),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(64, num_classes),
            )
        elif model_id == 'smallconvnet':
            model = SmallConvNet()
        elif model_id == 'mlp':
            if input_size is None:
                raise ValueError('Input size must be provided for MLP.')
            model = MLP(
                input_size=input_size,
                hidden_sizes=AVAILABLE_MODELS[model_id]['hidden_sizes'],
            )
        else:
            raise NotImplementedError(f'Model `{model_id}` not available.')

        model.num_classes = num_classes
        model.model_id = model_id

        return model


class SmallConvNet(nn.Module):
    """Small convnet for MNIST data."""

    def __init__(self):
        super().__init__()
        self.dropout_rate = 0.25
        self.n_classes = 10
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(3, 3))
        self.fc1 = nn.Linear(338, 128)
        self.fc2 = nn.Linear(128, self.n_classes)

    def forward(self, x):
        """Define forward pass."""
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = x.view(x.shape[0], -1)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, p=self.dropout_rate)
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output


class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(self, input_size: int, hidden_sizes: List[int]) -> None:
        """Instantiate MLP."""
        super().__init__()
        hidden_id = '_'.join([str(x) for x in hidden_sizes])
        self.model_id = f'MLP_{input_size}_{hidden_id}_1'
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.net = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]))
        for i, o in zip(hidden_sizes, hidden_sizes[1:] + [1]):
            self.net.append(nn.Tanh())
            self.net.append(nn.Linear(i, o))
        self.last_layer = self.net[-1]
        self.sigma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward pass."""
        return self.net(x)
