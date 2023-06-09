"""Datasets for experiments."""
import os
import random
import math
from typing import (
    Any,
    Dict,
    Final,
    Tuple, List,
)

import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor, Resize,
)

from udecomp.utils import generate_shape, draw_polygon

ROOT: Final[str] = 'data/'
AVAILABLE_DATASETS: Final[Dict] = {
    'cifar10': {'path': ROOT + 'cifar10/'},
    'mnist': {'path': ROOT + 'mnist/'},
    'mnist_standard': {'path': ROOT + 'mnist_standard/'},
    'airfoil': {'path': ROOT + 'uci/'},
    'concrete': {'path': ROOT + 'uci/'},
    'diabetes': {'path': ROOT + 'uci/'},
    'energy': {'path': ROOT + 'uci/'},
    'forest_fire': {'path': ROOT + 'uci/'},
    'wine': {'path': ROOT + 'uci/'},
    'yacht': {'path': ROOT + 'uci/'},
}


class DatasetFactory:
    """Custom class for experiments."""

    @staticmethod
    def get(
        dataset_id: str,
        train: bool,
        binarize: bool = False,
        small: bool = False,
        resolution: float = 1.,
        n_images: int = 100,
        input_size: int = 224,
        shape: str = 'rectangle',
        min_ratio: float = 1.,
        max_ratio: float = 100.,
        min_vertices: int = 3,
        max_vertices: int = 30,
    ) -> Any:
        """Return dataset from an identifier."""
        if dataset_id == 'cifar10':
            trafo = [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            if resolution < 1:
                trafo.append(Resize(math.ceil(resolution * 32)))
                trafo.append(Resize(32))
            data = datasets.CIFAR10(
                root=AVAILABLE_DATASETS[dataset_id]['path'],
                train=train,
                download=not os.path.exists(AVAILABLE_DATASETS[dataset_id]['path']),
                transform=Compose(trafo),
            )
        elif dataset_id == 'mnist':
            if resolution < 1:
                trafo = Compose(
                    [
                        Normalize((0.1307,), (0.3081,)),
                        Resize((int(resolution * 224), int(resolution * 224)))
                    ]
                )
            else:
                trafo = Normalize((0.1307,), (0.3081,))
            data = MNISTThreeChannels(
                root=AVAILABLE_DATASETS[dataset_id]['path'],
                train=train,
                transform=trafo,
                small=small,
            )
        elif dataset_id == 'mnist_standard':
            trafo = [ToTensor(), Normalize((0.1307,), (0.3081,))]
            if resolution < 1:
                trafo.append(Resize(math.ceil(resolution * 28)))
                trafo.append(Resize(28))
            data = datasets.MNIST(
                download=True,
                root=AVAILABLE_DATASETS[dataset_id]['path'],
                train=train,
                transform=Compose(trafo),
            )
        elif dataset_id == 'shapes':
            data = SyntheticShapes(n_images, min_ratio, max_ratio, input_size, shape)
        elif dataset_id == 'polygons':
            data = SyntheticPolygons(n_images, min_vertices, max_vertices, input_size)
        else:
            data = np.loadtxt(
                os.path.join(
                    AVAILABLE_DATASETS[dataset_id]['path'], dataset_id + '.data'
                )
            )
            x, y = data[:, :-1], data[:, -1]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=1
            )
            scaler_x = StandardScaler().fit(x_train)
            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
            x_train = scaler_x.transform(x_train)
            y_train = scaler_y.transform(y_train.reshape(-1, 1))
            x_test = scaler_x.transform(x_test)
            y_test = scaler_y.transform(y_test.reshape(-1, 1))
            if train:
                data = RegrDataset(x_train, y_train)
            else:
                data = RegrDataset(x_test, y_test)
        if binarize:
            pass
        return data


class MNISTThreeChannels(datasets.MNIST):
    """Three-channel MNIST dataset."""

    def __init__(
        self,
        root: str,
        transform: Any,
        train: bool = False,
        small: bool = False,
    ) -> None:
        """Instantiate three-channel MNIST dataset."""
        super().__init__(root=root, train=train, download=True, transform=transform)
        if small:
            self.data: torch.Tensor = self.data[:512, ...]
            self.targets: torch.Tensor = self.targets[:512, ...]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Getter method with three channels."""
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = ToTensor()(img)
        img_rgb = torch.stack((img.squeeze(0), img.squeeze(0), img.squeeze(0)))
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_rgb, target


class SyntheticShapes(Dataset):
    """Synthetic datasets of rectangles or ellipses with varying height/width ratio."""

    def __init__(
        self,
        n_images: int,
        min_ratio: float,
        max_ratio: float,
        input_size: int = 224,
        shape: str = 'rectangle'
    ) -> None:
        """Instantiate synthetic rectangles."""
        self.images: List = []
        self.targets: List = []
        for i in range(n_images // 2):
            random.seed(i)
            z = random.uniform(min_ratio, max_ratio)
            self.images.append(
                generate_shape(
                    image_size=input_size, edge_ratio=z, seed=i, shape=shape
                )
            )
            self.targets.append(1)
            self.images.append(
                generate_shape(
                    image_size=input_size, edge_ratio=1 / z, seed=i, shape=shape
                )
            )
            self.targets.append(0)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Getter method."""
        return self.images[index], self.targets[index]

    def __len__(self) -> int:
        """Retrieve number of observations."""
        return len(self.targets)


class SyntheticPolygons(Dataset):
    """Synthetic dataset of arbitrary polygons."""

    def __init__(
        self,
        n_images: int,
        min_vertices: int,
        max_vertices: int,
        input_size: int = 224
    ) -> None:
        """Instantiate synthetic rectangles."""
        self.images: List = []
        self.targets: List = []
        for i in range(n_images):
            random.seed(i)
            n_vertices = int(random.uniform(min_vertices, max_vertices))
            self.images.append(
                draw_polygon(image_size=input_size, n_vertices=n_vertices, seed=i)
            )
            self.targets.append(1)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Getter method."""
        return self.images[index], self.targets[index]

    def __len__(self) -> int:
        """Retrieve number of observations."""
        return len(self.targets)


class RegrDataset(Dataset):
    """Torch dataset for benchmark data."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Instantiate dataset."""
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
        self.n_features = self.x.shape[1]

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from index."""
        return self.x[idx], self.y[idx]
