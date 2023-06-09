"""Utility functions."""

import csv
import json
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
import math
import os.path
import random
from typing import Any, Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import accuracy, calibration_error


def load_json(file_path: str) -> Dict:
    """Load a json file as dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_as_json(dictionary: Any, target: str) -> None:
    """Save a python object as JSON file."""
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)


def save_as_csv(rows: List, target: str, header: Optional[List] = None) -> None:
    """Save a list of rows to a csv file."""
    with open(target, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)


def init_weights(layer: nn.Module, pretraining: bool = True) -> None:
    """Create checkpoint with network(s) to be loaded in learning."""
    if not pretraining:
        if isinstance(layer, nn.Conv2d):
            nn.init.normal(layer.weight, std=0.1)
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight)
    if getattr(layer, 'bias', None) is not None:
        nn.init.zeros_(layer.bias)


def comp_discr_entropy(probs: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """Compute Shannon entropy with base-two log."""
    probs_stabilized = probs + eps
    return -(probs * probs_stabilized.log2()).sum(-1)


def comp_diff_entropy(
    sigmas: torch.Tensor, eps: float = 0.0001
) -> torch.Tensor:
    """Compute differential entropy with natural log."""
    sigmas_stabilized = sigmas + eps
    return 0.5 * torch.log(2 * math.pi * sigmas_stabilized) + 0.5


def generate_shape(
    image_size: int,
    edge_ratio: float,
    seed: int,
    shape: str = 'rectangle',
) -> torch.Tensor:
    """Synthesize black image with white rectangle or ellipse."""
    img = Image.new('RGB', (image_size, image_size))
    random.seed(seed)
    if edge_ratio > 1:
        max_height = image_size - 2
        min_height = min(math.ceil(edge_ratio), max_height - 1)
        height = int(random.uniform(min_height, max_height))
        width = int(height // edge_ratio)
    else:
        max_width = image_size - 2
        min_width = min(math.ceil(1 / edge_ratio), max_width)
        width = int(random.uniform(min_width, max_width))
        height = math.ceil(width * edge_ratio)
    ref_x_max = image_size - (width + 1)
    ref_y_max = image_size - (height + 1)
    ref_x = int(random.uniform(1, ref_x_max))
    ref_y = int(random.uniform(1, ref_y_max))
    axes = ((ref_x, ref_y), (ref_x + width, ref_y + height))
    img_1 = ImageDraw.Draw(img)
    if shape == 'rectangle':
        img_1.rectangle(axes, fill='white')
    elif shape == 'ellipse':
        img_1.ellipse(axes, fill='white')
    else:
        raise ValueError('Can only draw rectangles and ellipses.')
    img_tensor = transforms.ToTensor()(img.convert('L'))
    return img_tensor


def draw_polygon(image_size: int, n_vertices: int, seed: int) -> torch.Tensor:
    """Synthesize black image with white polygon."""
    img = Image.new('RGB', (image_size, image_size))
    vertices: List = []
    for i in range(n_vertices):
        random.seed(seed + i)
        x_coord = int(random.uniform(1, image_size - 1))
        y_coord = int(random.uniform(1, image_size - 1))
        vertices.append((x_coord, y_coord))
    img_1 = ImageDraw.Draw(img)
    img_1.polygon(tuple(vertices), fill='white')
    img_tensor = transforms.ToTensor()(img.convert('L'))
    return img_tensor


def seed_everything(seed: int) -> None:
    """At least we tried."""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


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