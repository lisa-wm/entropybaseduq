"""Visualizations."""

import os
from typing import Dict, List
import pandas as pd
from udecomp.utils import load_json


class PlotDataGenerator:
    """Class to create plots."""

    def __init__(self, save_dir: str):
        """Instantiate plotter."""
        self.plots: Dict = {}
        self.save_dir = save_dir

    def make_plot_data(
            self,
            result_files: List[str],
            cases: List[str],
            random_seed: str,
            prob_learner: str,
            dataset_id: str,
            filename: str
    ) -> None:
        """Create plots from results."""
        results = [load_json(f) for f in result_files]
        identifier = [prob_learner, dataset_id, random_seed]
        rows = []
        for idx, _ in enumerate(results):
            if results[idx].get('tu') is None:
                continue
            tu = results[idx]['tu']
            au = results[idx]['au']
            eu = results[idx]['eu']
            ece = results[idx]['expce']
            acc = results[idx]['acc']
            rows.append(identifier + [cases[idx], 'tu', tu])
            rows.append(identifier + [cases[idx], 'au', au])
            rows.append(identifier + [cases[idx], 'eu', eu])
            rows.append(identifier + [cases[idx], 'ece', ece])
            rows.append(identifier + [cases[idx], 'acc', acc])
        df = pd.DataFrame(
            rows, columns=['pl', 'ds', 'rs', 'case', 'metric', 'value']
        )
        df.to_csv(os.path.join(self.save_dir, f'{filename}.csv'))