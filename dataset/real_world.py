import json
from dataclasses import dataclass

import numpy as np
import tyro

from dataset.generator import DatasetArgs
from dataset.models import Dataset


@dataclass
class RealWorldDatasetArgs(DatasetArgs):
    pass


def generate_real_world_dataset(args: RealWorldDatasetArgs, rng: np.random.RandomState) -> Dataset:
    """
    Generate a dataset with the specified arguments.
    """
    raise NotImplementedError()


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    dataset = generate_real_world_dataset(tyro.cli(RealWorldDatasetArgs), rng)
    json_data = json.dumps(dataset.to_json())
    print(json_data)
