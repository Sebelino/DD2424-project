from dataclasses import dataclass
from typing import Optional

import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from determinism import Determinism


def load_dataset(split_name: str, transform):
    return torchvision.datasets.OxfordIIITPet(
        root="./data",
        split=split_name,
        target_types="category",
        download=True,
        transform=transform,
    )


@dataclass
class DatasetParams:
    torch_seed: int
    batch_size: int
    # Size of training + validation set. None means "all".
    training_size: Optional[int] = None


def make_datasets(dataset_params: DatasetParams, transform):
    train_dataset = load_dataset("trainval", transform)

    if dataset_params.training_size is not None:
        subset_size = dataset_params.training_size
        small_train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))
        train_dataset = small_train_dataset

    num_workers = 3

    # 80% train, 20% val split
    num_train = int(0.8 * len(train_dataset))
    num_val = len(train_dataset) - num_train
    torch_generator = torch.Generator()
    Determinism.torch_generator(dataset_params.torch_seed, torch_generator)
    train_subset, val_subset = random_split(train_dataset, [num_train, num_val], generator=torch_generator)

    # DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=dataset_params.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.torch_seed),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=dataset_params.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.torch_seed),
    )

    return train_loader, val_loader
