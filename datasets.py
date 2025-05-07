from dataclasses import dataclass
from typing import Optional

import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from determinism import Determinism


def load_dataset(split_name: str, transform, target_types: str = "category"):
    return torchvision.datasets.OxfordIIITPet(
        root="./data",
        split=split_name,
        target_types=target_types,
        download=True,
        transform=transform,
    )


@dataclass
class DatasetParams:
    # Seed for splitting into train & val sets
    splitting_seed: int
    # Seed for shuffling the data in each epoch
    shuffler_seed: int
    batch_size: int
    # validation_set_fraction=0.2 means that the validation set is 20 % of the trainval dataset
    validation_set_fraction: float
    # Size of training + validation set. None means "all".
    trainval_size: Optional[int] = None
    binary: Optional[bool] = False


def make_datasets(dataset_params: DatasetParams, transform):
    target_types = "category"
    if dataset_params.binary:
        target_types = "binary-category"
    trainval_dataset = load_dataset("trainval", transform, target_types)

    if dataset_params.trainval_size is not None:
        subset_size = dataset_params.trainval_size
        trainval_dataset = torch.utils.data.Subset(trainval_dataset, range(subset_size))

    num_workers = 3

    # 80% train, 20% val split
    train_set_fraction = 1 - dataset_params.validation_set_fraction
    num_train = int(train_set_fraction * len(trainval_dataset))
    num_val = len(trainval_dataset) - num_train
    splitter_generator = torch.Generator().manual_seed(dataset_params.splitting_seed)
    train_subset, val_subset = random_split(trainval_dataset, [num_train, num_val], generator=splitter_generator)

    train_loader = DataLoader(
        train_subset,
        batch_size=dataset_params.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(dataset_params.shuffler_seed),
        worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.shuffler_seed),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=dataset_params.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.shuffler_seed),
    )

    return train_loader, val_loader
