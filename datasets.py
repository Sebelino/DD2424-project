import copy
from dataclasses import dataclass, asdict
from typing import Optional, Any

import torch
import torchvision
from torch.utils.data import DataLoader, random_split, Dataset

from determinism import Determinism
from util import dumps_inline_lists


def load_dataset(split_name: str, transform, target_types: str = "category"):
    return torchvision.datasets.OxfordIIITPet(
        root="./data",
        split=split_name,
        target_types=target_types,
        download=True,
        transform=transform,
    )

# Wrap the “unlabelled” subset so it only returns the image
class UnlabelledDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        img, *_ = self.ds[i]   # drop all labels
        return img


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
    labelled_data_fraction: Optional[float] = 1.0

    def copy(self) -> 'DatasetParams':
        return copy.deepcopy(self)

    def minimal_dict(self) -> dict[str, Any]:
        def prune(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    key: prune(val)
                    for key, val in obj.items()
                    if val is not None
                }
            if isinstance(obj, (list, tuple)):
                pruned = [prune(val) for val in obj if val is not None]
                return type(obj)(pruned)
            return obj

        return prune(asdict(self))

    def pprint(self):
        return dumps_inline_lists(self.minimal_dict())


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

    # Unlabelled dataset if present
    num_labelled = int(dataset_params.labelled_data_fraction * num_train)
    num_unlabelled = num_train - num_labelled
    labelled_subset, unlabelled_subset = random_split(train_subset, [num_labelled, num_unlabelled], generator=splitter_generator) 
        

    labelled_train_loader = DataLoader(
        labelled_subset,
        batch_size=dataset_params.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(dataset_params.shuffler_seed),
        worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.shuffler_seed),
    )

    unlabelled_train_loader = None
    if len(unlabelled_subset) > 0:
        unlabelled_train_loader = DataLoader(
            unlabelled_subset,
            batch_size=dataset_params.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=False,
            pin_memory=True,
            generator=torch.Generator().manual_seed(dataset_params.shuffler_seed),
            worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.shuffler_seed),
        )
        

    val_loader = DataLoader(
        val_subset,
        batch_size=dataset_params.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.shuffler_seed),
    )

    return labelled_train_loader, unlabelled_train_loader, val_loader
