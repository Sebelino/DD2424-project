import copy
from dataclasses import dataclass, asdict
from typing import Optional, Any

import torch
import torchvision
from joblib import Memory
from torch.utils.data import DataLoader, random_split, Dataset

from determinism import Determinism
from util import dumps_inline_lists

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


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
        img, *_ = self.ds[i]  # drop all labels
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


@memory.cache
def balanced_random_split(dataset, lengths, splitting_seed):
    """
    Splits a dataset into non-overlapping subsets, while maintaining the class distribution 
    in each subset. Class proportions are prioritized, so the split sizes might be slightly 
    adjusted to maintain these proportions. The final subsets are not shuffled.

    Args:
        dataset: A dataset, where each item is a tuple of (data, label).
        lengths: A list specifying the split sizes, either as absolute counts or as fractions 
                 that sum to 1.
        splitting_seed: Seed for `torch.Generator` for reproducibility.

    Returns:
        A list of `torch.utils.data.Subset` objects, each representing one subset of the dataset.
    """
    from tqdm.auto import tqdm
    print("Creating balanced split...")

    generator = torch.Generator().manual_seed(splitting_seed)
    # Group sample indices by class label (slow for large datasets)
    class_to_indices = dict()
    for i in tqdm(range(len(dataset))):
        label = dataset[i][1]
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(i)

    # If lengths are absolute, convert them to proportions
    if all(isinstance(x, int) for x in lengths):
        total_length = sum(lengths)
        lengths = [x / total_length for x in lengths]

    # Prepare lists to collect indices for each subset
    subset_indices = [[] for _ in range(len(lengths))]

    # Split each subset (class) separately
    for indices in class_to_indices.values():
        class_size = len(indices)
        split_sizes = [int(p * class_size) for p in lengths]
        remainder = class_size - sum(split_sizes)
        for i in range(remainder):
            split_sizes[i % len(lengths)] += 1

        # Shuffle and split
        shuffled = torch.randperm(class_size, generator=generator).tolist()
        class_indices = [indices[i] for i in shuffled]
        cursor = 0
        for i, size in enumerate(split_sizes):
            subset_indices[i].extend(class_indices[cursor:cursor + size])
            cursor += size

    # Create Subsets from the indices and return them
    subsets = [torch.utils.data.Subset(dataset, indices) for indices in subset_indices]

    return subsets


def make_datasets(dataset_params: DatasetParams, transform):
    target_types = "category"
    if dataset_params.binary:
        target_types = "binary-category"
    trainval_dataset = load_dataset("trainval", transform, target_types)

    if dataset_params.trainval_size is not None:
        subset_size = dataset_params.trainval_size
    else:
        subset_size = len(trainval_dataset)

    num_workers = 3

    # 80% train, 20% val split
    train_set_fraction = 1 - dataset_params.validation_set_fraction
    num_train = int(train_set_fraction * subset_size)
    num_val = subset_size - num_train
    num_discard = len(trainval_dataset) - subset_size
    splitter_generator = torch.Generator().manual_seed(dataset_params.splitting_seed)
    # Note: len(train_subset) might different from num_train when doing balanced split
    train_subset, val_subset, _ = balanced_random_split(trainval_dataset,
                                                        [num_train, num_val, num_discard],
                                                        splitting_seed=dataset_params.splitting_seed)

    # Unlabelled dataset if present
    num_labelled = int(dataset_params.labelled_data_fraction * len(train_subset))
    num_unlabelled = len(train_subset) - num_labelled
    labelled_subset, unlabelled_subset = balanced_random_split(train_subset, [num_labelled, num_unlabelled],
                                                      splitting_seed=dataset_params.splitting_seed)

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
