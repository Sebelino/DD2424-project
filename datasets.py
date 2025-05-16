import copy
from dataclasses import dataclass, asdict
from typing import Optional, Any, Tuple

import torch
import torchvision
from torchvision import transforms
from joblib import Memory
from torch.utils.data import DataLoader, random_split, Dataset

from determinism import Determinism
from util import dumps_inline_lists
from augmentation import create_fixmatch_transform, FixMatchTransform

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
    # Either "category" (default), "binary-category" (cat/dog), or a tuple with both
    target_types: Optional[str] = "category"
    labelled_data_fraction: Optional[float] = 1.0
    # Specify percentage of each class to use to create imbalanced dataset.
    # Default is to use 100% of each class. There are 37 classes in the pet dataset.
    # Example: To use 20% of the cat samples, set
    #   class_fractions = (0.2,)*25 + (1.0,)*12
    # or, if using binary-category
    #   class_fractions = (0.2,) + (1.0,)
    class_fractions: Optional[Tuple[float, ...]] = (1.0,) * 37
    # Per-class weights to use when oversampling
    oversampling_weights: Optional[Tuple[float, ...]] = None

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


class FixMatchDataset(Dataset):
    def __init__(self, base_dataset, fixmatch_transform: FixMatchTransform):
        self.base_dataset = base_dataset
        self.fixmatch_transform = fixmatch_transform
        
    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]
        
        # If img is already a tensor, convert back to PIL for transforms
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL
            img = transforms.ToPILImage()(img)
        
        # Apply both transformations to the same image
        weak_img, strong_img = self.fixmatch_transform(img)
        
        return (weak_img, strong_img), target
    
    def __len__(self):
        return len(self.base_dataset)


def balanced_random_split_indices(dataset, lengths, splitting_seed, class_fractions):
    """
    Splits a dataset into non-overlapping subsets while preserving
    class distributions.  The split prioritizes class proportions, and
    the subset sizes may be adjusted slightly to maintain these
    proportions. The subsets are not shuffled after the
    split. Optionally, you can control the fraction of samples to use
    from each class to create an imbalanced dataset by specifying
    `class_fractions`, where values range from 0.0 (exclude the class)
    to 1.0 (use all samples).

    Args:
        dataset: Dataset, where each item is a tuple (data, label).
        lengths: Tuple of floats (0.0-1.0) representing the fraction
                 of the full dataset that should go into each subset.
        splitting_seed: Seed for reproducibility.
        class_fractions: Tuple of floats (0.0-1.0) specifying the fraction
                         of samples to include from each class.

    Returns:
        A list of lists, where each list contains the indices of samples
        for each subset.
    """  
    assert 0 <= sum(lengths) <= 1.0, (
        "Sum of split proportions in 'lengths' must be between 0 and 1"
    )
    
    assert all(
        0.0 <= f <= 1.0 for f in class_fractions
    ), "All class fractions must be between 0 and 1"
    
    # Extract labels
    try:
        # Fast, but relies on private attribute
        labels = dataset._labels
    except AttributeError:
        # Fallback to slow, safe method
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    # Group indices by label (class)
    class_to_indices = dict()
    for i, label in enumerate(labels):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(i)
        
    generator = torch.Generator().manual_seed(splitting_seed)

    # Prepare lists to collect indices for each subset
    subset_indices = [[] for _ in range(len(lengths))]

    # Split each subset (class) separately
    for label, indices in class_to_indices.items():
        num_samples = int(len(indices) * class_fractions[label])      
        split_sizes = [int(p * num_samples) for p in lengths]
        remainder = num_samples - sum(split_sizes)
        for i in range(remainder):
            split_sizes[i % len(lengths)] += 1
        # Shuffle and split
        shuffled = torch.randperm(len(indices), generator=generator).tolist()
        class_indices = [indices[i] for i in shuffled[:num_samples]]
        cursor = 0
        for i, size in enumerate(split_sizes):
            subset_indices[i].extend(class_indices[cursor:cursor + size])
            cursor += size
            
    return subset_indices

def create_fixmatch_dataloaders(unlabelled_subset, dataset_params, num_workers, fixmatch_transform):
    if fixmatch_transform is None:
        return None    # Create FixMatch dataset
    fixmatch_dataset = FixMatchDataset(
        unlabelled_subset, 
        fixmatch_transform
    )
    
    # Create dataloader
    unlabelled_train_loader = DataLoader(
        fixmatch_dataset,
        batch_size=dataset_params.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(dataset_params.shuffler_seed),
        worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.shuffler_seed),
    )
    return unlabelled_train_loader

def make_datasets(dataset_params: DatasetParams, base_transform, training_transform, fixmatch_transform):
    target_types = dataset_params.target_types
    base_trainval_dataset = load_dataset("trainval", base_transform, target_types)
    augmented_trainval_dataset = load_dataset("trainval", training_transform, target_types)

    if dataset_params.trainval_size is not None:
        subset_fraction = dataset_params.trainval_size/len(base_trainval_dataset)
    else:
        subset_fraction = 1.0

    num_workers = 3

    # 80% train, 20% val split
    val_fraction = subset_fraction * dataset_params.validation_set_fraction
    train_fraction = subset_fraction * (1 - dataset_params.validation_set_fraction) 
    labelled_fraction = train_fraction * dataset_params.labelled_data_fraction
    unlabelled_fraction = train_fraction * (1 - dataset_params.labelled_data_fraction)
    
    labelled_indices, unlabelled_indices, val_indices = balanced_random_split_indices(
        base_trainval_dataset,
        (labelled_fraction, unlabelled_fraction, val_fraction),
        dataset_params.splitting_seed,
        dataset_params.class_fractions
    )
    
    labelled_subset = torch.utils.data.Subset(base_trainval_dataset, labelled_indices)
    unlabelled_subset = torch.utils.data.Subset(base_trainval_dataset, unlabelled_indices)
    val_subset = torch.utils.data.Subset(base_trainval_dataset, val_indices)
    
    # Weighted sampling to compensate for imbalanced classes
    # Note: sample_weights is a weight for each sample in the dataset,
    #       but oversampling_weights is a weight for each class!
    if dataset_params.oversampling_weights is not None:
        sample_weights = [
            dataset_params.oversampling_weights[labelled_subset.dataset[i][1]]
            for i in labelled_subset.indices
        ]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # Must be False when sampler is used
    else:
        sampler = None
        shuffle = True
        
    labelled_train_loader = DataLoader(
        labelled_subset,
        batch_size=dataset_params.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(dataset_params.shuffler_seed),
        worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.shuffler_seed),
    )

    unlabelled_train_loader = None
    if len(unlabelled_subset) > 0:
        unlabelled_train_loader = create_fixmatch_dataloaders(
            unlabelled_subset, 
            dataset_params, 
            num_workers,
            fixmatch_transform
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
