"""
split.py

Utilities to split datasets into balanced subsets for training and validation,
supporting label-wise subsampling, reproducible shuffling, and size constraints.
"""
import torch
from typing import List, Tuple, Optional
from torch.utils.data import Dataset

def get_labels(dataset: Dataset) -> torch.Tensor:
    """
    Extracts labels from a dataset.

    Tries to read a `_labels` attribute; if unavailable, extracts labels by indexing the dataset.
    Returns labels as a 1D torch.Tensor.
    """
    try:
        if isinstance(dataset, torch.utils.data.Subset):
            all_labels = torch.tensor(dataset.dataset._labels)
            return all_labels[dataset.indices]
        else:
            return torch.tensor(dataset._labels)
    except AttributeError:
        return torch.tensor([dataset[i][1] for i in range(len(dataset))])


def group_indices_by_label(labels: torch.Tensor, num_labels: int, seed: int) -> List[List[int]]:
    """
    Groups dataset indices by label and shuffles indices within each label.

    Uses a seeded random generator to shuffle the indices for reproducibility.
    Returns a list of lists, where each inner list contains shuffled indices of samples for one label.
    """
    generator = torch.Generator().manual_seed(seed)
    label_to_indices = [[] for _ in range(num_labels)]
    for i, lbl in enumerate(labels):
        label_to_indices[lbl].append(i)
    for i in range(num_labels):
        indices = torch.tensor(label_to_indices[i])
        perm = torch.randperm(len(indices), generator=generator)
        label_to_indices[i] = indices[perm].tolist()
    return label_to_indices


def split_validation_indices(label_to_indices: List[List[int]], val_fraction: float) -> Tuple[List[int], List[List[int]]]:
    """
    Splits indices per label into validation and leftover subsets.

    The validation subset contains the first `val_fraction` fraction of indices for each label.
    Returns a tuple of (validation indices list, leftover indices list per label).
    """
    val_indices = []
    leftover = []
    for indices in label_to_indices:
        n_val = int(len(indices) * val_fraction)
        val_indices.extend(indices[:n_val])
        leftover.append(indices[n_val:])
    return val_indices, leftover


def apply_label_fractions(label_indices: List[List[int]], label_fractions: Optional[Tuple[float, ...]]) -> List[List[int]]:
    """
    Applies per-label fraction subsampling to label indices.

    If `label_fractions` is None, returns original indices unchanged.
    Otherwise, truncates each label's indices to the corresponding fraction length.
    """
    if label_fractions is None:
        return label_indices
    return [indices[:int(len(indices) * frac)] for indices, frac in zip(label_indices, label_fractions)]


def downsample_to_max_total(label_indices: List[List[int]], max_total: int) -> List[List[int]]:
    """
    Downsamples each label's indices proportionally so that the total number of samples 
    does not exceed `max_total`.
    Uses largest remainder method to round up where possible.
    Returns original indices if total is already within the limit or if `max_total` is None.
    """
    total = sum(len(lst) for lst in label_indices)
    if max_total is None or total <= max_total:
        return label_indices
    scale = max_total / total
    exact_sizes = [len(indices) * max_total / total for indices in label_indices]
    floored_sizes = [int(s) for s in exact_sizes]
    remainders = [s - f for s, f in zip(exact_sizes, floored_sizes)]
    leftover = max_total - sum(floored_sizes)
    indices_sorted = sorted(enumerate(remainders), key=lambda x: -x[1])
    for i in range(leftover):
        floored_sizes[indices_sorted[i][0]] += 1
    return [indices[:k] for indices, k in zip(label_indices, floored_sizes)]


def proportional_split(label_indices: List[List[int]], split_fracs: List[float]) -> List[List[int]]:
    """
    Splits indices grouped by label into multiple subsets proportionally to `split_fracs`, 
    preserving both overall subset sizes and label distributions. 
    Handles rounding to ensure all samples are included without overlap.
    """
    num_subsets = len(split_fracs)
    total = sum(len(lst) for lst in label_indices)
    subset_sizes = [int(total * f / sum(split_fracs)) for f in split_fracs]
    for i in range(total - sum(subset_sizes)):
        subset_sizes[i % num_subsets] += 1

    subsets = [[] for _ in range(num_subsets)]
    for indices in label_indices:
        n = len(indices)
        splits = [int(n * f / sum(split_fracs)) for f in split_fracs]
        for i in range(n - sum(splits)):
            splits[i % num_subsets] += 1
        cursor = 0
        for i, size in enumerate(splits):
            subsets[i].extend(indices[cursor:cursor + size])
            cursor += size
    return subsets


def split_dataset_indices(
    dataset: Dataset,
    split_fractions: Tuple[float, ...],
    seed: int,
    label_fractions: Optional[Tuple[float, ...]] = None,
    max_training_samples: Optional[int] = None,
) -> List[List[int]]:
    """
    Split a dataset into multiple subsets by indices, maintaining label balance.

    The dataset is split into non-overlapping subsets according to `split_fractions`,
    where the last subset (typically validation) always preserves the full dataset's
    label distribution without any subsampling or size restriction.

    The first N-1 subsets (typically training and others) are formed from the remaining
    samples after reserving the last subset. These subsets can be optionally
    subsampled per label using `label_fractions` and/or limited in total size by
    `max_training_samples`, while attempting to preserve label proportions.

    Args:
        dataset:
            Dataset where each item is a tuple (data, label).
            Labels must be contiguous integers from 0 to C-1.

        split_fractions:
            Tuple of floats specifying the fraction of the dataset to allocate to each subset.
            Fractions must sum to <= 1.0.
            Example: (0.7, 0.1, 0.2) splits into 70%, 10%, and 20% subsets.
            The last fraction corresponds to the validation subset, which is never subsampled.

        seed:
            Integer seed for random operations to ensure reproducibility.

        label_fractions:
            Optional tuple of length C with floats in [0.0, 1.0], specifying the fraction
            of each label to include in the training subsets (first N-1 subsets).
            If None (default), no per-label subsampling occurs and label proportions are preserved.
            Example: For 3 labels, (1.0, 0.5, 0.8) uses all samples of label 0,
                     half of label 1, and 80% of label 2 in training subsets.
            The validation subset is unaffected by this parameter.

        max_training_samples:
            Optional integer limiting the total number of samples across the training subsets (first N-1).
            If set, the subsets are proportionally downsampled to respect this limit,
            preserving label proportions after applying `label_fractions` (if any).
            The validation subset is unaffected.

    Returns:
        A list of lists of indices, one per subset, corresponding to the input fractions and constraints.
    """
    assert 0 <= sum(split_fractions) <= 1.0, "Sum of split fractions must be <= 1.0"

    labels = get_labels(dataset)
    num_labels = labels.max().item() + 1
    val_fraction = split_fractions[-1]
    train_fractions = split_fractions[:-1]

    label_to_indices = group_indices_by_label(labels, num_labels, seed)
    val_indices, leftover = split_validation_indices(label_to_indices, val_fraction)
    leftover = apply_label_fractions(leftover, label_fractions)
    leftover = downsample_to_max_total(leftover, max_training_samples)
    assert all(len(label) > 0 for label in leftover), (
        "At least one label has zero samples in the training data."
    )
    train_subsets = proportional_split(leftover, list(train_fractions))

    return train_subsets + [val_indices]