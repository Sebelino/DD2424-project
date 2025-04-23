#!/usr/bin/env python3

import os
import tarfile
import urllib.request

from tqdm import tqdm

# Constants
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
DATA_DIR = "data/oxford-iiit-pet"
IMAGES_TAR = os.path.join(DATA_DIR, "images.tar.gz")
ANNOTATIONS_TAR = os.path.join(DATA_DIR, "annotations.tar.gz")


def download(url, dest_path):
    # Progress bar class for reporthook
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    print(f"Downloading {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(dest_path)) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
    print(f"Saved to {dest_path}")


def extract(tar_path, extract_to):
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted to {extract_to}")


def maybe_download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download images.tar.gz if not present
    if not os.path.exists(IMAGES_TAR):
        download(DATA_URL, IMAGES_TAR)
    else:
        print(f"{IMAGES_TAR} already exists. Skipping download.")

    # Download annotations.tar.gz if not present
    if not os.path.exists(ANNOTATIONS_TAR):
        download(ANNOTATIONS_URL, ANNOTATIONS_TAR)
    else:
        print(f"{ANNOTATIONS_TAR} already exists. Skipping download.")

    # Extract images/ if not already extracted
    images_dir = os.path.join(DATA_DIR, "images")
    if not os.path.exists(images_dir):
        extract(IMAGES_TAR, DATA_DIR)
    else:
        print(f"{images_dir} already extracted. Skipping.")

    # Extract annotations/ if not already extracted
    annotations_dir = os.path.join(DATA_DIR, "annotations")
    if not os.path.exists(annotations_dir):
        extract(ANNOTATIONS_TAR, DATA_DIR)
    else:
        print(f"{annotations_dir} already extracted. Skipping.")


import shutil
import random
from pathlib import Path


def copy_files(files_dir, filenames, dst_dir):
    for filename, label in filenames:
        src = files_dir / f"{filename}.jpg"
        dst = dst_dir / label / f"{filename}.jpg"
        if dst.exists():
            return  # Don't copy if it already exists
        if src.exists():
            shutil.copy(src, dst)


def make_partitioned_dataset(dataset_dir: str, output_dir):
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 1 - train_ratio - val_ratio

    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    annotations_file = dataset_dir / "annotations" / "list.txt"

    output_dir_already_exists = os.path.exists(output_dir)

    # Ensure output directories exist
    for split in ['train', 'val', 'test']:
        for cls in ['cat', 'dog']:
            subsubdir = output_dir / split / cls
            subsubdir.mkdir(parents=True, exist_ok=True)

    # Parse annotations to get filenames and binary labels
    with open(annotations_file) as f:
        lines = f.readlines()[6:]  # Skip header

    file_info = []
    for line in lines:
        parts = line.strip().split()
        filename, _, species_num_str, _ = parts
        species_num = int(species_num_str)  # 1 = Cat, 2 = Dog
        label = 'cat' if species_num == 1 else 'dog'
        file_info.append((filename, label))

    # Shuffle and split data
    random.shuffle(file_info)
    num_total = len(file_info)
    num_train = int(train_ratio * num_total)
    num_val = int(val_ratio * num_total)

    if output_dir_already_exists:
        print(f"Output directory {output_dir} already exists.")
        num_test = num_total - num_train - num_val
        return num_train, num_val, num_test

    train_files = file_info[:num_train]
    val_files = file_info[num_train:num_train + num_val]
    test_files = file_info[num_train + num_val:]

    # Copy to respective folders
    copy_files(images_dir, train_files, output_dir / "train")
    copy_files(images_dir, val_files, output_dir / "val")
    copy_files(images_dir, test_files, output_dir / "test")

    print(f"Dataset partitioned into: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images.")
    print(f"Output directory: {output_dir.resolve()}")

    return len(train_files), len(val_files), len(test_files)


if __name__ == "__main__":
    maybe_download_and_extract()
