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


if __name__ == "__main__":
    maybe_download_and_extract()
