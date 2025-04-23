#!/usr/bin/env bash

set -Eeuo pipefail

extra_flags="${1:-}"

INSTANCE_NAME="deeplearning-vm"
MACHINE_TYPE="n1-standard-4"  # 4 vCPUs, 15 GB RAM
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-latest-gpu"  # Preinstalled with PyTorch, CUDA
IMAGE_PROJECT="deeplearning-platform-release"

# Create the instance
# shellcheck disable=SC2086
gcloud compute instances create "$INSTANCE_NAME" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator=type="$GPU_TYPE",count="$GPU_COUNT" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --metadata="install-nvidia-driver=True" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  $extra_flags

gcloud compute instances list
