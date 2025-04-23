#!/usr/bin/env bash

set -Eeuo pipefail

vm_type="${1:-}"
extra_flags="${2:-}"

VM_NAME="deeplearning-vm"

if [[ "$vm_type" == "T4" ]]; then
  echo "🖥️ Creating a T4 instance..."
  GPU_TYPE="nvidia-tesla-t4"
  MACHINE_TYPE="n1-standard-4" # 4 vCPUs, 15 GB RAM
elif [[ "$vm_type" == "L4" ]]; then
  echo "🖥️ Creating a L4 instance..."
  GPU_TYPE="nvidia-l4" # Faster than T4, higher availabilty, expensive
  MACHINE_TYPE="g2-standard-4" # Compatible with L4 GPU
else
  echo "❌ Invalid VM type: '${vm_type}'. Use 'T4' or 'L4'." >&2
  echo "Usage: ./deploy-vm.sh <T4|L4> [extra_flags]" >&2
  echo "Example: ./deploy-vm.sh L4" >&2
  exit 1
fi

GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-latest-gpu"  # Preinstalled with PyTorch, CUDA
IMAGE_PROJECT="deeplearning-platform-release"

# List of zones to try
ZONES=(
  "us-central1-a"
  "us-central1-b"
  "us-central1-c"
  "us-central1-f"
  "us-west1-b"
  "us-west1-c"
  "us-west1-a"
)

if gcloud compute instances describe "$VM_NAME" &>/dev/null; then
  echo "Instance $VM_NAME already exists. Exiting."
  exit 1
fi

for ZONE in "${ZONES[@]}"; do
  echo "🔨Attempting to create instance in zone: $ZONE"

  # shellcheck disable=SC2086
  if gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator=type="$GPU_TYPE",count="$GPU_COUNT" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    $extra_flags; then
    echo "🌟Instance successfully created in zone: $ZONE"
    gcloud compute instances list
    exit 0
  else
    echo "🔥Failed to create instance in zone: $ZONE. Trying next zone..."
  fi
done

echo "💀All zones exhausted. Could not create instance."
exit 1
