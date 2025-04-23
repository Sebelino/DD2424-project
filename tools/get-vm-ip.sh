#!/usr/bin/env bash

set -Eeuo pipefail

VM_NAME="deeplearning-vm"

gcloud compute instances describe "$VM_NAME" --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
