#!/usr/bin/env bash

set -Eeuo pipefail

VM_NAME="deeplearning-vm"

gcloud compute instances delete "$VM_NAME" --quiet
