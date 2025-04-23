#!/usr/bin/env bash

set -Eeuo pipefail

VM_NAME="deeplearning-vm"
PORT=8888

gcloud compute firewall-rules create allow-jupyter \
    --allow "tcp:${PORT}" \
    --source-ranges=0.0.0.0/0 \
    --target-tags=jupyter-server \
    --description="Allow Jupyter Notebook traffic on port ${PORT}"

gcloud compute instances add-metadata "$VM_NAME" \
  --metadata=proxy-mode=project_editors
