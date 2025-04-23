#!/usr/bin/env bash

set -Eeuo pipefail

VM_NAME="deeplearning-vm"
PORT=8888
FIREWALL_RULE="allow-jupyter"

# Check if the firewall rule exists
if ! gcloud compute firewall-rules describe "$FIREWALL_RULE" &>/dev/null; then
  echo "Creating firewall rule: $FIREWALL_RULE"
  gcloud compute firewall-rules create "$FIREWALL_RULE" \
      --allow "tcp:${PORT}" \
      --source-ranges=0.0.0.0/0 \
      --target-tags=jupyter-server \
      --description="Allow Jupyter Notebook traffic on port ${PORT}"
else
  echo "Firewall rule $FIREWALL_RULE already exists. Skipping creation."
fi

echo "Add project_editors metadata to $VM_NAME..."
gcloud compute instances add-metadata "$VM_NAME" \
  --metadata=proxy-mode=project_editors
