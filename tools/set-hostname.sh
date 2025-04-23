#!/usr/bin/env bash

set -Eeuo pipefail

HOSTS_PATH="/etc/hosts"
VM_NAME="deeplearning-vm"

ip_address="$(gcloud compute instances describe "$VM_NAME" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

sudo sed -i '/deeplearning-vm/d' "$HOSTS_PATH"

echo "${ip_address} ${VM_NAME}" | sudo tee -a "$HOSTS_PATH"
echo "$HOSTS_PATH updated."
