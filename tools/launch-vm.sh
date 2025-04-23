#!/usr/bin/env bash

set -Eeuo pipefail

VM_NAME="deeplearning-vm"
ZONE="us-central1-a"

# Check if VM is running
VM_STATUS=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format='get(status)')

if [ "$VM_STATUS" != "RUNNING" ]; then
  echo "Starting VM..."
  gcloud compute instances start "$VM_NAME" --zone="$ZONE"
else
  echo "VM is already running."
fi

# SSH into VM with port forwarding
echo "Connecting to VM with port forwarding..."

gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- -T << 'EOF'
# Start Jupyter Notebook in the background if not already running
if ! pgrep -f "jupyter-notebook" > /dev/null; then
  echo "Starting Jupyter Notebook server..."
  nohup jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password='' > jupyter.log 2>&1 &
  sleep 2  # give it a moment to start
else
  echo "Jupyter Notebook server already running."
fi
EOF

echo "Starting port forwarding on localhost:8888..."

gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- -N -L 8888:localhost:8888 &
PORT_FWD_PID=$!

echo "Connecting interactively to the VM (CTRL+D to exit)..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE"

echo "Killing port forwarding process..."
kill "$PORT_FWD_PID" || true

# After exiting SSH session, stop the VM

read -r -p "Stop the VM ($VM_NAME)? (y/n): " STOP_VM

if [[ "$STOP_VM" =~ ^[Yy]$ ]]; then
  echo "Stopping VM..."
  gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
else
  echo "VM will remain running."
fi
