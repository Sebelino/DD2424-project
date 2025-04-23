#!/usr/bin/env bash

set -Eeuo pipefail

VM_NAME="deeplearning-vm"
PORT=8888

# Check if VM is running
vm_status=$(gcloud compute instances describe "$VM_NAME" --format='get(status)')

if [ "$vm_status" != "RUNNING" ]; then
  echo "Starting VM..."
  gcloud compute instances start "$VM_NAME"
else
  echo "VM is already running."
fi

# SSH into VM with port forwarding
echo "Connecting to VM with port forwarding..."

gcloud compute ssh "$VM_NAME" -- -T << 'EOF'
# Start Jupyter Notebook in the background if not already running
if ! pgrep -f "jupyter-notebook" > /dev/null; then
  echo "Starting Jupyter Notebook server..."
  nohup jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password='' > jupyter.log 2>&1 &
  sleep 2  # give it a moment to start
else
  echo "Jupyter Notebook server already running."
fi
EOF

echo "Starting port forwarding on localhost:${PORT}..."

gcloud compute ssh "$VM_NAME" -- -N -L ${PORT}:localhost:${PORT} &
PORT_FWD_PID=$!

echo "Jupyter Notebook server should now be accessible at http://localhost:${PORT}."
echo "Connecting interactively to the VM..."
gcloud compute ssh "$VM_NAME"

echo "Killing port forwarding process..."
kill "$PORT_FWD_PID" || true

# After exiting SSH session, stop the VM

read -r -p "Stop the VM ($VM_NAME)? (y/n): " STOP_VM

if [[ "$STOP_VM" =~ ^[Yy]$ ]]; then
  echo "Stopping VM..."
  gcloud compute instances stop "$VM_NAME"
else
  echo "VM will remain running."
fi
