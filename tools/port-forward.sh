#!/usr/bin/env bash

VM_NAME="deeplearning-vm"
PORT=8888

gcloud compute ssh "$VM_NAME" -- -L "${PORT}:localhost:${PORT}"
