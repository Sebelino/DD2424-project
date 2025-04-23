#!/usr/bin/env bash

set -Eeuo pipefail

INSTANCE_NAME="deeplearning-vm"

gcloud compute instances start "$INSTANCE_NAME"
