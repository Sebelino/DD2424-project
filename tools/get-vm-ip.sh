#!/usr/bin/env bash

set -Eeuo pipefail

gcloud compute instances describe deeplearning-vm --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
