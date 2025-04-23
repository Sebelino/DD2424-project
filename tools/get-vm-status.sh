#!/usr/bin/env bash

gcloud compute instances describe deeplearning-vm --zone=us-central1-a --format='get(status)'
