# tools

## Configure `gcloud`

List projects:

```bash
gcloud projects list

PROJECT_ID                NAME                  PROJECT_NUMBER
dd2424-455312             DD2424                1004679150008
```

Set default project:

```bash
gcloud config set project dd2424-455312
```

List instances:

```bash
gcloud compute instances list

NAME             ZONE           MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP   STATUS
deeplearning-vm  us-central1-a  n1-standard-4  true         10.128.0.2   34.72.232.36  RUNNING
```

Set default zone:
```bash
gcloud config set compute/zone us-central1-a
```

or whichever zone you prefer. You can list available zones with:

```bash
gcloud compute zones list --filter="region:(us-*)"
```

## Create VM

Create a VM instance in the zone you selected above:

```bash
./deploy-vm.sh
```

Open ports:

```bash
./configure-vm.sh
```

## Start VM

```bash
./start-vm.sh
```

## Stop VM

```bash
./stop-vm.sh
```

## Describe VM

```bash
./get-vm-ip.sh

34.72.232.36
```

```bash
./get-vm-status.sh

RUNNING
```

## Run VM

The following script:

1. Starts the VM in case it is not already started
1. SSHs into the VM
1. Starts the Jupyter Notebook server process on the VM
1. Port-forwards port 8888, so you can access http://localhost:8888 on your local machine
1. Gives you a shell
1. Upon exiting the shell, port-forwarding is stopped
1. Upon exiting the shell, you are given the option to stop the VM

```bash
./run-vm.sh
```

## Set hostname

This script tells your operating system to recognize the hostname
`deeplearning-vm` as the IP address of the VM. This way, you can use the
hostname instead of the IP address directly in scripts/configuration. This is
useful in case the IP address of your VM changes, which tends to happen every
time you restart it.

```bash
./set-hostname.sh

ping deeplearning-vm

PING deeplearning-vm (34.72.232.36) 56(84) bytes of data.
64 bytes from deeplearning-vm (34.72.232.36): icmp_seq=1 ttl=52 time=330 ms
64 bytes from deeplearning-vm (34.72.232.36): icmp_seq=2 ttl=52 time=130 ms
64 bytes from deeplearning-vm (34.72.232.36): icmp_seq=3 ttl=52 time=172 ms
```

## Delete VM

```bash
./delete-vm.sh
```
