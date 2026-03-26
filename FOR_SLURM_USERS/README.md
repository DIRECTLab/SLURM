# SLURM Cluster User Guide

Welcome to the SLURM Cluster. This comprehensive guide covers everything from logging in, managing your environment, submitting jobs, and storing large data, to complete end-to-end examples like training PyTorch models.

---

## Table of Contents

1. [Getting Connected (SSH & VS Code)](#1-getting-connected-ssh--vs-code)
2. [Basic SLURM Usage (Quickstart)](#2-basic-slurm-usage-quickstart)
3. [SSH Direct Access to Compute Nodes](#3-ssh-direct-access-to-compute-nodes)
4. [Environment Management (Conda)](#4-environment-management-conda)
5. [Storage and Large Data](#5-storage-and-large-data)
6. [Example: Training MNIST with PyTorch on GPUs](#6-example-training-mnist-with-pytorch-on-gpus)

---

## 1. Getting Connected (SSH & VS Code)

### Logging into the Login Node

To interact with the cluster, you must first SSH into the **login node**. Your SSH public key must be registered by an administrator, and you need the private key on your local machine.

- **Server IP**: `129.123.61.22`
- **Username**: Your provided username (or A-number for student accounts)
- **Port**: `4242`

**Basic Connection**
```bash
ssh -p 4242 username@129.123.61.22
```

*(On your first login, type `yes` to accept the host key fingerprint.)*

### Configuring SSH Aliases and ProxyJump

To avoid typing the IP and port every time, and to prepare for direct compute-node access later, add the following to your local SSH config file (`~/.ssh/config`):

```ssh-config
# The main login node
Host clusteruser
    HostName 129.123.61.22
    User username
    Port 4242
    IdentityFile ~/.ssh/id_ed25519  # Or your specific private key
    ForwardAgent yes

# Example Compute Nodes (ProxyJump through the login node)
Host frutiger
    HostName frutiger
    User username
    ProxyJump clusteruser
    ForwardAgent yes

Host chunli
    HostName chunli
    User username
    ProxyJump clusteruser
    ForwardAgent yes
```
*Be sure to replace `username` with your actual username above.* 

Once configured, you can simply run:
```bash
ssh clusteruser
```

**File Transfers (`scp`)**
You can use the new alias to easily transfer files:
```bash
# Upload a file
scp local_file clusteruser:/home/username/

# Download a file
scp clusteruser:/home/username/remote_file ./
```

### VS Code Remote SSH Setup

You can fully interact with the cluster using VS Code's Remote - SSH extension.

1. **Install Expansion:** Open Extensions (Ctrl+Shift+X) and install **Remote - SSH** by Microsoft.
2. **Access Targets:** Open the Command Palette (Ctrl+Shift+P) -> `Remote-SSH: Connect to Host...`
3. Select `clusteruser` to connect to the login node. **This is usually all you need to edit your files, submit jobs, and check results!**
4. If you need to run heavy graphical extensions (like Jupyter Notebooks) that require compute resources, you can select `frutiger` or `chunli` to connect *directly* to a compute node. (Note: **You must have an active job allocation on that node first**—see Section 3).
5. Add a folder to your workspace via File -> Open Folder.

### SSH Troubleshooting
- **Permission Denied (publickey):** Verify your username, private key location, ensure the host key exactly matches what was given to the admin, and `chmod 600 ~/.ssh/id_ed25519`. 
- **Connection Refused:** Ensure the server IP is exactly `129.123.61.22`, and you are connected to the campus eduroam wifi or university VPN.

---

## 2. Basic SLURM Usage (Quickstart)

### Checking Cluster Status

Check Node states and partition availability:
```bash
sinfo
scontrol show nodes
```
Check your currently running/queued subjobs:
```bash
squeue -u $USER
```

### Running Interactive Jobs (`srun`)

Run a single command on an available compute node:
```bash
srun -N1 -n1 hostname
```

Drop into a general terminal session on an available compute node:
```bash
srun -p debug -N1 -n1 --pty bash
```

Drop into a login shell (loads `/etc/profile`, modules, etc. reliably):
```bash
srun -p debug -N1 -n1 --pty bash -l
```

### Submitting Batch Scripts (`sbatch`)

For long-running tasks, write an `sbatch` script. Example `slurm_test.slurm`:

```bash
#!/bin/bash
#SBATCH -J demo-job
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -w frutiger
#SBATCH -t 00:02:00
#SBATCH -o %x-%j.out

echo "HOST=$(hostname)"
echo "DATE=$(date)"
id
sleep 30
```

Submit it:
```bash
sbatch slurm_test.slurm
```
Cancel it:
```bash
scancel <jobid>
```
**NOTE: The default job time is 8 hours, if you need more time, specify it with `--time=HH:MM:SS` or `-t HH:MM:SS`.**

### Requesting GPUs
If nodes are configured with GRES GPUs, request them. 
For NVIDIA: `#SBATCH --gres=gpu:1`
For AMD: `#SBATCH --gres=amdgpu:1`

### Common Flags Cheat-Sheet
- Choose partition: `--partition=debug` (or `-p debug`)
- Walltime limit: `--time=HH:MM:SS` (or `-t HH:MM:SS`)
- Number of Nodes: `--nodes=N` (or `-N N`)
- Number of Tasks: `--ntasks=N` (or `-n N`)
- CPUs per task: `--cpus-per-task=N` (or `-c N`)
- Memory allowed: `--mem=4G`
- Output log file: `--output=%x-%j.out` (or `-o %x-%j.out`)

---

## 3. SSH Direct Access to Compute Nodes

While `srun` works great for terminals, sometimes you want to SSH directly into a compute worker node so you can launch VS Code, utilize external debugging applications, etc.

**Security Policy:** Compute nodes are firewalled. You can only SSH to them through the login node (`ProxyJump`), **AND you must have an active SLURM job running on that target node.**

### Step 1: Allocate Resources (`salloc`)

Allocate resources to start a background interactive job assignment. Most of the time, you should let SLURM automatically select any available node by removing the `-w` flag:
```bash
# General CPU allocation (4 hours, 8 CPUs) on *any* available node
salloc -t 04:00:00 -c 8
```

If you specifically need a particular machine (e.g. `frutiger`), you can target it:
```bash
# General CPU allocation (4 hours, 8 CPUs) specifically on frutiger
salloc -w frutiger -t 04:00:00 -c 8

# GPU targeted allocation (24 hours, 16 CPUs, with advanced Memory flags)
salloc -w chunli -t 1-00:00:00 -c 16 --mem=32G
```
*(Do not set `--gpus=1` unless checking exact parameters, as locking a GPU blocks all other GPU users for the node!)*

### Step 2: Connect via SSH / VS Code

Once `salloc` states the allocation is granted:
```bash
ssh frutiger
```
Or simply use the `frutiger` target in your VS Code Remote extension. 

### Allocation Lifecycle
- The node remains SSH-accessible only until your `salloc` time limit expires (`-t`) or you `exit` / `Ctrl+C` the `salloc` terminal window. 
- Try to release jobs early when done to free up resources.
- If you receive *"Access denied: you have no active RUNNING Slurm jobs"*, verify your allocation works via `squeue -u $USER`.

---

## 4. Environment Management (Conda)

This cluster provides a shared instance of Miniforge located at `/home/software/miniforge3`. Please **do not** install packages into the shared base environment. Use user-specific environments.

### 1) Sourcing Conda
Initialize Conda on your terminal or `.bashrc`:
```bash
source /home/software/miniforge3/etc/profile.d/conda.sh
```

### 2) Creating and using local Environments
Creates the environment natively inside `~/.conda/envs`:
```bash
conda create -n myenv python=3.11
conda activate myenv
conda install numpy scipy pandas   # or use `mamba install ...`
```

### 3) Conda in SLURM Batch Scripts
Important: **Do not assume** that your currently-activated conda environment will carry over to a Slurm batch job automatically! Always activate it securely inside the Slurm script itself:

```bash
#!/bin/bash
#SBATCH -J conda-test

# Initialize conda for non-interactive bash environments
source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate myenv

python -c "import sys; print(sys.executable)"
```

---

## 5. Storage and Large Data

To ensure high performance and avoid filling up the shared `/home` NFS volume, jobs that read or write large data files should use the cluster's local Scratch storage.

- **On Worker Nodes (During jobs):** Output temporary datasets directly to `/scratch/$USER`.
- **On the Login Node:** Access this saved data remotely via `/scratch/<worker_node>/$USER` (e.g., `/scratch/frutiger/$USER`).

> **⚠️ Warning:** Scratch storage is strictly meant for temporary job data and working output. Always remember to copy or move your important final results back to your `/home/$USER` directory once your jobs finish, as scratch spaces generally offer no backups and could be purged!

**Batch Example:**
```bash
#!/bin/bash
#SBATCH -J large-data-job
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o %x-%j.out

# Write large temporary or output data to the node's local scratch
OUTPUT_FILE="/scratch/$USER/my_large_output_${SLURM_JOB_ID}.dat"

echo "Generating large data on node $(hostname)..."
dd if=/dev/urandom of="$OUTPUT_FILE" bs=1M count=100

echo "Job finished. Access this file from the login node at:"
echo "/scratch/$(hostname)/$USER/my_large_output_${SLURM_JOB_ID}.dat"
```

---

## 6. Example: Training MNIST with PyTorch on GPUs

This section ties everything together to execute a PyTorch ML training pipeline.

### Step 1. Prepare Environment
```bash
source /home/software/miniforge3/etc/profile.d/conda.sh
conda create -n test_mnist python=3.10
conda activate test_mnist
pip install torch torchvision
```

### Step 2. Create the Python Training File
Create `train_mnist_detector.py`:
```python
import argparse, os, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LightMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--out-dir", type=str, default="./runs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dl = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=tfm), batch_size=128, shuffle=True)

    model = LightMNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        print(f"Epoch {epoch} complete.")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "mnist.pt"))

if __name__ == "__main__":
    main()
```

### Step 3. Create Batch Script
Create `train_mnist.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mnist_smoketest
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out

set -euo pipefail

# Ensure environment variables are loaded
source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate test_mnist

# Basic checks
python -V
python -c "import torch; print('CUDA built:', torch.cuda.is_available())"

# Execute
srun python train_mnist_detector.py --epochs 2 --out-dir /scratch/$USER/runs
```

### Step 4. Run Job
```bash
sbatch train_mnist.slurm
```
Monitor with `squeue -u $USER` and read your `%x_%j.out` log file to observe the network compiling and saving successfully.

**Once the job finishes, remember you can retrieve your trained model checkpoint right from the login node by checking the NFS-mounted directory: `/scratch/<worker_node_it_ran_on>/$USER/runs/mnist.pt`!**
