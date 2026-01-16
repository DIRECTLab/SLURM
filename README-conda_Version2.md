# Conda / Miniforge usage on this cluster

Last updated: 2026-01-16

This cluster provides a shared Miniforge install located at:

- `/home/software/miniforge3`

Because `/home` is shared across nodes, the binaries are available everywhere. Each node must still add Miniforge to `PATH` (typically via `/etc/profile.d/miniforge.sh`).

---

## 1) Getting Conda on your PATH

If the admins have configured `/etc/profile.d/miniforge.sh` on the node you are on, then after logging in you should have:

```bash
which conda
conda --version
```

If `conda` is not found, you can always use the full path:

```bash
/home/software/miniforge3/bin/conda --version
```

Or temporarily add it for your current shell:

```bash
export PATH=/home/software/miniforge3/bin:$PATH
```

---

## 2) Recommended: create per-user environments

Do **not** install packages into the shared base environment. Instead, create environments under your own home directory.

Create an environment:

```bash
conda create -n myenv python=3.11
```

Activate it (interactive shell):

```bash
source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate myenv
```

Confirm:

```bash
which python
python --version
```

Deactivate:

```bash
conda deactivate
```

---

## 3) Installing packages

Examples:

```bash
conda install numpy scipy pandas
```

If `mamba` is available (often faster):

```bash
mamba install numpy scipy pandas
```

---

## 4) Listing and removing environments

List envs:

```bash
conda env list
```

Remove an env:

```bash
conda env remove -n myenv
```

---

## 5) Using Conda inside Slurm jobs (batch)

Important: **Do not assume** that your currently-activated conda environment will be active inside a Slurm batch job.
Always activate (or `conda run`) inside the job script.

### 5.1 Preferred (simple): `conda run`
Example `job.sbatch`:

```bash
#!/bin/bash
#SBATCH -J conda-test
#SBATCH -p debug
#SBATCH -t 00:02:00

# Relative output path; stdout and stderr to the same file
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

export PATH=/home/software/miniforge3/bin:$PATH

conda run -n myenv python -c "import sys; print(sys.executable); print(sys.version)"
```

Submit:

```bash
sbatch job.sbatch
```

### 5.2 Alternative: activate the env in the job script
Use this if you need activation semantics (e.g., environment variables set by activation scripts):

```bash
#!/bin/bash
#SBATCH -J conda-activate-test
#SBATCH -p debug
#SBATCH -t 00:02:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

export PATH=/home/software/miniforge3/bin:$PATH

# Initialize conda for non-interactive shells
source /home/software/miniforge3/etc/profile.d/conda.sh

conda activate myenv

python -c "import sys; print(sys.executable); print(sys.version)"
```

---

## 6) Reproducible environments (YAML)

Export your env:

```bash
conda env export -n myenv > myenv.yml
```

Create from YAML:

```bash
conda env create -f myenv.yml
```

---

## 7) Troubleshooting

### 7.1 `conda: command not found`
Run:

```bash
export PATH=/home/software/miniforge3/bin:$PATH
which conda
```

If that works, the node is missing `/etc/profile.d/miniforge.sh` (admins can add it).

### 7.2 Activation fails in a Slurm job
Use `conda run -n <env> ...` or ensure your script includes:

```bash
source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate <env>
```

### 7.3 Where are my environments stored?
By default, envs are typically created under your home directory (e.g., `~/.conda/envs`) unless you configure otherwise.

To see conda config:

```bash
conda config --show
```

## 8) Training MNIST with SLURM

### 8.1 Set up a conda environment

Run:

```
conda create -n test_mnist python=3.10
conda activate test_mnist
pip install torch torchvision
```

### 8.2 train_mnist.slurm

```bash
#!/bin/bash
#SBATCH --job-name=mnist_smoketest
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

# export PATH=/home/software/miniforge3/bin:$PATH

set -euo pipefail

mkdir -p logs

echo "=== SLURM INFO ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES-<unset>}"
echo "==================="

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate test_mnist

# Basic sanity checks
python -V
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda devices:', torch.cuda.device_count())"

# Run the training (srun is good practice under Slurm)
srun python train_mnist_detector.py \
  --epochs 2 \
  --batch-size 128 \
  --num-workers 2 \
  --data-dir ./data \
  --out-dir ./runs

```

### 8.3 train_mnist_detector.py

```python
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LightMNISTNet(nn.Module):
    """Small CNN: enough to verify training + GPU without heavy compute."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 14x14
            nn.Conv2d(16, 32, 3, padding=1), # 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./runs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    if device.type == "cuda":
        print(f"[info] cuda device count={torch.cuda.device_count()}")
        print(f"[info] cuda device name={torch.cuda.get_device_name(0)}")

    # Data
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model / loss / optim
    model = LightMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

        acc = evaluate(model, test_loader, device)
        dt = time.time() - t0
        print(f"[epoch {epoch}/{args.epochs}] loss={running_loss/len(train_loader):.4f}  acc={acc*100:.2f}%  time={dt:.1f}s")

    # Save a small checkpoint as an I/O test
    ckpt_path = os.path.join(args.out_dir, "mnist_smoketest.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "device": str(device),
        },
        ckpt_path,
    )
    print(f"[done] saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

```

### 8.4 Run the job

```bash
sbatch train_mnist.slurm
```

You should see something like this in the logs

```text
=== SLURM INFO ===
Job ID: 52
Node list: frutiger
CPUs per task: 4
CUDA_VISIBLE_DEVICES: 0
===================
Python 3.10.19
torch: 2.9.1+cu128
cuda available: True
cuda devices: 1

0.3%
0.7%
1.0%
1.3%
1.7%
2.0%
2.3%
2.6%
3.0%
3.3%
```