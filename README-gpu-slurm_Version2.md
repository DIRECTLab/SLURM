# Using GPUs with Slurm (batch jobs) + Adding a GPU worker node

Last updated: 2026-01-16

This README covers:
1) How to request and use GPUs in Slurm batch jobs.
2) How to add a worker node that has NVIDIA GPUs (configure Slurm GRES).

**Logging policy for this cluster (per request):**
- Always use **relative** output paths (no `/home/...`).
- Always write **stdout and stderr to the same file**.

> With relative paths, Slurm writes log files in the job’s submit directory (typically `$SLURM_SUBMIT_DIR`).

---

## 1) Using GPUs in a Slurm batch job

### 1.1 Prerequisites
On the GPU worker node (example: `frutiger`), the OS must see the GPUs:

```bash
nvidia-smi -L
ls -l /dev/nvidia0 /dev/nvidia1
```

If `nvidia-smi` fails, fix the NVIDIA driver/CUDA stack first. Slurm can only schedule GPUs it can see.

### 1.2 Minimal GPU batch script (stdout+stderr combined, relative output)
Example requesting **1 GPU**:

```bash
#!/bin/bash
#SBATCH -J gpu-test
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -w frutiger
#SBATCH --gres=gpu:1
#SBATCH -t 00:02:00

# Relative path: written to the submit directory ($SLURM_SUBMIT_DIR)
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

echo "Running on $(hostname)"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi -L
nvidia-smi

# Example: run your code
python3 python_test.py
```

Submit it:

```bash
sbatch gpu-test.sbatch
```

Request **both GPUs** on a 2-GPU node:

```bash
#SBATCH --gres=gpu:2
```

### 1.3 Token reference (for filenames)
Common Slurm filename tokens:

- `%x` = job name (from `#SBATCH -J ...`)
- `%j` = job ID

So `%x-%j.out` becomes like: `gpu-test-123.out`.

---

## 2) Adding a GPU worker node (Slurm GRES configuration)

This section configures a GPU node to report its GPUs to Slurm so you can schedule them via `--gres=gpu:<N>`.

### 2.1 Overview of required files
- Controller (slurmctld host, example: `kenmasters`):
  - `/etc/slurm/slurm.conf` must declare `GresTypes=gpu`
  - GPU nodes must advertise `Gres=gpu:<count>`

- GPU worker node (slurmd host, example: `frutiger`):
  - `/etc/slurm/slurm.conf` must match the controller’s config
  - `/etc/slurm/gres.conf` must list GPU devices (e.g. `/dev/nvidia0`, `/dev/nvidia1`)
  - `slurmd` must be running

### 2.2 On the GPU worker node: verify GPUs
On the new GPU node:

```bash
nvidia-smi -L
ls -l /dev/nvidia*
```

You should see one entry per GPU, such as `/dev/nvidia0`, `/dev/nvidia1`, etc.

### 2.3 Create `/etc/slurm/gres.conf` on the GPU worker node
Example for **2 GPUs**:

```conf
Name=gpu Type=nvidia File=/dev/nvidia0
Name=gpu Type=nvidia File=/dev/nvidia1
```

Confirm:

```bash
sudo cat /etc/slurm/gres.conf
sudo chmod 644 /etc/slurm/gres.conf
```

### 2.4 Update `/etc/slurm/slurm.conf` on the controller
On the controller (`kenmasters`), ensure:

1) Near the top:
```conf
GresTypes=gpu
```

2) The GPU node line includes the GPU count, for example:
```conf
NodeName=frutiger Sockets=1 CoresPerSocket=12 ThreadsPerCore=2 CPUs=24 RealMemory=64202 Gres=gpu:2 State=UNKNOWN
```

> Keep your real CPU/memory values; the key part is `Gres=gpu:2`.

### 2.5 Ensure `slurm.conf` is identical on all nodes
A common failure mode is config drift (different `slurm.conf` on controller vs node).

Verify hashes:

On controller:
```bash
sudo sha256sum /etc/slurm/slurm.conf
```

On GPU node:
```bash
sudo sha256sum /etc/slurm/slurm.conf
```

They should match. If not, copy the controller’s `slurm.conf` to the node and restart `slurmd`.

### 2.6 Restart Slurm daemons
On the controller:
```bash
sudo systemctl restart slurmctld
```

On the GPU worker node:
```bash
sudo systemctl restart slurmd
```

### 2.7 Verify the controller sees the GPUs
On the controller:

```bash
scontrol show node frutiger | egrep -i "State=|Reason=|Gres=|CfgTRES=|AllocTRES="
sinfo -N -o "%N %t %G %E"
```

You want to see `Gres=gpu:2` and the node in `IDLE` (not `DRAIN`/`INVALID`).

### 2.8 If the node is DRAIN/INVALID with a GRES error
Example error:
- `Reason=gres/gpu count reported lower than configured (0 < 2)`

This means Slurm is configured for 2 GPUs, but `slurmd` is reporting 0. Check:

On the GPU node:
```bash
nvidia-smi -L
sudo cat /etc/slurm/gres.conf
sudo systemctl status slurmd --no-pager
```

Also ensure `slurm.conf` matches exactly across controller/node.

To clear a drain after fixing the cause, on controller:
```bash
sudo scontrol update NodeName=frutiger State=UNDRAIN Reason=""
```

---

## 3) Quick “does GPU scheduling work?” test

### 3.1 Interactive test (1 GPU)
```bash
srun -p debug -w frutiger --gres=gpu:1 --pty bash -lc 'hostname; nvidia-smi -L; echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES'
```

### 3.2 Batch test (1 GPU)
Create `gpu-smoke.sbatch`:

```bash
#!/bin/bash
#SBATCH -J gpu-smoke
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -w frutiger
#SBATCH --gres=gpu:1
#SBATCH -t 00:01:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

hostname
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
```

Submit:
```bash
sbatch gpu-smoke.sbatch
```

Then view output in the submit directory:
```bash
cat gpu-smoke-<jobid>.out
```