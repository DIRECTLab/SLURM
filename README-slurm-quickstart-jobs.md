
# SLURM Quickstart: Submitting Jobs

This is a practical cheat-sheet for running jobs on a working SLURM cluster.

---

## 1) Check cluster status

```bash
sinfo
scontrol show nodes
```

Check your account/job view:

```bash
squeue -u $USER
```

---

## 2) Run an interactive job (`srun`)

Run a single command on one node:

```bash
srun -N1 -n1 hostname
```

Run a shell on an allocated node:

```bash
srun -N1 -n1 --pty bash
```

Request CPU resources:

```bash
srun -N1 -n1 -c 4 --mem=4G --time=00:10:00 --pty bash
```

---

## 3) Drop into a shell on a specific node (interactive allocation)

This is the standard way to “SSH into” a compute node **through Slurm** (recommended vs direct SSH).

### 3.1 Drop into *any* available node in a partition
```bash
srun -p debug -N1 -n1 --pty bash
```

### 3.2 Drop into a specific node (example: `frutiger`)
```bash
srun -p debug -N1 -n1 -w frutiger --pty bash
```

### 3.3 Drop into a node with a GPU allocated (if configured)
```bash
srun -p debug -N1 -n1 --gres=gpu:1 --pty bash
```

Pin GPU shell to a specific node:
```bash
srun -p debug -N1 -n1 -w frutiger --gres=gpu:1 --pty bash
```

### 3.4 Start a *login shell* (loads `/etc/profile`, modules, etc.)
If your cluster uses `/etc/profile.d/*` (or environment modules) and you want that loaded reliably:

```bash
srun -p debug -N1 -n1 --pty bash -l
```

### 3.5 Exit the allocation
Just exit the shell:
```bash
exit
```

---

## 4) Submit a batch script (`sbatch`)

Create a script: `slurm_test.slurm`

```bash
#!/bin/bash
#SBATCH -J out-test
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -w frutiger
#SBATCH -t 00:02:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out

echo "HOST=$(hostname)"
echo "DATE=$(date)"
id
sleep 30
```

Submit the script:

```bash
sbatch slurm_test.slurm
```

Monitor:

```bash
squeue -u $USER
```

View output (relative output path: written to submit directory):

```bash
cat out-test-<jobid>.out
```

Cancel a job:

```bash
scancel <jobid>
```

---

## 5) Requesting GPUs (only if configured)

If your cluster is configured with GRES GPUs, you can request them like:

```bash
srun --gres=gpu:1 -N1 -n1 nvidia-smi
```

Notes:
- Intel GPUs typically use `/dev/dri/*`; they require GRES config to schedule properly.
- The exact `--gres=` name depends on your `gres.conf`.

---

## 6) Common flags cheat-sheet

- Choose partition: `--partition=debug` (or `-p debug`)
- Walltime: `--time=HH:MM:SS` (or `-t HH:MM:SS`)
- Nodes: `--nodes=N` (or `-N N`)
- Tasks: `--ntasks=N` (or `-n N`)
- CPUs per task: `--cpus-per-task=N` (or `-c N`)
- Memory: `--mem=4G`
- Job name: `--job-name=name` (or `-J name`)
- Output file: `--output=%x-%j.out` (or `-o %x-%j.out`)
- Error file: `--error=%x-%j.out` (or `-e %x-%j.out`)

---

## 7) Debugging jobs

Why is a job pending?

```bash
squeue -j <jobid> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
scontrol show job <jobid>
```

See node details:

```bash
scontrol show node <nodename>
```