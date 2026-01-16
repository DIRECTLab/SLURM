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

## 3) Submit a batch script (`sbatch`)

Create a script:

```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --partition=debug

echo "Running on: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
sleep 30
EOF

chmod +x job.sh
sbatch job.sh
```

Monitor:

```bash
squeue -u $USER
```

View output:

```bash
cat slurm-<jobid>.out
```

Cancel a job:

```bash
scancel <jobid>
```

---

## 4) Requesting GPUs (only if configured)

If your cluster is configured with GRES GPUs, you can request them like:

```bash
srun --gres=gpu:1 -N1 -n1 nvidia-smi
```

Notes:
- Intel GPUs typically use `/dev/dri/*`; they require GRES config to schedule properly.
- The exact `--gres=` name depends on your `gres.conf`.

---

## 5) Common flags cheat-sheet

- Choose partition: `--partition=debug`
- Walltime: `--time=HH:MM:SS`
- Nodes: `--nodes=N` (or `-N N`)
- Tasks: `--ntasks=N` (or `-n N`)
- CPUs per task: `--cpus-per-task=N` (or `-c N`)
- Memory: `--mem=4G`
- Job name: `--job-name=name`
- Output file: `--output=slurm-%j.out`

---

## 6) Debugging jobs

Why is a job pending?

```bash
squeue -j <jobid> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
scontrol show job <jobid>
```

See node details:

```bash
scontrol show node <nodename>
```

---