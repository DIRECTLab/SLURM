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
