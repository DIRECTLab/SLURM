# Setting up a SLURM Controller Node (Single-Node Controller + Compute)

This guide sets up a SLURM controller (`slurmctld`) on a machine that can also run jobs locally via `slurmd`.

Tested notes based on a simple lab setup (no `slurmdbd` required).

---

## 1) Install packages

On the controller node (example hostname: `kenmasters`):

```bash
sudo apt update
sudo apt -y install munge slurm-wlm slurmctld slurmd
```

Enable and start Munge:

```bash
sudo systemctl enable --now munge
sudo systemctl status munge --no-pager -l
```

Quick Munge test:

```bash
munge -n | unmunge
```

---

## 2) Create SLURM directories and permissions

```bash
sudo mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
sudo chown -R slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
sudo chmod 0755 /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
```

---

## 3) Configure `/etc/slurm/slurm.conf`

Edit:

```bash
sudo vim /etc/slurm/slurm.conf
```

Minimum recommended fields (example):

```conf
ClusterName=kencluster
SlurmctldHost=kenmasters

SlurmUser=slurm
SlurmdUser=root

StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd

SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log

# Auth via Munge
AuthType=auth/munge

# For older Slurm on cgroup v2 systems, linuxproc is often easiest:
ProctrackType=proctrack/linuxproc

# Scheduler / selection defaults (common)
SelectType=select/cons_tres
SelectTypeParameters=CR_Core

# Node + partition (fill with your actual topology)
NodeName=kenmasters Sockets=2 CoresPerSocket=18 ThreadsPerCore=2 CPUs=72 RealMemory=128000 State=UNKNOWN
PartitionName=debug Nodes=kenmasters Default=YES MaxTime=INFINITE State=UP
```

### Tip: generate your NodeName line automatically
Use:

```bash
sudo slurmd -C
```

Then paste the resulting `NodeName=...` line into `slurm.conf` (adjust `NodeName=` to your hostname if needed).

---

## 4) Start services (start slurmd first, then slurmctld)

```bash
sudo systemctl enable --now slurmd
sudo systemctl enable --now slurmctld
```

Check status:

```bash
sudo systemctl status slurmd --no-pager -l
sudo systemctl status slurmctld --no-pager -l
```

---

## 5) Verify cluster health

```bash
sinfo
scontrol show node kenmasters
```

Expected: node shows `idle` in the partition.

If node shows `UNKNOWN`:
- check topology mismatch vs hardware (`slurmd -C`)
- check `slurmd` logs: `/var/log/slurm/slurmd.log`
- check Munge: `munge -n | unmunge`
- check name resolution: `getent hosts <hostname>`

---