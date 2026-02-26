# Slurm Accounting (sacct) with MariaDB + slurmdbd (Ubuntu 22.04 / Slurm 21.08)

This README sets up **Slurm accounting** so commands like `sacct`, `sreport`, and `sacctmgr` work across the cluster.

Architecture:
- **MariaDB** stores accounting data
- **slurmdbd** is the daemon that talks to MariaDB
- **slurmctld** (controller) sends job accounting to slurmdbd
- **clients** (`sacct`, `sacctmgr`) query slurmdbd (NOT MariaDB directly)

> Important: `sacct` will fail with `Connection refused localhost:6819` if nodes are configured with `AccountingStorageHost=localhost`. Set it to the **controller hostname** cluster-wide.

---

## 0) Assumptions / Naming

- Controller hostname: `kenmasters`
- ClusterName in slurm.conf: `kencluster`
- OS: Ubuntu 22.04
- Slurm: 21.08.x (works similarly for newer)
- slurmdbd runs on the controller (`kenmasters`)
- MariaDB runs on the controller (simplest). You can move it later.

Paths used:
- `/etc/slurm/slurm.conf`
- `/etc/slurm/slurmdbd.conf`
- Logs: `/var/log/slurm/slurmdbd.log`, `/var/log/slurm/slurmctld.log`

---

## 1) Install packages (controller)

```bash
sudo apt-get update
sudo apt-get install -y mariadb-server slurmdbd
sudo systemctl enable --now mariadb

