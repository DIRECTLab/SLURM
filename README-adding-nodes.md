# Adding New Compute Nodes to a SLURM Cluster

This guide adds a new compute node (worker) to an existing SLURM controller.

Assumptions:
- Controller host: `kenmasters`
- New worker host: `frutiger` (replace with your worker hostname)
- Static IPs are already configured
- You are using Munge authentication

---

## 1) Ensure name resolution works on BOTH machines

On **both** nodes, ensure `/etc/hosts` has both entries (or DNS is configured):

```text
<controller-ip>  kenmasters
<worker-ip>      frutiger
```

Verify from controller:

```bash
getent hosts frutiger
ping -c1 frutiger
```

Verify from worker:

```bash
getent hosts kenmasters
ping -c1 kenmasters
```

---

## 2) Install Munge on the worker and copy the Munge key (MUST match controller)

On worker:

```bash
sudo apt update
sudo apt -y install munge
sudo systemctl stop munge
```

From controller, copy the key:

```bash
sudo scp /etc/munge/munge.key frutiger:/tmp/munge.key
```

On worker, install it with correct permissions:

```bash
sudo mv /tmp/munge.key /etc/munge/munge.key
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 0400 /etc/munge/munge.key
sudo systemctl enable --now munge
munge -n | unmunge
```

---

## 3) Install SLURM compute daemon on the worker

On worker:

```bash
sudo apt -y install slurm-wlm slurmd
```

If `slurmctld` gets installed, ensure it is NOT running on workers:

```bash
sudo systemctl disable --now slurmctld || true
sudo systemctl mask slurmctld || true
```

---

## 4) Generate the correct node hardware line on the worker

On worker:

```bash
sudo slurmd -C
```

Copy the `NodeName=...` line output.

Example output shape:

```conf
NodeName=frutiger CPUs=24 Boards=1 SocketsPerBoard=1 CoresPerSocket=12 ThreadsPerCore=2 RealMemory=64202
```

---

## 5) Update `slurm.conf` on the controller

On controller:

```bash
sudo vim /etc/slurm/slurm.conf
```

Add a node definition for the worker. Prefer explicit topology fields:

```conf
NodeName=frutiger Sockets=1 CoresPerSocket=12 ThreadsPerCore=2 CPUs=24 RealMemory=64202 State=UNKNOWN
```

Add the worker to a partition (example `debug`):

```conf
PartitionName=debug Nodes=kenmasters,frutiger Default=YES MaxTime=INFINITE State=UP
```

Restart controller:

```bash
sudo systemctl restart slurmctld
```

---

## 6) Copy the updated `slurm.conf` to the worker

From controller:

```bash
sudo scp /etc/slurm/slurm.conf frutiger:/tmp/slurm.conf
ssh frutiger 'sudo mkdir -p /etc/slurm && sudo mv /tmp/slurm.conf /etc/slurm/slurm.conf'
```

---

## 7) Create worker runtime directories and start `slurmd`

On worker:

```bash
sudo mkdir -p /var/spool/slurmd /var/log/slurm
sudo chown -R slurm:slurm /var/spool/slurmd /var/log/slurm
sudo systemctl enable --now slurmd
sudo systemctl status slurmd --no-pager -l
```

---

## 8) Verify from the controller

On controller:

```bash
sinfo
scontrol show node frutiger
```

Expected: node transitions to `idle`.

---

## Troubleshooting

### Node stuck `UNKNOWN`
On worker:

```bash
sudo tail -n 200 /var/log/slurm/slurmd.log
# or
sudo journalctl -u slurmd --no-pager -n 200
```

On controller:

```bash
sudo tail -n 200 /var/log/slurm/slurmctld.log
```

Common causes:
- Munge key mismatch (copy `/etc/munge/munge.key` from controller)
- Hostname mismatch (NodeName must match `hostname` / DNS name SLURM uses)
- Wrong CPU topology in `slurm.conf` (use `slurmd -C` output)
- Firewall blocking ports (see SLURM ports below)

### Ports to allow (typical defaults)
- slurmctld: TCP 6817
- slurmd: TCP 6818

If you run a firewall (ufw/iptables), allow controller inbound 6817 and worker inbound 6818.

---