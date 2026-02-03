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

## 9) Mounting the home directory on compute nodes (make `/home` shared from `kenmasters`)



If you plan to mount the same `/home` on every compute node (so paths like `/home/software/miniforge3` are identical everywhere), follow these steps. The examples below assume Debian/Ubuntu on controller and workers; RHEL/CentOS notes are included. **Note:** The NFS host node is now `ryu` (not `kenmasters`).

---

## A. Setting Up the NFS Host (`ryu`) (Ignore if already setup)

**Important notes before you start:**
- Exporting the whole `/home` is common but consider exporting a specific subtree (e.g. `/export/home` or `/export/software`) if you want finer control.
- Restrict the export to your compute subnet (do NOT use `*(rw)` in production).
- Ensure UID/GID consistency (LDAP/SSSD or synchronized passwd/group) so file ownerships are identical on all nodes.
- Be careful with `no_root_squash` — it lets remote root act as root on the server and has security implications. Prefer default `root_squash` unless you explicitly need root permissions from clients.

1. **Install NFS server utilities:**
  ```bash
  sudo apt update
  sudo apt -y install nfs-kernel-server
  ```
  (RHEL/CentOS: `sudo yum install -y nfs-utils` and enable nfs-server service)

2. **Configure the export:**
  - Edit `/etc/exports` and add (replace `192.168.0.0/24` with your subnet if different):
    ```
    /home 192.168.0.0/24(rw,sync,no_subtree_check)
    ```
  - For less restrictive root access (not recommended for most cases):
    ```
    /home 192.168.0.0/24(rw,sync,no_subtree_check,no_root_squash)
    ```

3. **Apply the export and start the NFS server:**
  ```bash
  sudo exportfs -ra
  sudo exportfs -v
  sudo systemctl enable --now nfs-server
  ```

4. **Verify the export from another host:**
  ```bash
  showmount -e ryu
  rpcinfo -p ryu
  ```

---

## B. Adding a New NFS Client (e.g., a compute node)

1. **Install NFS client utilities:**
  ```bash
  sudo apt update
  sudo apt -y install nfs-common
  ```
  (RHEL/CentOS: `sudo yum install -y nfs-utils`)

2. **Create the mount point and mount temporarily:**
  ```bash
  sudo mkdir -p /home
  sudo mount -t nfs -o vers=4.2,proto=tcp ryu:/home /home
  # or simply:
  sudo mount ryu:/home /home
  ```

3. **Add a persistent mount in `/etc/fstab`:**
  - Add this line:
    ```
    ryu:/home  /home  nfs  defaults,_netdev,vers=4.2,hard,intr  0 0
    ```
  - Then reload and mount:
    ```bash
    sudo systemctl daemon-reload
    sudo mount -a
    ```

4. **Test the mount:**
  ```bash
  mount | grep ' /home '
  df -h /home
  ls -la /home
  echo "test from $(hostname)" | sudo tee /home/nfs_test.$(hostname)
  ```

5. **Verify on the NFS host (`ryu`):**
  ```bash
  ls -l /home/nfs_test.*
  ```

---

C. Troubleshooting & common issues

- "bad option; need mount helper" — indicates `nfs-common` (client) not installed.
- If `mount` hangs or fails, check firewall between nodes and server (NFS/RPC ports). On the server, ensure `nfs-server` is running.
- If files appear but ownerships differ, check UIDs/GIDs. Use a centralized identity service (LDAP/SSSD) or synchronize `/etc/passwd` and `/etc/group`.
- If root cannot write remotely, the server may be using `root_squash`. Change export options only if required.
- SELinux: on RHEL/CentOS, SELinux contexts can block access — check `sudo ausearch -m avc -ts recent` or set permissive while debugging.
- Performance: exports over NFS may require tuning (rsize/wsize, NFS v4.1/4.2, network tuning). For heavy I/O conda activation can be slow; consider `conda-pack` + local extraction or use containers.

D. Automation

Use Ansible (recommended) to add the mount and client packages on many nodes. Example (NFS client role):

```yaml
- hosts: compute_nodes
  become: true
  tasks:
    - apt: name=nfs-common state=present update_cache=yes
    - file:
        path: /home
        state: directory
        owner: root
        group: root
        mode: '0755'
    - mount:
        path: /home
        src: "kenmasters:/home"
        fstype: nfs
        opts: "defaults,_netdev,vers=4.2,hard,intr"
        state: mounted
        dump: 0
        passno: 0
```

E. Quick verification for conda visibility

Once `/home` is mounted on a compute node, test:

```bash
# on compute node
ls -ld /home/software/miniforge3
/home/software/miniforge3/bin/conda --version
# Example Slurm job snippet to ensure conda is usable in jobs:
#!/bin/bash
#SBATCH --job-name=test-conda
#SBATCH --output=test-conda.%j.out

source /home/software/miniforge3/etc/profile.d/conda.sh
conda --version
conda env list
```

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
- NFS (if using NFS export): TCP/UDP 2049 (and RPCbind/other RPC ports)

If you run a firewall (ufw/iptables), allow controller inbound 6817 and worker inbound 6818; allow NFS ports between controller and compute nodes.

---