# Adding New Compute Nodes to a SLURM Cluster

This guide adds a new compute node (worker) to an existing SLURM controller.

---
### Table of Contents
1. [Name Resolution](#1-ensure-name-resolution-works-on-both-machines)
1. [Install Munge](#2-install-munge-on-the-worker-and-copy-the-munge-key-must-match-controller)
1. [Install SLURM daemon](#3-install-slurm-compute-daemon-on-the-worker)
1. [Generate worker hardware line](#4-generate-the-correct-node-hardware-line-on-the-worker)
1. [Update slurm config on controller](#5-update-slurmconf-on-the-controller)
1. [Copy slurm config to worker](#6-copy-the-updated-slurmconf-to-the-worker)
1. [Create worker runtime dirs and start slurmd](#7-create-worker-runtime-directories-and-start-slurmd)
1. [Verify from controller](#8-verify-from-the-controller)
1. [Mounting home dir on compute nodes](#9-mounting-the-home-directory-on-compute-nodes-make-home-shared-from-kenmasters)
    1. [A. Setting up NFS Host](#a-setting-up-the-nfs-host-ryu-ignore-if-already-setup)
    1. [B. Adding NFS Client](#b-adding-a-new-nfs-client-eg-a-compute-node)
1. [Configure SSSD with LDAP](#10-configure-sssd-to-use-login-nodes-ldap-auth-server)
1. [Troubleshooting](#troubleshooting)
---


Assumptions:
- Controller host: `kenmasters`
- New worker host: `frutiger` (replace with your worker hostname)
- Static IPs are already configured
- You are using Munge authentication

---

## 1) Ensure name resolution works on BOTH machines
*[Table of Contents](#table-of-contents)*

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
*[Table of Contents](#table-of-contents)*

On worker:

```bash
sudo apt update
sudo apt -y install munge
sudo systemctl stop munge
```

From controller, copy the key:

```bash
sudo scp /etc/munge/munge.key direct@frutiger:/tmp/munge.key
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
*[Table of Contents](#table-of-contents)*

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
*[Table of Contents](#table-of-contents)*

On worker:

```bash
sudo slurmd -C
```

Copy the full `NodeName=...` line output you will need it in step 5.

Example output shape:

```conf
NodeName=frutiger CPUs=24 Boards=1 SocketsPerBoard=1 CoresPerSocket=12 ThreadsPerCore=2 RealMemory=64202
```

---

## 5) Update `slurm.conf` on the controller
*[Table of Contents](#table-of-contents)*

On controller:

```bash
sudo vim /etc/slurm/slurm.conf
```

Add a node definition for the worker.

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
*[Table of Contents](#table-of-contents)*

From controller:

```bash
sudo scp /etc/slurm/slurm.conf frutiger:/tmp/slurm.conf
ssh frutiger 'sudo mkdir -p /etc/slurm && sudo mv /tmp/slurm.conf /etc/slurm/slurm.conf'
```

---

## 7) Create worker runtime directories and start `slurmd`
*[Table of Contents](#table-of-contents)*

On worker:

```bash
sudo mkdir -p /var/spool/slurmd /var/log/slurm
sudo chown -R slurm:slurm /var/spool/slurmd /var/log/slurm
sudo systemctl enable --now slurmd
sudo systemctl status slurmd --no-pager -l
```

---

## 8) Verify from the controller
*[Table of Contents](#table-of-contents)*

On controller:

```bash
sinfo
scontrol show node frutiger
```

Expected: node transitions to `idle`.

---


## 9) Mounting the home directory on compute nodes (make `/home` shared from `kenmasters`)
*[Table of Contents](#table-of-contents)*



If you plan to mount the same `/home` on every compute node (so paths like `/home/software/miniforge3` are identical everywhere), follow these steps. The examples below assume Debian/Ubuntu on controller and workers; RHEL/CentOS notes are included. **Note:** The NFS host node is now `ryu` (not `kenmasters`).

---

## A. Setting Up the NFS Host (`ryu`) (Ignore if already setup)
*[Table of Contents](#table-of-contents)*

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
*[Table of Contents](#table-of-contents)*

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

## 10) Configure SSSD to use login node's LDAP Auth Server
*[Table of Contents](#table-of-contents)*


1. Update the system and install required packages

```
sudo apt update
sudo apt install -y \
  sssd sssd-ldap \
  libnss-sss libpam-sss \
  ldap-utils \
  openssh-server \
  sssd-tools
```

2. Place config file in `/etc/sssd/sssd.conf`

```
[sssd]
services = nss, pam
config_file_version = 2
domains = LDAP

[domain/LDAP]
id_provider = ldap
auth_provider = ldap
chpass_provider = ldap

ldap_uri = ldap://kenmasters
ldap_search_base = dc=cluster,dc=local

# Disable TLS completely
ldap_id_use_start_tls = false
ldap_tls_reqcert = never
ldap_auth_disable_tls_never = true

ldap_schema = rfc2307

cache_credentials = true
enumerate = false

access_provider = permit

fallback_homedir = /home/%u
default_shell = /bin/bash
```

3. set the correct permissions for `sssd.conf`

```
sudo chmod 600 /etc/sssd/sssd.conf
sudo chown root:root /etc/sssd/sssd.conf
```

4. Enable the sssd service

```
sudo systemctl enable --now sssd
```

Check the status with this command:

```
systemctl status sssd
sssctl domain-status LDAP
```

(It should say `Online status: Online` somewhere in the output)

Now you should be able to jump from the login node to the new node. Try testing with:

```
srun -w [new_node_name] --pty bash
```

You should see your LDAP account is carried over to the new node.

## Troubleshooting
*[Table of Contents](#table-of-contents)*

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
