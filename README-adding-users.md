# Adding Users to the Cluster (Linux + SLURM)

For basic SLURM usage without accounting (`slurmdbd`), users are simply **Linux accounts** on the cluster nodes.

---

## 1) Create a new Linux user

On each machine where the user will log in (often the controller / login node):

```bash
sudo adduser alice
```

This creates:
- user `alice`
- home directory `/home/alice`

---

## 2) Ensure the same user exists on all compute nodes (recommended)

If users will access files on compute nodes directly (or you are not using shared home directories), create the same username on all nodes.

For consistent UIDs/GIDs across nodes (recommended), you can set UID explicitly:

```bash
sudo adduser --uid 1101 alice
```

Check UID:

```bash
id alice
```

> Best practice for multi-node clusters: use centralized identity (LDAP/FreeIPA) or ensure identical UID/GID manually.

---

## 3) Optional: allow sudo (admin users only)

```bash
sudo usermod -aG sudo alice
```

---

## 4) Optional: GPU/graphics permissions (common for Intel GPU via /dev/dri)

Many GPU device files are owned by group `video`. Add user:

```bash
sudo usermod -aG video alice
```

User must log out/in for group membership to apply.

---

## 5) Test job submission as the new user

```bash
su - alice
srun -N1 -n1 hostname
exit
```

---

## 6) Optional: Restrict who can submit jobs to a partition

Create a group:

```bash
sudo groupadd slurmusers
sudo usermod -aG slurmusers alice
```

Then restrict the partition in `/etc/slurm/slurm.conf`:

```conf
PartitionName=debug Nodes=kenmasters,frutiger Default=YES MaxTime=INFINITE State=UP AllowGroups=slurmusers
```

Restart controller after changes:

```bash
sudo systemctl restart slurmctld
```

---

## 7) If you enable accounting later (slurmdbd)

If you later deploy `slurmdbd` and want accounts/QOS, you’ll manage users with `sacctmgr` (not covered in this basic guide).

---