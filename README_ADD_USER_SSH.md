# Add a Linux user and enable SSH login with a public key

This guide creates a new Linux user and configures **key-based SSH access** (public key authentication).

## Prerequisites
- You can SSH into the server as `root` **or** a user with `sudo` privileges.
- You have the user’s **SSH public key** (typically a single line starting with `ssh-ed25519` or `ssh-rsa`).

---

## 1) Create the user
Replace `newuser` with the username you want.

```bash
sudo adduser newuser
```

(Optional) Add to sudoers (common for admin users):
```bash
sudo usermod -aG sudo newuser
```

On RHEL/CentOS/Amazon Linux, the admin group may be `wheel`:
```bash
sudo usermod -aG wheel newuser
```

---

## 2) Create the `.ssh` directory and set permissions
```bash
sudo mkdir -p /home/newuser/.ssh
sudo chmod 700 /home/newuser/.ssh
sudo chown -R newuser:newuser /home/newuser/.ssh
```

---

## 3) Add the public key to `authorized_keys`
Paste the public key into the file (single line).

### Option A: Using an editor
```bash
sudo nano /home/newuser/.ssh/authorized_keys
```
Paste the key, save, and exit.

### Option B: Using `tee` (replace the key below)
```bash
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... user@laptop" | sudo tee -a /home/newuser/.ssh/authorized_keys > /dev/null
```

Set correct permissions/ownership:
```bash
sudo chmod 600 /home/newuser/.ssh/authorized_keys
sudo chown newuser:newuser /home/newuser/.ssh/authorized_keys
```

---

## 4) (Recommended) Ensure SSH server allows public key auth
Open `/etc/ssh/sshd_config` and confirm these settings:

```text
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
```

Then reload SSH:

```bash
sudo systemctl reload sshd || sudo systemctl reload ssh
```

---

## 5) Test login
From your local machine:

```bash
ssh newuser@SERVER_IP
```

If the key is not your default key:
```bash
ssh -i ~/.ssh/id_ed25519 newuser@SERVER_IP
```

---

## 6) (Optional but recommended) Disable password SSH authentication
If you are confident key login works, you can harden SSH by disabling password auth.

Edit `/etc/ssh/sshd_config`:
```text
PasswordAuthentication no
ChallengeResponseAuthentication no
```

Reload SSH:
```bash
sudo systemctl reload sshd || sudo systemctl reload ssh
```

**Important:** Keep your current session open while testing a new session, in case you need to revert.

---

## Troubleshooting
- **Permissions are wrong** (most common):
  - `/home/newuser/.ssh` must be `700`
  - `authorized_keys` must be `600`
  - Owned by `newuser:newuser`
- **Wrong key format**: The public key must be one line starting with `ssh-ed25519` / `ssh-rsa` etc.
- **SSH logs**:
  - Debian/Ubuntu: `sudo tail -n 200 /var/log/auth.log`
  - RHEL/CentOS: `sudo tail -n 200 /var/log/secure`

  # Delete User and their home directory

  sudo userdel -r username