# Create a Public SSH Key (Windows + WSL, macOS, Linux)

This guide walks you through creating an SSH key pair and adding the key to `ssh-agent` on:
- Windows (using WSL)
- macOS
- Linux

At the end you’ll have:
- a **private key** (keep secret) and
- a **public key** (safe to share; added to GitHub/GitLab/servers)

---

## 0) Before you start

### Choose a key type
Recommended:
- **Ed25519** (modern, fast, strong): `ssh-ed25519`

Fallback (only if Ed25519 isn’t supported in your environment):
- **RSA 4096-bit**: `rsa -b 4096`

### Pick an email/comment
Most examples use `you@example.com` as the key “comment” (label). Use your real email or any helpful label.

### Know where keys are stored
Keys are created in your home directory under:

- `~/.ssh/id_ed25519` (private key)
- `~/.ssh/id_ed25519.pub` (public key)

If you already have keys and don’t want to overwrite them, use a different filename (shown below).

---

## 1) Windows (with WSL)

These steps run **inside WSL** (Ubuntu, Debian, etc.). Open your WSL terminal first.

### 1.1 Generate the key
Ed25519:

```bash
ssh-keygen -t ed25519 -C "you@example.com"
```

When prompted:
- **File location**: press **Enter** to accept the default (`/home/<you>/.ssh/id_ed25519`), or enter a custom name (recommended if you already have a key), e.g.:
  - `/home/<you>/.ssh/id_ed25519_github`
- **Passphrase**: recommended (you can leave blank, but a passphrase is safer)

### 1.2 Start `ssh-agent` and add the key
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

If you used a custom filename, add that instead:
```bash
ssh-add ~/.ssh/id_ed25519_github
```

### 1.3 Copy the public key to your clipboard
Print it:
```bash
cat ~/.ssh/id_ed25519.pub
```

Copy it using one of these options:

**Option A: `clip.exe` (usually works in WSL)**
```bash
cat ~/.ssh/id_ed25519.pub | clip.exe
```

**Option B: Just copy from the terminal**
- Select the output of `cat ...pub` and copy it manually.

---

## 2) macOS

Open **Terminal**.

### 2.1 Generate the key
```bash
ssh-keygen -t ed25519 -C "you@example.com"
```

### 2.2 Start `ssh-agent` and add the key
```bash
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

If `--apple-use-keychain` is not recognized on your system, use:
```bash
ssh-add ~/.ssh/id_ed25519
```

### 2.3 Copy the public key
```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

Or print it to copy manually:
```bash
cat ~/.ssh/id_ed25519.pub
```

---

## 3) Linux

Open your terminal.

### 3.1 Generate the key
```bash
ssh-keygen -t ed25519 -C "you@example.com"
```

### 3.2 Start `ssh-agent` and add the key
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 3.3 Copy the public key
Print it:
```bash
cat ~/.ssh/id_ed25519.pub
```

Optional clipboard commands (depends on your desktop environment):

- **xclip** (X11):
  ```bash
  sudo apt-get update && sudo apt-get install -y xclip
  xclip -selection clipboard < ~/.ssh/id_ed25519.pub
  ```

- **wl-copy** (Wayland):
  ```bash
  sudo apt-get update && sudo apt-get install -y wl-clipboard
  wl-copy < ~/.ssh/id_ed25519.pub
  ```

---

## 4) Add the public key to a service (example: GitHub)

1. Copy your **public** key (the `.pub` file contents).
2. In GitHub: **Settings → SSH and GPG keys → New SSH key**
3. Paste the key and save.

---

## 5) Test your SSH connection (example: GitHub)

```bash
ssh -T git@github.com
```

Expected output includes something like:
- “Hi `<username>`! You’ve successfully authenticated...”

If you have multiple keys, you may need a config file (see next section).

---

## 6) (Optional) Using multiple SSH keys with `~/.ssh/config`

Create or edit `~/.ssh/config`:

```sshconfig
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

If using a custom key filename:

```sshconfig
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
```

Then test again:
```bash
ssh -T git@github.com
```

---

## Troubleshooting

### “Permission denied (publickey).”
Common causes:
- You added the wrong key to the service (ensure you used the `.pub` contents).
- `ssh-agent` isn’t running or the key wasn’t added (`ssh-add -l` to list loaded keys).
- You have multiple keys and SSH is offering the wrong one (use `~/.ssh/config` with `IdentitiesOnly yes`).

### Check what key SSH is using
```bash
ssh -vT git@github.com
```

Look for lines mentioning `Offering public key:`.

### View your public key again
```bash
cat ~/.ssh/id_ed25519.pub
```

---

## Security notes
- Never share your **private key** (`id_ed25519`) with anyone.
- Use a passphrase when possible.
- Consider using separate keys for work vs personal accounts.