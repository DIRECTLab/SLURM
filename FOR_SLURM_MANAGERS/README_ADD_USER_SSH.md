# Add a Linux user and enable SSH login with a public key

This guide creates a new Linux user and configures **key-based SSH access** (public key authentication).

## Prerequisites
- You can SSH into the server as `root` **or** a user with `sudo` privileges.
- You have the user’s **SSH public key** (typically a single line starting with `ssh-ed25519` or `ssh-rsa`).


---

## 1) Create the user
Replace `newuser` with the username you want.

```bash
sudo useradd -m newuser
```

---

## 2) Login into user and create ssh profile
```bash
sudo -i
sudo su - newuser
ssh-keygen
```
---

## 3) Make authorized key directory and add public key
```bash
touch ~/.ssh/authorized_keys
```
Paste the public key from the users machine into the `authorized_keys` file.
---

# Delete User and their home directory

```bash
  sudo userdel -r username
```