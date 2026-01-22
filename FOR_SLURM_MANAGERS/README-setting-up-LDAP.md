# LDAP + SSSD Node Setup (Plain LDAP) + User Management via `ldap-account-manager` / `ldapscripts`

_disclaimer: This document was authored by Chat GPT 5.2_

This document covers:
1) How to configure a node to authenticate against LDAP using SSSD (no TLS / no StartTLS).
2) How to create and manage users/groups using the **`ldapscripts`** package (no hand-written `.ldif`).

LDAP server: `kenmasters`  
Base DN: `dc=cluster,dc=local`  
People OU: `ou=people,dc=cluster,dc=local`  
Groups OU: `ou=groups,dc=cluster,dc=local`  
Access group: `hpcusers`

---

## Part A — Node setup (SSSD, no TLS)

### 1) Install packages

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y sssd sssd-ldap libnss-sss libpam-sss ldap-utils
```

#### RHEL / Rocky / Alma
```bash
sudo dnf install -y sssd sssd-ldap openldap-clients
```

---

### 2) Configure SSSD

Create `/etc/sssd/sssd.conf`:

```ini
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

# Disable TLS / StartTLS completely
ldap_id_use_start_tls = false
ldap_tls_reqcert = never
ldap_auth_disable_tls_never = true

ldap_schema = rfc2307

cache_credentials = true
enumerate = false

access_provider = simple
simple_allow_groups = hpcusers

fallback_homedir = /home/%u
default_shell = /bin/bash
```

Set permissions:

```bash
sudo chmod 600 /etc/sssd/sssd.conf
```

Restart and clear cache:

```bash
sudo systemctl enable sssd
sudo systemctl restart sssd
sudo sss_cache -E
```

---

### 3) Verify NSS is using SSSD

```bash
grep sss /etc/nsswitch.conf
```

Expected:
```
passwd: files sss
group:  files sss
shadow: files sss
```

---

### 4) Verify user lookup works

```bash
getent passwd chandler2
getent group hpcusers
```

---

### 5) Verify authentication path (PAM)

```bash
sudo sssctl user-checks chandler2 -a auth -s sshd
```

---

### 6) SSH notes (if you can’t SSH but can `salloc`)

If `salloc` drops you into a node shell, identity + PAM can still be “fine” while SSH is broken due to:
- sshd not running
- firewall rules
- sshd not configured to use PAM
- compute nodes intentionally deny inbound SSH

To check sshd:
```bash
sudo systemctl status ssh || sudo systemctl status sshd
sudo journalctl -u ssh -n 50 || sudo journalctl -u sshd -n 50
```

---

## Part B — User + Group management using `ldapscripts` (no `.ldif`)

The `ldapscripts` tooling provides commands like:
- `ldapadduser`
- `ldapaddgroup`
- `ldapaddusertogroup`
- `ldapsetpasswd`
- `ldapdeleteuser`
- `ldapdeletegroup`

These wrap the LDIF work for you.

### 1) Install `ldapscripts` on **kenmasters** (or your admin box)

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y ldapscripts
```

#### RHEL / Rocky / Alma
`ldapscripts` may not be in default repos; easiest is to run it from an Ubuntu/Debian admin host or container.
(You can still administer the LDAP server remotely.)

---

### 2) Configure `ldapscripts`

Main config file is usually:
- `/etc/ldapscripts/ldapscripts.conf`
and a secrets file:
- `/etc/ldapscripts/ldapscripts.passwd`

Edit `/etc/ldapscripts/ldapscripts.conf`:

```bash
sudoedit /etc/ldapscripts/ldapscripts.conf
```

Set these fields:

```conf
SERVER="kenmasters"
BINDDN="cn=admin,dc=cluster,dc=local"
BINDPWDFILE="/etc/ldapscripts/ldapscripts.passwd"

SUFFIX="dc=cluster,dc=local"
GSUFFIX="ou=groups"
USUFFIX="ou=people"
MSUFFIX="ou=machines"

# Plain LDAP (no TLS)
LDAPSSL="no"
LDAPTLS="no"

# Typical POSIX ranges (adjust to your environment)
GIDSTART="20000"
UIDSTART="10000"

# Home/shell defaults used when creating users
HOMESKEL="/etc/skel"
HOMEPERMS="700"
DEFAULTSHELL="/bin/bash"
```

Create the password file:

```bash
sudo sh -c 'umask 077; printf "%s\n" "REPLACE_WITH_ADMIN_PASSWORD" > /etc/ldapscripts/ldapscripts.passwd'
sudo chmod 600 /etc/ldapscripts/ldapscripts.passwd
```

Sanity check:

```bash
sudo ldapinit
```

If `ldapinit` prints errors, the bind DN/password or suffix/ou settings are wrong.

---

### 3) Create the access group (`hpcusers`)

```bash
sudo ldapaddgroup hpcusers
```

Verify:

```bash
getent group hpcusers || ldapsearch -x -H ldap://kenmasters -b "ou=groups,dc=cluster,dc=local" "(cn=hpcusers)" dn
```

---

### 4) Add a user

Create user `newuser` (interactive prompts vary slightly by distro/version):

```bash
sudo ldapadduser newuser hpcusers
```

If you prefer to create user first, then add to group:

```bash
sudo ldapadduser newuser
sudo ldapaddusertogroup newuser hpcusers
```

Set password:

```bash
sudo ldapsetpasswd newuser
```

---

### 5) Verify the user exists (server-side)

```bash
ldapsearch -x -H ldap://kenmasters -b dc=cluster,dc=local "(uid=newuser)" dn uidNumber gidNumber homeDirectory loginShell
```

---

### 6) Verify from a node

On any node configured with SSSD:

```bash
getent passwd newuser
id newuser
```

If caching is stale:

```bash
sudo sss_cache -E
```

---

### 7) Delete user / group

Remove user:

```bash
sudo ldapdeleteuser newuser
```

Remove group:

```bash
sudo ldapdeletegroup hpcusers
```

---

## Operational notes

### A) SSSD caches identities
When you change group membership and it doesn’t show up immediately on a node:

```bash
sudo sss_cache -E
```

### B) Group-based login control
You’re using:

```ini
access_provider = simple
simple_allow_groups = hpcusers
```

So if a user can resolve but can’t log in, first confirm they’re in `hpcusers`:

```bash
getent group hpcusers
id username
```

### C) If you do not want StartTLS anywhere
Ensure the LDAP server is not “forcing” StartTLS and that clients aren’t trying it.

On nodes, confirm these exist in `sssd.conf`:

```ini
ldap_id_use_start_tls = false
ldap_tls_reqcert = never
ldap_auth_disable_tls_never = true
```

---

## Summary cheat-sheet (admin)

```bash
# one-time
sudo apt install -y ldapscripts
sudoedit /etc/ldapscripts/ldapscripts.conf
sudo sh -c 'umask 077; echo "PASSWORD" > /etc/ldapscripts/ldapscripts.passwd'
sudo chmod 600 /etc/ldapscripts/ldapscripts.passwd
sudo ldapinit

# group
sudo ldapaddgroup hpcusers

# user
sudo ldapadduser alice
sudo ldapsetpasswd alice
sudo ldapaddusertogroup alice hpcusers
```

End of document.

