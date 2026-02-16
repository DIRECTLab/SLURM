# Logging in to the SLURM Server

This guide shows you how to SSH into the server using your public key that has already been configured.

## Prerequisites
- Your SSH public key has been added to the server's authorized keys
- You have the private key on your local machine (typically at `~/.ssh/id_ed25519` or `~/.ssh/id_rsa`)

## Server Details
- **Server IP**: 129.123.61.22
- **Username**: Your username.

## How to SSH into the Server

Replace `username` with your actual username, for student accounts, this will be your A-number, for permanent accounts this will be the username you set on the form.

### Linux and macOS
```bash
ssh -p 4242 username@129.123.61.22
```

### Windows (PowerShell)
```powershell
ssh -p 4242 username@129.123.61.22
```

## First Time Login

On your first login, you may see a message asking if you want to add the server to your known hosts. Type `yes` and press Enter:

```
The authenticity of host '129.123.61.22 (129.123.61.22)' can't be established.
...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

## Troubleshooting

### Permission Denied (publickey)
- Verify you're using the correct A-number
- Ensure your SSH private key is in the correct location (`~/.ssh/`)
- Check that your public key matches the one registered on the server
- Make sure your private key has the correct permissions: `chmod 600 ~/.ssh/id_ed25519`

### Connection Refused
- Verify the server IP is correct: `129.123.61.22`
- Check that you have network connectivity to the server
- Ensure the server is running and accessible
- Verify that you are on eduroam wifi or connected via VPN

### Using a Non-Default Key Location

If your SSH key is not in the default location, specify it with the `-i` flag:

```bash
ssh -i /path/to/your/private/key -p 4242 a_number@129.123.61.22
```

## Tips

- **Avoid retyping your login each time**: Add the server to your SSH config file (`~/.ssh/config`):
  ```
  Host slurm-server
      HostName 129.123.61.22
      Port 4242
      User a_number
      IdentityFile ~/.ssh/id_ed25519
  ```
  Then simply login with: `ssh slurm-server`

- **File transfer**: Use `scp` to copy files to/from the server:
  ```bash
  # Copy file to server
  scp -P 4242 local_file a_number@129.123.61.22:/path/on/server/
  
  # Copy file from server
  scp -P 4242 a_number@129.123.61.22:/path/on/server/file local_file
  ```

## Using VS Code Remote SSH

You can use VS Code to edit and work directly on the server with the Remote SSH extension.

### Setup

1. **Install the Remote SSH extension**:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X / Cmd+Shift+X)
   - Search for "Remote SSH" by Microsoft
   - Click Install

2. **Add the server to your SSH config** (if you haven't already):
   - Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
   - Type "Remote-SSH: Open Configuration File..."
   - Select `~/.ssh/config`
   - Add the following entry (or update if it exists):
     ```
     Host slurm-server
         HostName 129.123.61.22
         Port 4242
         User a_number
         IdentityFile ~/.ssh/id_ed25519
     ```

3. **Connect to the server**:
   - Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
   - Type "Remote-SSH: Connect to Host..."
   - Select `slurm-server` from the list
   - VS Code will open a new window connected to the server

4. **Open a folder on the server**:
   - Once connected, use File → Open Folder
   - Navigate to the directory you want to work in
   - Click OK

### Tips for Remote SSH

- **Terminal on server**: Open a terminal in VS Code (Ctrl+`) and it will be a shell on the remote server
- **File Explorer**: Browse and edit files directly on the server in the left sidebar
- **Extensions on server**: Some extensions may need to be installed on the remote server; VS Code will prompt you when needed
- **Forwarding ports**: If you need to access services running on the server, use the Port Forwarding feature in the Remote SSH extension
