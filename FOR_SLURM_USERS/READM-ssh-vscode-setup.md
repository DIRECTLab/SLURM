# SSH Access & VS Code Remote Setup

This guide explains how to SSH directly into compute nodes in the SLURM cluster using SSH config.

## Prerequisites

- You have an account on the SLURM cluster
- SSH client installed on your local machine
- (Optional) VS Code with Remote - SSH extension

## How SSH Access Works

The cluster uses an SSH gate system that enforces a security policy:
- You must have an **active SLURM job running** on the target compute node before you can SSH into it
- Compute nodes are not directly accessible from external networks; you must SSH through the **login node** (ProxyJump)

## Step 1: Allocate Resources on a Node

Before SSHing into a node, you must allocate resources using `salloc`:

```bash
salloc -w <nodename> -c 8
```

Note: Please use the above command, unless you know what you are doing. Also note, if you set `--gpus=1`, no one else will be able to use that node, so don't do it unless absolutely necessary.

**Parameters breakdown:**
- `-w, --nodelist=<nodename>`: Specific node to use (required) - e.g., `ryu`, `chunli`
- `-t, --time=<time>`: Time limit for the allocation - e.g., `04:00:00` (4 hours), `1-00:00:00` (1 day)
- `-c, --cpus-per-task=<count>`: Number of CPU cores to allocate - e.g., `8`, `16`

**Additional optional parameters:**
- `-p, --partition=<name>`: Partition/queue name (e.g., `debug`, `gpu`, `cpu`)
- `-N, --nodes=<count>`: Number of nodes (default: 1)
- `-n, --ntasks=<count>`: Total number of tasks
- `--mem=<size>`: Memory per node (e.g., `16G`, `32GB`)

### Example Allocations

**Basic allocation (4 hours, 8 CPUs):**
```bash
salloc -w ryu -t 04:00:00 -c 8
```

**Longer allocation (24 hours, 16 CPUs):**
```bash
salloc -w chunli -t 1-00:00:00 -c 16
```

**With specific memory:**
```bash
salloc -w ryu -t 04:00:00 -c 8 --mem=32G
```

Once `salloc` succeeds, your job is running and you can SSH into the node.

## Step 2: Configure SSH

Edit your SSH config file at `~/.ssh/config` and add the following entries:

```
Host clusteruser
    HostName 129.123.61.22
    User username
    Port 4242
    ForwardAgent yes

Host ryu
    HostName ryu
    User username
    ProxyJump clusteruser
    ForwardAgent yes

Host chunli
    HostName chunli
    User username
    ProxyJump clusteruser
    ForwardAgent yes
```

Replace `username` with your actual cluster username. Add additional compute node entries as needed.

## Step 3: SSH into a Node

Once you have an active `salloc` allocation, SSH into the node:

```bash
ssh ryu
```

The SSH gate will verify that you have a running job on that node. If successful, you'll be in a bash shell.

Other examples:
```bash
ssh chunli
ssh ryu "python my_script.py"    # Run a command directly
```

## VS Code Remote SSH Setup

### Installation

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Remote - SSH" by Microsoft
4. Click Install

### Connecting via VS Code

1. Ensure you have an active `salloc` allocation (Step 1)
2. Click the Remote Explorer icon in the left sidebar (or press Ctrl+Shift+P and search "Remote-SSH: Connect to Host...")
3. Select the node you want to connect to
4. A new VS Code window will open connected to that node

## Troubleshooting

### "Access denied: you have no active RUNNING Slurm jobs"

You don't have a running job on the target node. Solution:
1. Open another terminal
2. Run `salloc` to allocate resources (see Step 1)
3. Try SSH again

### SSH Connection Issues

Verify you can reach the login node:
```bash
ssh clusteruser
```

Check your SLURM allocation status:
```bash
squeue -u $USER
```

Ensure your SSH agent is running with your key:
```bash
eval $(ssh-agent)
ssh-add ~/.ssh/id_rsa
```

## Keeping Your Allocation Active

Your `salloc` allocation will remain active until:
- The time limit expires (`-t` parameter)
- You explicitly release it with `exit` or `Ctrl+C` in the salloc terminal
- The system administrator terminates it

To extend an allocation, release it and create a new one with a longer time limit.

## Best Practices

1. **Allocate only what you need**: More CPUs/time = less availability for others
2. **Release early**: When done, end your `salloc` session to free up resources
3. **Monitor your jobs**: Use `squeue -u $USER` to see active allocations
4. **Test first**: Before long computations, test with a shorter time limit

## Additional Resources

For more SLURM commands:
```bash
man salloc
man squeue
man sinfo
```

Questions? Contact your system administrator.
