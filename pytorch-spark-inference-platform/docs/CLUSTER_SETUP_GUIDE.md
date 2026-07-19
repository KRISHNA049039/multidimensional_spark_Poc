# End-to-End Spark Inference Cluster Setup Guide

This document explains the complete process of deploying and running the multi-model inference benchmark on a 2-node AWS Spark cluster (1 CPU master + 1 GPU worker), including what each command does internally.

---

## Architecture Overview

```
+---------------------------+          +---------------------------+
|   spark-master (t3.large) |          | spark-gpu-worker          |
|   - CPU only, 100GB EBS   |          |   (g4dn.xlarge)          |
|   - Spark Master daemon   |          |   - NVIDIA T4 GPU        |
|   - Spark CPU Worker      |          |   - 150GB EBS            |
|   - Driver (spark-submit) |          |   - Spark GPU Worker     |
|   - Port 7077 (RPC)       |<-------->|   - NVIDIA driver + CUDA |
|   - Port 8080 (Master UI) |          |   - nvidia-container-tk  |
|   - Port 4040 (App UI)    |          |   - Port 8081 (Worker UI)|
+---------------------------+          +---------------------------+
         |                                        |
         +------- Same VPC / Security Group ------+
         |                                        |
    +----------+                            +----------+
    |    S3    |  <-- project.zip           |CloudWatch|
    |  Bucket  |  <-- Docker image tar      | Metrics  |
    +----------+                            +----------+
```

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Node.js 20+ | CDK CLI runtime | https://nodejs.org |
| AWS CDK CLI | Infrastructure as code deployment | `npm install -g aws-cdk` |
| AWS CLI v2 | AWS API access | https://awscli.amazonaws.com/AWSCLIV2.msi |
| SSM Plugin | Terminal access to EC2 instances | https://s3.amazonaws.com/session-manager-downloads/plugin/latest/windows/SessionManagerPluginSetup.exe |
| Python 3.10+ | CDK app runtime | https://python.org |
| AWS credentials | Account access (Access Key + Secret) | IAM Console |

---

## Phase 1: CDK Deployment

### 1.1 Set AWS credentials

```powershell
$env:AWS_ACCESS_KEY_ID = "AKIA..."
$env:AWS_SECRET_ACCESS_KEY = "..."
$env:AWS_DEFAULT_REGION = "ap-south-1"
```

**What this does:** Sets environment variables that all AWS tools (CDK, CLI) read to authenticate API calls. These persist only in the current PowerShell session.

### 1.2 Bootstrap CDK (one-time per account/region)

```powershell
cd D:\Spark_poc\simple_spark_torch_poc\pytorch-spark-inference-platform\deploy\aws-cdk
cdk bootstrap aws://368287210840/ap-south-1
```

**What this does:** Creates a CloudFormation stack called `CDKToolkit` that provisions:
- An S3 bucket for storing CDK assets (compiled templates, Lambda code)
- IAM roles that CDK assumes during deployment
- An ECR repository for Docker image assets (if needed)

This is a prerequisite for any CDK deployment in a given account+region.

### 1.3 Deploy the stack

```powershell
cdk deploy --parameters AdminCidr=49.205.250.126/32 -c region=ap-south-1
```

**What this does internally:**
1. **Synthesize** — CDK runs `app.py`, which instantiates `SparkClusterStack`. Python code is converted into a CloudFormation JSON template.
2. **Diff** — CDK compares the new template against what's currently deployed (nothing, for a fresh deploy).
3. **Deploy** — CDK uploads the template to S3, then calls `CloudFormation:CreateStack`. CloudFormation provisions resources in dependency order:
   - VPC (3 AZs, public subnets, internet gateway, route tables)
   - Security Group (inter-node traffic, admin UI ports)
   - IAM Role (SSM, CloudWatch, S3, EC2 describe)
   - S3 Artifacts Bucket
   - EC2 Master Instance (t3.large, Amazon Linux 2023, 100GB gp3)
   - EC2 GPU Worker Instance (g4dn.xlarge, Amazon Linux 2023, 150GB gp3)
   - CloudWatch Dashboard (pre-configured widgets)

**`AdminCidr` parameter:** Opens ports 8080 and 4040 in the security group to your IP, so you can access the Spark Master UI and Application UI from your browser.

### 1.4 Stack outputs

After deployment, CDK prints outputs:
- `MasterInstanceId` — for SSM connection
- `MasterPublicIp` — for Spark UI access
- `MasterPrivateIp` — for worker→master communication
- `GpuWorkerInstanceId` — for SSM connection
- `ArtifactsBucketName` — for uploading code/images
- `DashboardUrl` — CloudWatch metrics dashboard

---

## Phase 2: Upload Code to S3

### 2.1 Package the project

```powershell
cd D:\Spark_poc\simple_spark_torch_poc\pytorch-spark-inference-platform
Compress-Archive -Path .\* -DestinationPath .\project.zip -Force
```

**What this does:** Creates a ZIP archive of the entire project directory. The `-Force` flag overwrites any existing `project.zip`.

### 2.2 Upload to S3

```powershell
& "C:\Program Files\Amazon\AWSCLIV2\aws.exe" s3 cp project.zip s3://<BUCKET-NAME>/inference/project.zip --region ap-south-1
```

**What this does:** Uploads the ZIP to the S3 bucket created by CDK. Both EC2 instances have IAM permissions to read from this bucket, so they can pull the code without any credential wiring.

---

## Phase 3: Master Setup

### 3.1 Connect to master

```powershell
& "C:\Program Files\Amazon\AWSCLIV2\aws.exe" ssm start-session --target <MASTER-INSTANCE-ID> --region ap-south-1

```

**What this does:** Opens a secure shell session to the EC2 instance via AWS Systems Manager. No SSH keys or open ports needed — SSM uses the instance's IAM role and the SSM agent (pre-installed on Amazon Linux 2023) to establish an encrypted tunnel.

### 3.2 Download and extract code

```bash
sudo su -
export ARTIFACTS_BUCKET=<BUCKET-NAME>
aws s3 cp s3://$ARTIFACTS_BUCKET/inference/project.zip /opt/spark-inference/project.zip
mkdir -p /opt/spark-inference/app
unzip -o /opt/spark-inference/project.zip -d /opt/spark-inference/app
cd /opt/spark-inference/app
```

**What this does:**
- `sudo su -` — switches to root (needed for Docker operations)
- `export ARTIFACTS_BUCKET=...` — sets the bucket name as an env var for reuse
- `aws s3 cp` — downloads from S3 using the instance's IAM role (no credentials needed on the instance)
- `unzip -o` — extracts, overwriting existing files (`-o` = overwrite)

### 3.3 Build Docker image

```bash
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
```

**What this does:** Builds a Docker image containing:
- NVIDIA CUDA 12.1 runtime (base image)
- Python 3.11 + Java 17 (for PySpark)
- Apache Spark 3.5.1 (master/worker scripts)
- PyTorch 2.2.0 + torchvision (CUDA-enabled)
- All project code (models, inference, benchmark, monitoring)

Building on EC2 is fast because it has high-bandwidth internet for pulling base images and pip packages.

### 3.4 Start Spark Master

```bash
docker run -d --name spark-master --network host multi-model-inference:latest \
  bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
```

**What this does:**
- `docker run -d` — runs container in background (detached)
- `--network host` — container shares the host's network stack (no port mapping needed, Spark uses direct IP communication)
- `start-master.sh` — Spark script that starts the master daemon on port 7077 (RPC) and 8080 (Web UI)
- `tail -f ...` — keeps the container alive by tailing the log file (Docker stops when the main process exits)

### 3.5 Start CPU Worker on Master

```bash
MASTER_IP=$(hostname -I | awk '{print $1}')
docker run -d --name spark-cpu-worker --network host multi-model-inference:latest \
  bash -c "start-worker.sh spark://$MASTER_IP:7077 -c 2 -m 4g && tail -f /opt/spark/logs/*worker*"
```

**What this does:**
- `hostname -I` — gets the instance's private IP
- `start-worker.sh spark://<IP>:7077` — starts a Spark worker that registers with the master
- `-c 2` — offers 2 CPU cores to the cluster
- `-m 4g` — offers 4GB memory to the cluster
- This makes the master participate in computation (not just coordination)

---

## Phase 4: GPU Worker Setup

### 4.1 Connect to GPU worker (separate terminal)

Use SSM or EC2 Console → Connect → Session Manager.

### 4.2 Install NVIDIA driver

```bash
sudo su -
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
dnf module install -y nvidia-driver:latest-dkms
```

**What this does:** Installs the NVIDIA GPU kernel driver from NVIDIA's official RPM repository. The `dkms` variant automatically recompiles the kernel module if the kernel is updated. The g4dn.xlarge has an NVIDIA T4 GPU but the driver is not pre-installed on Amazon Linux 2023.

### 4.3 Install nvidia-container-toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
dnf install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

**What this does:**
- Adds NVIDIA's container toolkit RPM repo
- Installs `nvidia-container-toolkit` — the bridge between Docker and NVIDIA GPUs
- `nvidia-ctk runtime configure` — edits Docker's `/etc/docker/daemon.json` to register the NVIDIA runtime
- Docker restart picks up the new runtime config
- After this, `docker run --gpus all` will expose GPUs inside containers

### 4.4 Load Docker image and start worker

```bash
export ARTIFACTS_BUCKET=<BUCKET-NAME>
aws s3 cp s3://$ARTIFACTS_BUCKET/inference/multi-model-inference.tar - | docker load
```

**What this does:** Streams the Docker image directly from S3 into `docker load` without writing to disk. The pipe (`|`) avoids needing disk space for the full image file.

```bash
MASTER_IP=10.0.0.187
docker run -d --name spark-gpu-worker --network host --gpus all --shm-size=4g \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://$MASTER_IP:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
```

**What this does:**
- `--gpus all` — passes all NVIDIA GPUs into the container (requires nvidia-container-toolkit)
- `--shm-size=4g` — increases shared memory (PyTorch DataLoader uses /dev/shm for IPC)
- `-c 4` — offers 4 CPU cores
- `-m 12g` — offers 12GB memory (g4dn.xlarge has 16GB total, leave headroom for OS)
- Worker registers with master, which now has 2 workers (1 CPU + 1 GPU)

---

## Phase 5: Run Benchmarks

### 5.1 Verify cluster

```bash
curl -s http://localhost:8080/json/ | docker exec -i spark-master python -c \
  "import sys,json; d=json.load(sys.stdin); print(f'Workers: {d.get(\"aliveworkers\",0)}')"
```

**What this does:** Queries the Spark Master REST API (port 8080, `/json/` endpoint) which returns cluster state as JSON. We pipe it into a Python one-liner to extract the alive worker count.

### 5.2 Run distributed mode (uses GPU worker)

```bash
docker exec -it spark-master bash -c \
  "SPARK_MASTER_URL=spark://10.0.0.187:7077 python benchmark/run_benchmark.py --mode distributed"
```

**What this does:**
- `SPARK_MASTER_URL=spark://...` — tells the benchmark to connect to the cluster (not local mode)
- `--mode distributed` — runs only Mode 1 (Spark RDD distribution)
- Internally: driver serializes model weights, broadcasts to executors, partitions data, each executor loads models on its local device (GPU if available), runs inference, returns results to driver

### 5.3 Run single GPU mode (on GPU worker directly)

```bash
docker exec -it spark-gpu-worker python benchmark/run_benchmark.py --mode single_gpu
```

**What this does:** Runs Mode 2 directly on the GPU worker container where CUDA is available. All 10 models are loaded on the T4 GPU and run with CUDA streams for parallel inference.

### 5.4 Run hybrid mode (on GPU worker)

```bash
docker exec -it spark-gpu-worker python benchmark/run_benchmark.py --mode hybrid
```

**What this does:** Runs Mode 3 — memory-aware placement splits models between GPU (high priority/fits in VRAM) and CPU (overflow). Uses the T4's 16GB VRAM budget to decide placement.

### 5.5 Run all modes

```bash
# On GPU worker (has GPU for modes 2 and 3, and can connect to Spark for mode 1)
docker exec -it spark-gpu-worker bash -c \
  "SPARK_MASTER_URL=spark://10.0.0.187:7077 python benchmark/run_benchmark.py --mode all"
```

---

## Phase 6: View Metrics

### CloudWatch Dashboard

Access at: `https://ap-south-1.console.aws.amazon.com/cloudwatch/home?region=ap-south-1#dashboards:name=SparkInferenceCluster`

Metrics collected:
| Level | Namespace | Source |
|-------|-----------|--------|
| Host (CPU/mem/disk) | CWAgent | CloudWatch Agent |
| Spark (workers/executors/tasks) | SparkInference/Spark | monitoring/spark_metrics_publisher.py |
| GPU (utilization/memory/temp) | SparkInference/GPU | monitoring/gpu_metrics_publisher.py |
| Benchmark (throughput/latency) | SparkInference/Benchmark | monitoring/benchmark_metrics_publisher.py |

### Spark Master UI

```
http://<MASTER-PUBLIC-IP>:8080
```

Shows: registered workers, running applications, completed applications, executor state.

---

## Phase 7: Cleanup

```powershell
cd D:\Spark_poc\simple_spark_torch_poc\pytorch-spark-inference-platform\deploy\aws-cdk
cdk destroy
```

**What this does:** Deletes all AWS resources created by the stack (EC2 instances, VPC, security groups, S3 bucket, IAM roles, CloudWatch dashboard). The 4-hour auto-shutdown safety net also protects against forgotten instances.

---

## Command Reference

| Command | Where | Purpose |
|---------|-------|---------|
| `cdk bootstrap` | Local | One-time CDK setup per account/region |
| `cdk deploy` | Local | Create/update AWS infrastructure |
| `cdk destroy` | Local | Delete all infrastructure |
| `aws ssm start-session` | Local | Shell access to EC2 instance |
| `docker build` | EC2 | Build inference image from source |
| `docker run --network host` | EC2 | Start Spark master/worker containers |
| `docker run --gpus all` | GPU EC2 | Start container with GPU access |
| `start-master.sh` | Container | Spark script to start master daemon |
| `start-worker.sh spark://IP:7077` | Container | Spark script to register worker with master |
| `spark-submit --master spark://IP:7077` | Container | Submit job to Spark cluster |
| `nvidia-smi` | GPU EC2 | Verify GPU driver is working |
