# EC2 GPU Setup Guide - EW PyTorch-Spark Benchmark

## Cost Estimate

| Instance | GPU | Cost/hr | PoC Runtime (~10min) | With Buffer (30min) |
|----------|-----|---------|---------------------|---------------------|
| g4dn.xlarge | 1x T4 16GB | $0.526 | ~$0.09 | ~$0.26 |
| g5.xlarge | 1x A10G 24GB | $1.006 | ~$0.17 | ~$0.50 |
| p3.2xlarge | 1x V100 16GB | $3.06 | ~$0.51 | ~$1.53 |

**Recommended: g4dn.xlarge** — cheapest GPU option, more than enough for this PoC.

**Total estimated cost: $0.10 - $0.30** (assuming you terminate immediately after)

Additional costs:
- EBS storage (8GB gp3): ~$0.002 for 30 minutes
- Data transfer: negligible (results are tiny)

---

## Prerequisites

1. AWS account with GPU quota (see Step 0)
2. AWS CLI configured locally, OR access to AWS Console
3. An SSH key pair in your target region

---

## Step 0: Request GPU Quota (one-time, if needed)

1. Go to: https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas
2. Search: "Running On-Demand G and VT instances"
3. Click → "Request quota increase" → Request **4 vCPUs**
4. Wait for approval (usually 15-30 minutes)

---

## Step 1: Launch EC2 Instance

### Option A: AWS Console (GUI)

1. Go to EC2 → Launch Instance
2. **Name**: `ew-pytorch-benchmark`
3. **AMI**: Search "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"
   - Or use Amazon Linux 2 with NVIDIA drivers
4. **Instance type**: `g4dn.xlarge`
5. **Key pair**: Select or create one
6. **Security group**: Allow SSH (port 22) from your IP
7. **Storage**: 30 GB gp3 (default is fine)
8. Click "Launch Instance"

### Option B: AWS CLI

```bash
# Find the Deep Learning AMI ID for your region (us-east-1 example)
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

# Launch instance
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type g4dn.xlarge \
  --key-name YOUR_KEY_NAME \
  --security-group-ids sg-YOUR_SG_ID \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ew-pytorch-benchmark}]' \
  --query 'Instances[0].InstanceId' \
  --output text
```

---

## Step 2: Connect to Instance

```bash
# Wait for instance to be running, then get public IP
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ew-pytorch-benchmark" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# SSH in
ssh -i your-key.pem ubuntu@<PUBLIC_IP>
```

---

## Step 3: Upload Project

From your local Windows machine:

```powershell
# From the pytorch-spark-ew-poc directory
scp -i your-key.pem -r . ubuntu@<PUBLIC_IP>:~/pytorch-spark-ew-poc/
```

Or clone from a repo if you have one set up.

---

## Step 4: Run the Benchmark

```bash
# On the EC2 instance
cd ~/pytorch-spark-ew-poc

# Verify GPU is visible
nvidia-smi

# Run the deploy script (handles Docker setup + benchmark)
chmod +x deploy_ec2.sh
./deploy_ec2.sh
```

Or run directly without Docker (faster, no build step):

```bash
# The Deep Learning AMI already has PyTorch + CUDA installed
pip install pyspark==3.5.1 numpy==1.26.4 pandas==2.2.2 pyarrow==16.1.0 scikit-learn==1.5.0 matplotlib==3.9.0 tabulate==0.9.0
python benchmark/run_benchmark.py
```

---

## Step 5: View Results

```bash
# On EC2 - view comparison
python benchmark/compare_metrics.py

# Results are in:
#   results/metrics_report.md   (formatted markdown report)
#   results/raw_results.json    (raw data for further analysis)
```

---

## Step 6: Copy Results Back

From your local machine:

```powershell
scp -i your-key.pem ubuntu@<PUBLIC_IP>:~/pytorch-spark-ew-poc/results/* .\results\
```

---

## Step 7: TERMINATE THE INSTANCE (important!)

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ew-pytorch-benchmark" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

# Terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

Or in Console: EC2 → Instances → Select → Instance State → Terminate

**Do not leave the instance running** — g4dn.xlarge costs $12.62/day if left on.

---

## Expected Output

The benchmark produces metrics like:

```
  SCALE: 500,000 signals
  [BASELINE] Single-sample:    ~3,000 samples/sec
  [BASELINE] Batched:          ~200,000 samples/sec (GPU)
  [BASELINE] DataLoader:       ~180,000 samples/sec (GPU)
  [SPARK]    RDD distributed:  ~250,000+ samples/sec (GPU)

  Speedup vs batched: ~1.2-1.5x
  Speedup vs single:  ~80-100x
```

GPU-accelerated batched inference is already fast. Spark's value shows more at:
- Larger scales (millions of signals)
- Multi-GPU/multi-node clusters
- Fault-tolerant streaming workloads

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Quota limit 0 | Request increase (Step 0), wait 15-30 min |
| nvidia-smi not found | Use Deep Learning AMI, not standard Ubuntu |
| Docker permission denied | Run `sudo usermod -aG docker $USER && newgrp docker` |
| OOM errors | Reduce SCALES in run_benchmark.py or use larger instance |
| Slow build | Deep Learning AMI already has PyTorch; skip Docker, run directly |
