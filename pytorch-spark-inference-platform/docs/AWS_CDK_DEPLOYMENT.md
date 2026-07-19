# AWS CDK Deployment — Spark Inference Cluster (1 Master + N GPU Workers)

Companion to `docs/ec2_dep.md` (manual `aws ec2 run-instances` walkthrough) and
`docs/RUNNING_AND_DEPLOYMENT.md`. This doc covers the **CDK app** at
`deploy/aws-cdk/` that automates the same cluster topology, plus the new
`monitoring/` package that publishes metrics at 4 levels to CloudWatch.

## Table of Contents

1. [What Gets Deployed](#1-what-gets-deployed)
2. [Metrics Architecture](#2-metrics-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Deploy Steps](#4-deploy-steps)
5. [Stack Parameters & Context](#5-stack-parameters--context)
6. [Building and Publishing the Docker Image](#6-building-and-publishing-the-docker-image)
7. [Viewing Metrics](#7-viewing-metrics)
8. [Running the Benchmark on the Cluster](#8-running-the-benchmark-on-the-cluster)
9. [Cost and Cleanup](#9-cost-and-cleanup)
10. [Security Notes](#10-security-notes)

---

## 1. What Gets Deployed

`deploy/aws-cdk/spark_cluster/spark_cluster_stack.py` (`SparkInferenceClusterStack`) creates:

- A VPC with 1 public subnet, no NAT gateway (keeps cost down; matches the
  public-IP topology already used in `docs/ec2_dep.md`).
- A single security group: all-traffic open between cluster members only,
  no public SSH/UI ports unless you explicitly pass `SshCidr` / `AdminCidr`.
- **1 Spark master** (`t3.medium`, on-demand, no GPU) running the
  `spark-master` container plus `monitoring/spark_metrics_publisher.py` and
  `monitoring/benchmark_metrics_publisher.py`.
- **N Spark GPU workers** (`g4dn.xlarge` by default) in an Auto Scaling Group
  backed by a spot-priced Launch Template, each running the `spark-worker`
  container plus `monitoring/gpu_metrics_publisher.py`. Set the count with
  `-c workerCount=<N>` (default 2).
- An S3 bucket for the pre-built Docker image tarball and benchmark result
  archives.
- A shared IAM role (SSM Session Manager + CloudWatch Agent + S3 + PutMetricData).
- A CloudWatch Dashboard (`SparkInferenceCluster`) with widgets for host,
  Spark master/executor, GPU, and benchmark metrics.
- An auto-shutdown safety net (`AutoShutdownHours`, default 4h) so a forgotten
  GPU worker doesn't run (and bill) forever.

This is a **greenfield CDK app** — no CDK/Terraform/CloudFormation existed in
the repo before. It does not replace `deploy/docker-compose*.yml` (still used
for local dev) or the manual `docs/ec2_dep.md` walkthrough — it automates that
same manual process.

## 2. Metrics Architecture

Four levels, all publishing to CloudWatch (see `monitoring/cloudwatch_publisher.py`
for the shared, credential-graceful publishing helper):

| Level | Publisher | Namespace | Source |
|---|---|---|---|
| Host (CPU/mem/disk) | CloudWatch Agent (installed by user-data) | `CWAgent` | OS-level counters |
| Spark master + executor | `monitoring/spark_metrics_publisher.py` (runs on master) | `SparkInference/Spark` | Spark's own REST APIs: master JSON (`:8080/json/`) and application UI (`:4040/api/v1/applications/.../executors`) |
| GPU (per worker) | `monitoring/gpu_metrics_publisher.py` (runs on each worker) | `SparkInference/GPU` | `nvidia-smi` + `inference/gpu_memory_manager.py`'s `get_current_gpu_usage()` |
| Benchmark results | `monitoring/benchmark_metrics_publisher.py` (runs on master, `--watch` mode) | `SparkInference/Benchmark` | `results/raw_results.json`, produced unchanged by `benchmark/run_benchmark.py` |

All dimensions include `NodeRole` and `InstanceId`; Spark executor metrics add
`ExecutorId`/`Host`, GPU metrics add `GpuIndex`, benchmark metrics add `Mode`
(`single_gpu` / `hybrid_cpu_gpu` / `distributed_gpu`).

Nothing about `benchmark/run_benchmark.py` was changed — the publisher treats
`results/raw_results.json` as a read-only external contract, so local native
runs and `docker compose` dev workflows are unaffected.

## 3. Prerequisites

- Node.js 18+ (CDK v2's jsii runtime requirement — check with `node --version`;
  if you're on an older Node like 14.x, install a newer version just for CDK).
- Python 3.9+ and `pip`.
- AWS CLI v2, configured (`aws configure`) with credentials that can create
  VPCs, EC2, IAM roles, S3 buckets, and CloudWatch dashboards.
- The AWS account must be **CDK-bootstrapped** in the target region
  (`cdk bootstrap aws://<account-id>/<region>`), one-time setup per account/region.
- An EC2 key pair (optional — SSM Session Manager works without one; the
  IAM role already includes `AmazonSSMManagedInstanceCore`).

```bash
cd pytorch-spark-inference-platform/deploy/aws-cdk
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
npm install -g aws-cdk          # or use `npx aws-cdk` per-command
cdk bootstrap                   # one-time per account/region
```

## 4. Deploy Steps

```bash
# 1. Synthesize (sanity check, no AWS calls)
cdk synth

# 2. Deploy with 2 GPU workers (default), no SSH, admin UI open to your IP
cdk deploy -c workerCount=2 --parameters AdminCidr=<your-ip>/32

# 3. Note the outputs: MasterPublicIp, MasterUiUrl, ApplicationUiUrl,
#    ArtifactsBucketNameOutput, WorkerAsgName, DashboardUrl
```

The stack deploys with **0 running Spark processes** until the Docker image
is available in the artifacts bucket (see §6) — the user-data script fetches
`s3://<bucket>/inference/multi-model-inference.tar.gz` on boot and starts the
`spark-master`/`spark-worker` containers only if the image loaded successfully.
If you deploy before uploading the image, just re-run the fetch/load/start
commands manually via SSM Session Manager once the image is in S3 (or
terminate and let the ASG relaunch worker instances after upload).

## 5. Stack Parameters & Context

CloudFormation parameters (`--parameters Name=Value`, changeable per deploy
without touching code):

| Parameter | Default | Purpose |
|---|---|---|
| `KeyName` | *(blank)* | EC2 key pair for SSH. Blank = SSM Session Manager only. |
| `SshCidr` | *(blank)* | CIDR allowed on port 22. Blank = no SSH ingress rule created. |
| `AdminCidr` | *(blank)* | CIDR allowed to reach Spark master UI (8080) / app UI (4040). Blank = UIs stay private. |
| `MasterInstanceType` | `t3.medium` | Driver instance type. |
| `WorkerInstanceType` | `g4dn.xlarge` | GPU worker instance type. |
| `ArtifactsBucketName` | *(auto-generated)* | Explicit S3 bucket name. |
| `AutoShutdownHours` | `4` | Safety-net self-shutdown; `0` disables it. |
| `WorkerSpotMaxPrice` | *(blank)* | Max spot price/hr for workers. Blank = pay up to on-demand price. |

CDK context values (`-c key=value`, resolved at synth time, since they affect
resource counts/IAM policies rather than runtime config):

| Context key | Default | Purpose |
|---|---|---|
| `workerCount` | `2` | Number of GPU worker instances (ASG min=0, max=max(N,1), desired=N). |
| `masterAmiSsmPath` | Ubuntu 22.04 canonical AMI SSM path | Override master AMI lookup. |
| `workerAmiSsmPath` | AWS Deep Learning AMI (GPU, PyTorch, Ubuntu 22.04) SSM path | Override worker AMI lookup — must already have NVIDIA driver + Docker + nvidia-container-toolkit. |
| `account` / `region` | *(from AWS CLI env)* | Explicit target account/region. |

## 6. Building and Publishing the Docker Image

The stack doesn't build the image (CDK asset-bundling a 6-8 GB CUDA image on
every deploy would be slow and expensive) — build once and upload, same
pattern as the manual guide in `docs/ec2_dep.md`:

```bash
cd pytorch-spark-inference-platform
docker build -f deploy/Dockerfile -t multi-model-inference:latest .
docker save multi-model-inference:latest | gzip > multi-model-inference.tar.gz

# Bucket name comes from the stack output ArtifactsBucketNameOutput
aws s3 cp multi-model-inference.tar.gz s3://<ArtifactsBucketNameOutput>/inference/
```

Existing instances don't auto-pull new images. After a re-upload, either:
- Terminate and let the ASG relaunch workers (they re-run user-data), or
- SSH/SSM into each node and re-run the `aws s3 cp ... && docker load` +
  `docker restart spark-master`/`spark-worker` commands manually.

## 7. Viewing Metrics

Open the `DashboardUrl` stack output, or go to **CloudWatch → Dashboards →
SparkInferenceCluster** in the console. Widgets are grouped by level (host,
Spark, GPU, benchmark) as described in §2. Metrics only appear once the
relevant publisher has run at least once:

- Host metrics (`CWAgent` namespace): populated within ~1 minute of any node booting.
- Spark master/executor metrics: populated once `spark-master` is running;
  executor-level rows only appear while a benchmark job is actively holding a
  SparkSession open (the app UI on port 4040 only serves the currently running app).
- GPU metrics: populated once `spark-worker` is running and `nvidia-smi` is reachable.
- Benchmark metrics: populated after the first `run_benchmark.py` run writes
  `results/raw_results.json` (the watcher polls this file's mtime every 30s).

You can also query metrics directly:
```bash
aws cloudwatch get-metric-statistics \
  --namespace SparkInference/GPU --metric-name GpuUtilizationPercent \
  --start-time $(date -u -d '-1 hour' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 --statistics Average Maximum
```

## 8. Running the Benchmark on the Cluster

Same commands as `docs/ec2_dep.md` §8 / `docs/RUNNING_AND_DEPLOYMENT.md` §5 —
the CDK deployment doesn't change how the benchmark is invoked, only how the
cluster is provisioned and monitored:

```bash
# Via SSM Session Manager (no SSH key needed)
aws ssm start-session --target <master-instance-id>

# Once connected to the master instance:
docker exec spark-master python /app/benchmark/run_benchmark.py --mode all \
  --signal-samples 50000 --image-samples 2000 --detection-samples 500 \
  --batch-size 256 --partitions 8   # 8 = 4 cores × 2 workers

# results/raw_results.json is picked up automatically by benchmark_metrics_publisher.py
# within 30s and published to CloudWatch + (if ARTIFACTS_BUCKET is set) archived to S3.
```

## 9. Cost and Cleanup

Same instance types/pricing as `docs/ec2_dep.md` §2/§10 (`t3.medium` master
~$0.04/hr on-demand, `g4dn.xlarge` workers ~$0.16-0.20/hr spot). The
`AutoShutdownHours` safety net (default 4h) self-terminates the Spark process
and shuts down each node even if you forget to tear down the stack — but it
does **not** delete AWS resources, only stops the running instance (worker
ASG instances that self-shutdown will be seen as unhealthy and may be
replaced by the ASG; set `AutoShutdownHours=0` if you don't want that).

To fully tear down:
```bash
cdk destroy
```
This deletes the VPC, security group, ASG (terminating all workers), master
instance, IAM role, and S3 bucket (the bucket has `autoDeleteObjects` enabled
via CDK's removal policy, so no manual emptying is needed — note this means
benchmark result archives in that bucket are deleted too; copy anything you
want to keep out first).

## 10. Security Notes

- **No public SSH/UI by default.** `SshCidr` and `AdminCidr` are opt-in; the
  stack relies on SSM Session Manager for access by default, which requires
  no open inbound ports and is the recommended path for this deployment.
- **No long-lived credentials in the image or containers.** `boto3` inside
  `monitoring/*.py` picks up temporary credentials from the EC2 instance
  role automatically via the instance metadata service (containers run with
  `--network host`, so IMDS is reachable without extra Docker networking).
- **`cloudwatch:PutMetricData` is granted with `resource: "*"`** — this is
  required because the action doesn't support resource-level scoping in IAM;
  it's still limited to that single action, not broad CloudWatch/EC2 access.
- **Spot interruption:** GPU workers use one-time spot requests. A reclaimed
  worker will not auto-rejoin — either accept reduced worker count for that
  run or manually adjust `desired_capacity` back up via the ASG.
