# Challenges and Lessons Learned

This document captures all the issues encountered during the AWS deployment of the Spark inference cluster, root causes, and how they were resolved.

---

## Challenge 1: Node.js Version Too Old for CDK

**Error:**
```
process.setSourceMapsEnabled is not a function
TypeError: process.setSourceMapsEnabled is not a function
```

**Root Cause:** System had Node.js v14.17.6 installed. AWS CDK CLI requires Node 18+. The `process.setSourceMapsEnabled()` API was introduced in Node 16.6.

**Resolution:** Installed Node.js v20.18.1 from https://nodejs.org. After install, closed and reopened terminal so the new PATH takes effect.

**Lesson:** Always verify `node --version` before using CDK. CDK is a Node.js application regardless of what language your stack is written in (Python, TypeScript, etc.).

---

## Challenge 2: AWS Credentials Not Found

**Error:**
```
Unable to resolve AWS account to use. It must be either configured when you define your CDK Stack, or through the environment
```

**Root Cause:** AWS CLI was not installed/configured. CDK needs credentials to know which account to deploy to.

**Resolution:** Set credentials as environment variables in PowerShell:
```powershell
$env:AWS_ACCESS_KEY_ID = "AKIA..."
$env:AWS_SECRET_ACCESS_KEY = "..."
$env:AWS_DEFAULT_REGION = "ap-south-1"
```

**Lesson:** CDK reads standard AWS credential sources in order: environment variables > `~/.aws/credentials` > instance metadata. For quick testing, env vars are simplest. For persistent setup, run `aws configure`.

---

## Challenge 3: CDK Bootstrap Missing

**Error:**
```
SSM parameter /cdk-bootstrap/hnb659fds/version not found. Has the environment been bootstrapped?
```

**Root Cause:** CDK requires a one-time "bootstrap" step per account/region that creates staging resources (S3 bucket, IAM roles).

**Resolution:**
```powershell
cdk bootstrap aws://368287210840/ap-south-1
```

**Lesson:** Bootstrap is required once per account+region combination. If you deploy to a new region, you need to bootstrap there too.

---

## Challenge 4: Non-ASCII Characters in Security Group Description

**Error:**
```
Value (Spark inference cluster ? master/worker RPC + UIs) for parameter GroupDescription is invalid.
Character sets beyond ASCII are not supported.
```

**Root Cause:** The Python source code contained an em-dash character (`—`) in the security group description string. AWS EC2 API only accepts ASCII characters in the GroupDescription field.

**Resolution:** Replaced `—` with `-` in the description string.

**Lesson:** Always use plain ASCII in strings that get passed to AWS APIs (resource descriptions, tag values, etc.). Non-ASCII characters in Python comments are fine.

---

## Challenge 5: S3 Bucket Delete Failure During Rollback

**Error:**
```
Unable to marshall request to JSON: Bucket cannot be empty.
DELETE_FAILED: ArtifactsBucket2AAC5544
```

**Root Cause:** When CloudFormation rolled back the failed stack, it tried to delete the S3 bucket. A CloudFormation bug caused the delete to fail with a JSON marshalling error because the bucket name was generated but the resource wasn't fully created.

**Resolution:** Used AWS CLI to force-delete the stack while retaining the problematic bucket:
```powershell
aws cloudformation delete-stack --stack-name SparkInferenceClusterStack --retain-resources ArtifactsBucket2AAC5544 --region us-east-1
```

**Lesson:** `--retain-resources` is the escape hatch when CloudFormation gets stuck on a resource during deletion. The `cdk destroy` command doesn't support this flag — you need the raw AWS CLI.

---

## Challenge 6: Ubuntu AMI Not Resolvable

**Error:**
```
Unable to fetch parameters [/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp3/ami-id] from parameter store
```

**Root Cause:** The CDK stack used Canonical's SSM parameter path to look up the Ubuntu 22.04 AMI. This public SSM parameter was not available/resolvable in the target account.

**Resolution:** Switched to Amazon Linux 2023 using CDK's built-in `ec2.MachineImage.latest_amazon_linux2023()` which always works. Updated bootstrap scripts from `apt-get` to `dnf`/`yum`.

**Lesson:** `MachineImage.latest_amazon_linux2023()` is the most reliable AMI choice for CDK because it uses CDK's internal resolution (no external SSM parameters needed). Third-party AMI SSM parameters (Canonical, Deep Learning AMIs) may not be available in all accounts/regions.

---

## Challenge 7: Disk Space Exhaustion on Master (30GB)

**Error:**
```
no space left on device
```

**Root Cause:** The master instance had only 30GB EBS. After building the Docker image (~8GB for CUDA + PyTorch + Spark), there wasn't enough space to `docker save` the image for sharing with workers.

**Resolution:** Increased master disk to 100GB in the CDK stack. Also used streaming pipe to S3 to avoid local disk usage:
```bash
docker save multi-model-inference:latest | aws s3 cp - s3://$BUCKET/image.tar
```

**Lesson:** GPU/ML Docker images are large (5-10GB). Always allocate at least 100GB for machines that build Docker images. Use streaming (`|`) to avoid doubling disk usage.

---

## Challenge 8: g4dn.xlarge Not Available in us-east-1a

**Error:**
```
We currently do not have sufficient g4dn.xlarge capacity in the Availability Zone you requested (us-east-1a)
```

**Root Cause:** The VPC was configured with `max_azs=1`, locking to a single AZ. GPU instance capacity varies by AZ.

**Resolution:** Changed `max_azs=3` so the VPC spans multiple AZs, giving CloudFormation more options for placing GPU instances.

**Lesson:** Always use multiple AZs when deploying GPU instances. GPU capacity is not uniform across AZs. Also consider using spot instances with multi-AZ allocation strategies for better availability.

---

## Challenge 9: GPU vCPU Quota = 0 in us-east-1

**Error:**
```
You have requested more vCPU capacity than your current vCPU limit of 0 allows for the instance bucket that the specified instance type belongs to.
```

**Root Cause:** New AWS accounts have a default vCPU limit of 0 for G-type (GPU) instances. This is a safety measure by AWS.

**Resolution:** Switched to `ap-south-1` (Mumbai) region where the account already had GPU quota. Alternatively, request a quota increase via Service Quotas console (search for "Running On-Demand G and VT instances").

**Lesson:** Check GPU quotas before deploying. Go to Service Quotas → EC2 → "Running On-Demand G and VT instances". Request increases early — they're usually auto-approved for small amounts (4-8 vCPUs) within minutes.

---

## Challenge 10: NVIDIA Driver Installation Failed (Manual .run File)

**Error:**
```
ERROR: Unable to load the kernel module 'nvidia.ko'. This happens most frequently when this kernel module was built against the wrong or improperly configured kernel sources
```

**Root Cause:** Manually downloading and running `NVIDIA-Linux-x86_64-535.183.01.run` failed because the kernel headers/devel packages didn't match the running kernel version exactly, or GCC version mismatch.

**Resolution:** Used NVIDIA's official RPM repository for Amazon Linux 2023 instead:
```bash
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
dnf module install -y nvidia-driver:latest-dkms
```

**Lesson:** Never manually compile NVIDIA drivers on cloud instances. Always use the distro-specific package manager (dnf/yum for Amazon Linux, apt for Ubuntu). NVIDIA provides pre-built packages for all major distros that handle kernel module compilation correctly via DKMS.

---

## Challenge 11: Docker `--gpus all` Failed (Container Status: "Created")

**Error:**
```
CONTAINER ID   IMAGE   COMMAND   CREATED   STATUS: Created   (never starts)
```

**Root Cause:** The `--gpus all` flag requires `nvidia-container-toolkit` to be installed and Docker's runtime to be configured. Without it, Docker can't map GPUs into the container and the container fails to start silently.

**Resolution:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
dnf install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

**Lesson:** The NVIDIA driver alone is not enough for Docker GPU access. You need the full chain: NVIDIA driver → nvidia-container-toolkit → Docker runtime configuration → restart Docker. Verify with `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`.

---

## Challenge 12: Spark Not Installed in Docker Image

**Error:**
```
bash: line 1: /sbin/start-master.sh: No such file or directory
```

**Root Cause:** The original Dockerfile only installed Python, Java, and PyTorch — but not Apache Spark. The `start-master.sh` script is part of Spark's sbin directory.

**Resolution:** Added Spark 3.5.1 installation to the Dockerfile:
```dockerfile
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"
RUN wget -q "https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz" ...
```

**Lesson:** When running Spark in containers, the Spark binaries must be baked into the image. The `start-master.sh` and `start-worker.sh` scripts come from `$SPARK_HOME/sbin/`.

---

## Challenge 13: Benchmark Only Ran on CPU (Driver Has No GPU)

**Error:**
```
[WARN] CUDA not available, falling back to CPU
```

**Root Cause:** `spark-submit` runs the benchmark script on the **driver** (master node), which has no GPU. The benchmark's "Single GPU" and "Hybrid" modes check `torch.cuda.is_available()` locally on the driver.

**Resolution:** 
- Mode 2 (Single GPU) and Mode 3 (Hybrid) must run directly on the GPU worker: `docker exec -it spark-gpu-worker python benchmark/run_benchmark.py --mode single_gpu`
- Mode 1 (Distributed) connects to Spark cluster and distributes work to executors (which run on the GPU worker)
- Fixed `create_spark_session()` to read `SPARK_MASTER_URL` env var instead of hardcoding `local[4]`

**Lesson:** In Spark, the driver orchestrates but doesn't do the heavy lifting — executors do. For GPU inference, the executors must be on GPU machines. Single-GPU and Hybrid modes don't use Spark at all, so they must run where the GPU is.

---

## Challenge 14: Spark Distributed Mode Used local[4] Instead of Cluster

**Error:** Distributed mode created a local Spark session instead of connecting to the cluster master.

**Root Cause:** `create_spark_session()` was hardcoded with `.master("local[4]")`, ignoring the `--master` flag passed to `spark-submit`.

**Resolution:** Modified `create_spark_session()` to check `SPARK_MASTER_URL` and `SPARK_MASTER` environment variables before falling back to local mode:
```python
master_url = os.environ.get("SPARK_MASTER_URL") or os.environ.get("SPARK_MASTER")
if not master_url:
    master_url = f"local[{num_cores}]"
```

**Lesson:** When using `spark-submit`, SparkSession inherits the master from the submit command. But if you create a SparkSession explicitly in code with `.master(...)`, it overrides the spark-submit setting. Either don't set `.master()` at all (let spark-submit control it), or read from environment.

---

## Challenge 15: Clock Skew Causing Signature Errors

**Error:**
```
SignatureDoesNotMatch: Signature expired: 20260717T155812Z is now earlier than 20260718T151716Z
```

**Root Cause:** The system clock was off by more than 5 minutes. AWS API signatures include a timestamp, and AWS rejects requests where the timestamp differs from server time by more than 5 minutes.

**Resolution:** Synced the system clock via Windows Settings → Date & time → toggle "Set time automatically" off then on.

**Lesson:** AWS APIs are time-sensitive. If your machine clock drifts (common in VMs or after hibernation), all AWS calls fail with signature errors. Keep time sync enabled.

---

## Challenge 16: `$ARTIFACTS_BUCKET` Empty on EC2 Instances

**Error:**
```
Invalid bucket name "": Bucket name must match the regex...
```

**Root Cause:** Environment variables set in user-data don't persist across new shell sessions. Every time you SSM into the instance or run `sudo su -`, you start a fresh shell without the variable.

**Resolution:** Always set it manually at the start of a session:
```bash
export ARTIFACTS_BUCKET=sparkinferenceclusterstack-artifactsbucket2aac5544-zlcd7fgifndh
```

The CDK stack also writes it to `/etc/environment` for persistence, but this only applies to new login shells.

**Lesson:** User-data environment variables are available only during the initial boot script execution. For interactive sessions, either source `/etc/environment` or set them manually. Consider writing to `/etc/profile.d/spark.sh` for automatic loading.

---

## Summary: Key Takeaways

1. **GPU infrastructure requires multiple prerequisites:** Driver → Container Toolkit → Docker Runtime Config → Restart Docker. Miss any step and `--gpus all` silently fails.

2. **CDK/CloudFormation has sharp edges:** Non-ASCII strings, AMI SSM parameter availability, and rollback bugs can block deployment. Always use ASCII, prefer built-in AMI lookups, and know the `--retain-resources` escape hatch.

3. **Disk space matters for ML workloads:** Docker images with CUDA + PyTorch are 5-10GB. Allocate 100GB+ for build machines.

4. **GPU capacity varies by AZ and region:** Use multiple AZs and check quotas before deploying. Have a fallback region ready.

5. **Spark driver vs executor distinction is critical:** The driver coordinates; executors do the work. GPU code must run on executors (or directly on GPU machines for non-distributed modes).

6. **Streaming pipes save disk:** `docker save | aws s3 cp -` and `aws s3 cp - | docker load` avoid writing multi-GB files to disk.

7. **Always test locally first:** Validate the benchmark runs in `local[4]` mode before trying cluster mode. This catches code bugs without infrastructure complexity.


---

## Challenge 17: Executor Cannot Find App Modules (ModuleNotFoundError)

**Error:**
```
ModuleNotFoundError: No module named 'inference'

File "/opt/spark/python/lib/pyspark.zip/pyspark/serializers.py", line 472, in loads
    return cloudpickle.loads(obj, encoding=encoding)
```

**Context:** Running the distributed benchmark from the master (`spark-submit --master spark://10.0.0.187:7077`). The driver (master) serializes the `infer_on_partition` function via `cloudpickle` and sends it to the executor on the GPU worker. When the executor tries to deserialize the function, it fails because the `inference` module is not on the executor's Python path.

**Root Cause:** The Spark executor on the GPU worker runs inside a Python subprocess spawned by Spark's JVM. This subprocess inherits Spark's default Python path (`/opt/spark/python/lib/pyspark.zip`), but does NOT inherit the container's working directory (`/app`) where our code lives. The `infer_on_partition` function references classes from the `inference` and `models` modules, and `cloudpickle` tries to import those modules during deserialization — but they aren't on `sys.path` in the executor's subprocess.

**Why this is tricky:**
- In `local[N]` mode, driver and executors share the same process/filesystem, so imports work fine
- In cluster mode, executors are separate processes on remote machines with their own `sys.path`
- Even though the Docker image has the code at `/app`, the Spark executor subprocess doesn't know to look there

**Resolution (two changes):**

1. Set `PYTHONPATH` on executors via Spark config:
```python
builder = builder.config(
    "spark.executorEnv.PYTHONPATH",
    "/app:/app/inference:/app/models:/app/data"
)
```

2. Add explicit `sys.path` manipulation inside the worker function (belt-and-suspenders):
```python
def infer_on_partition(partition_idx):
    import sys
    import os
    for p in ["/app", "/app/inference", "/app/models", "/app/data"]:
        if p not in sys.path:
            sys.path.insert(0, p)
    # ... rest of function
```

**Alternative approaches (not used):**
- `spark-submit --py-files /app/inference.zip,/app/models.zip` — would work but requires zipping each module
- `sc.addPyFile("inference.zip")` — uploads Python files to executors at runtime
- Setting `spark.submit.pyFiles` config — same as `--py-files` but via config

**Lesson:** In Spark cluster mode, always explicitly configure `spark.executorEnv.PYTHONPATH` or use `--py-files` to ship application code to executors. Don't assume executors can find your modules just because they're in the Docker image — the executor subprocess has a minimal `sys.path` by default.
