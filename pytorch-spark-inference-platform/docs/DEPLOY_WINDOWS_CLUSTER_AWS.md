# Deploying & Testing the Windows Spark Cluster on AWS

Covers `WindowsSparkClusterStack` (`deploy/aws-cdk/spark_cluster/windows_spark_cluster_stack.py`)
— a native, Docker-free Windows Server Spark cluster (1 master + 1 GPU worker),
built specifically so the BYOM framework (`submit_job.py`, `models/plugins/`)
can be tested on real AWS GPU hardware. See `docs/BRING_YOUR_OWN_MODEL.md` for
the plugin framework itself; this doc is only about standing up and using the
cluster it runs on.

This creates **real, billed AWS resources**. Read §7 (cost) before deploying.

---

## 1. Prerequisites

- AWS CLI authenticated: `aws sts get-caller-identity` must succeed (re-auth
  via your org's SSO flow if it says the session expired — not something
  scriptable here).
- Node.js 18+ (`node --version`) — required by the CDK CLI.
- Python with the CDK app's own dependencies installed:
  ```powershell
  cd pytorch-spark-inference-platform/deploy/aws-cdk
  python -m pip install -r requirements.txt
  ```
- An AWS account already CDK-bootstrapped in your target region (one-time,
  see §2).

---

## 2. One-time bootstrap

```powershell
cd pytorch-spark-inference-platform/deploy/aws-cdk
npx aws-cdk bootstrap aws://<ACCOUNT_ID>/<REGION> --context account=<ACCOUNT_ID> --context region=<REGION>
```

Skip this if the account/region has already been bootstrapped for CDK before
(e.g. because `SparkInferenceClusterStack` or `GpuBenchmarkStack` were
deployed here previously).

---

## 3. Sanity-check before touching AWS: synth only

```powershell
npx aws-cdk synth WindowsSparkClusterStack --context account=<ACCOUNT_ID> --context region=<REGION>
```

This renders the CloudFormation template locally — no resources are created.
If this errors with something like *"Need to perform AWS calls... no
credentials configured"*, that's almost certainly `GpuBenchmarkStack`'s AMI
lookup (a **different, already-existing** stack in the same CDK app — all
three sibling stacks' Python code runs on every `cdk synth`/`cdk deploy`
invocation regardless of which one you name) hitting a stale cached AMI
lookup for a different account. It is unrelated to `WindowsSparkClusterStack`
itself, which deliberately uses an SSM-parameter AMI reference that needs no
synth-time AWS call. If it blocks you, either supply valid credentials for
that account or run `cdk context --clear` to force a fresh lookup once
authenticated.

---

## 4. Deploy

```powershell
npx aws-cdk deploy WindowsSparkClusterStack `
    --context account=<ACCOUNT_ID> --context region=<REGION> `
    --parameters AdminCidr=<your-ip>/32
```

- `AdminCidr` is optional — leave it off and the Spark UIs (8080/4040) and
  RDP (3389) stay closed to the internet; you'll only reach the cluster via
  SSM Session Manager (§6), which needs no inbound port at all. Set it if you
  want to open the Spark master UI in a browser from your own IP.
- Other parameters you can override: `MasterInstanceType` (default
  `m5.xlarge`), `WorkerInstanceType` (default `g4dn.xlarge`),
  `AutoShutdownHours` (default `4`, `0` disables the safety net),
  `KeyPairName` (only needed if you want RDP password retrieval instead of
  SSM).

Deploy takes several minutes (VPC + 2 EC2 instances + IAM + S3 + dashboard).
Note the outputs — you'll need `ArtifactsBucketName`, `MasterInstanceId`,
`GpuWorkerInstanceId`, `MasterPrivateIp`, and `MasterUiUrl`.

At this point the instances are running but **have no project code yet** —
their boot script tries to fetch `project.zip` from the (still-empty) bucket,
finds nothing, and skips straight to installing Java/Python/Spark/CloudWatch
Agent only. That's expected; continue to §5.

---

## 5. Upload the project and start Spark

```powershell
cd pytorch-spark-inference-platform
Compress-Archive -Path * -DestinationPath project.zip -Force
aws s3 cp project.zip s3://<ArtifactsBucketName>/inference/project.zip
```

Now re-run the boot script on both instances via SSM (no reboot needed — the
script is idempotent and safe to re-run):

```powershell
aws ssm send-command `
    --instance-ids <MasterInstanceId> <GpuWorkerInstanceId> `
    --document-name "AWS-RunPowerShellScript" `
    --parameters commands="schtasks /run /tn SparkBootstrap"
```

Give it a few minutes — the master installs Java/Python/Spark then starts
the master + a local CPU worker; the GPU worker additionally downloads and
silently installs the NVIDIA driver, which **triggers one automatic reboot**.
That's expected and self-healing: the same `SparkBootstrap` scheduled task is
registered to run at every startup, so after the reboot it re-runs, finds the
driver already present (`nvidia-smi` succeeds), and proceeds to launch the
Spark worker.

Check progress:

```powershell
aws ssm send-command --instance-ids <GpuWorkerInstanceId> --document-name "AWS-RunPowerShellScript" `
    --parameters commands="Get-Command nvidia-smi -ErrorAction SilentlyContinue; nvidia-smi"
```

---

## 6. Verify the cluster is up

**Master UI** (only if you set `AdminCidr`): open the `MasterUiUrl` output in
a browser — you should see 2 workers registered (the master's own local CPU
worker, and the GPU worker).

**Without `AdminCidr`**, use SSM Session Manager instead (works with zero
inbound ports open):

```powershell
aws ssm start-session --target <MasterInstanceId>
```

This drops you into a PowerShell session on the master itself. From there:

```powershell
Invoke-RestMethod http://localhost:8080/json/ | Select-Object -ExpandProperty workers | Select-Object host, cores
```

should list both workers.

---

## 7. Run a BYOM job

Still inside the SSM session on the master (this is the supported way to
submit jobs — port 7077 is only open *within* the cluster's security group,
not to the internet, by design):

```powershell
cd C:\app
C:\Python312\python.exe submit_job.py --model example_mlp --samples 2000 --mode hybrid `
    --master spark://<MasterPrivateIp>:7077
```

Expect to see one executor log line with `device=cpu` (the master's local
worker) and one with `device=cuda` (the GPU worker) — confirming the GPU
worker's native NVIDIA driver + CUDA torch install actually works, no
container layer involved. `submit_job.py` also writes
`results/example_mlp_<timestamp>.json` on the master and — since
`ARTIFACTS_BUCKET` is set machine-wide by the boot script — uploads it to
`s3://<bucket>/results/` automatically.

---

## 8. Pull results back to your local machine

From your own machine (not the SSM session):

```powershell
cd pytorch-spark-inference-platform
.\deploy\aws-cdk\pull_results.ps1 -BucketName <ArtifactsBucketName>
```

New files appear under `.\results\`.

---

## 9. Later: testing a real model (e.g. NER) without redeploying

1. Locally: add `models/plugins/ner_plugin.py` (+ weights), add its entry to
   `models/plugins/manifest.json` — see `docs/BRING_YOUR_OWN_MODEL.md`.
2. Re-zip and re-upload:
   ```powershell
   Compress-Archive -Path * -DestinationPath project.zip -Force
   aws s3 cp project.zip s3://<ArtifactsBucketName>/inference/project.zip
   ```
3. Refresh code on the running instances and relaunch Spark:
   ```powershell
   aws ssm send-command --instance-ids <MasterInstanceId> <GpuWorkerInstanceId> `
       --document-name "AWS-RunPowerShellScript" `
       --parameters commands="schtasks /run /tn SparkBootstrap"
   ```
4. Re-run `submit_job.py --model ner_plugin ...` as in §7. No `cdk deploy`
   needed for this whole cycle.

---

## 10. Teardown

```powershell
npx aws-cdk destroy WindowsSparkClusterStack --context account=<ACCOUNT_ID> --context region=<REGION>
```

Do this even if `AutoShutdownHours` already stopped the instances —
self-shutdown only powers them off, it doesn't delete them (still billed for
EBS storage) or the S3 bucket/CloudWatch dashboard.

---

## 11. Cost (confirm you're comfortable before §4)

`MasterInstanceType` default `m5.xlarge` (Windows) + `WorkerInstanceType`
default `g4dn.xlarge` (Windows) ≈ **$1.08/hr combined** (Windows Server
licensing adds ~$0.18/hr per instance over Linux pricing), plus ~$0.03/hr for
the 250GB of gp3 EBS across both instances. With the default
`AutoShutdownHours=4`, a forgotten-running cluster tops out around **$4.35**
in compute before it self-shuts-down — but remember §10, EBS/S3 keep billing
until you `cdk destroy`.

---

## 12. Is this airgapped-compatible?

**No — this specific stack is not, by design.** Its whole point is a
convenient AWS test bed, and its boot script pulls Java, Python, Spark, the
NVIDIA driver, the CloudWatch Agent, and pip/PyPI packages (including a
~2GB CUDA torch wheel) live from the internet every time it (re-)runs. That's
fine on AWS, which has internet egress by default, but it would not work
unmodified inside a network with no internet access.

This is a separate question from whether *the rest of the repo* — the
Linux/Docker platform this Windows stack was built alongside — is airgapped-
ready. See the answer in the accompanying chat response for that; short
version: yes, the Linux/Docker path already has a documented airgapped
workflow (`docs/AIRGAPPED_5NODE_DEPLOYMENT.md`, `docs/AIRGAPPED_TROUBLESHOOTING.md`),
but it works by building the Docker image on an internet-connected machine
and transferring the tarball — not by running the repo directly inside the
airgapped network.
