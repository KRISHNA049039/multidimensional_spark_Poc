"""
WindowsSparkClusterStack - 2 EC2 instances (Windows Server 2022, no Docker):
  1. Master (CPU) - Spark master + driver + CPU worker, native JVM process
  2. GPU Worker (g4dn.xlarge) - Spark worker with GPU, native JVM process

Docker Desktop does not run on Windows Server, and WSL2/Hyper-V nested
virtualization is unavailable on any right-sized GPU EC2 instance family
(g4dn/g5/g6/p-series only supports it on oversized .metal instances) — so
both nodes run the Spark master/worker JVM processes directly via
spark-class2.cmd instead of containers, mirroring what already runs
successfully on native Windows locally (Temurin JDK 17 + PySpark + torch,
no Docker).

Each node's bootstrap logic is written to disk as one idempotent PowerShell
script and registered as an "At startup" Scheduled Task (in addition to being
invoked once immediately from UserData). This makes the GPU worker's
NVIDIA-driver-install reboot self-healing: the script re-runs after reboot,
finds the driver already present, and proceeds straight to launching the
Spark worker process.

Metrics are published to CloudWatch at host/Spark/GPU/benchmark levels,
mirroring spark_cluster_stack.py's Linux/Docker cluster (same CWAgent /
SparkInference/Spark / SparkInference/GPU / SparkInference/Benchmark
namespaces and metric names, so the same style of dashboard widgets apply).
"""
from aws_cdk import (
    Stack,
    CfnParameter,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Tags,
    Fn,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    aws_cloudwatch as cloudwatch,
)
from constructs import Construct

SPARK_MASTER_RPC_PORT = 7077
SPARK_MASTER_UI_PORT = 8080
SPARK_APP_UI_PORT = 4040
SPARK_RDP_PORT = 3389
SPARK_VERSION = "3.5.1"


class WindowsSparkClusterStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------
        admin_cidr = CfnParameter(
            self, "AdminCidr",
            type="String",
            default="",
            description="CIDR allowed to reach Spark UIs (8080/4040) and RDP (3389), e.g. '203.0.113.5/32'.",
        )
        master_instance_type = CfnParameter(
            self, "MasterInstanceType", type="String", default="m5.xlarge",
            description="Instance type for Spark master (CPU, Windows). Also acts as a CPU worker.",
        )
        worker_instance_type = CfnParameter(
            self, "WorkerInstanceType", type="String", default="g4dn.xlarge",
            description="Instance type for the Windows GPU worker.",
        )
        auto_shutdown_hours = CfnParameter(
            self, "AutoShutdownHours", type="Number", default=4, min_value=0,
            description="Safety net: instances shut down after this many hours (0 = disabled).",
        )
        key_pair_name = CfnParameter(
            self, "KeyPairName", type="String", default="",
            description="EC2 key pair for RDP password retrieval (optional - SSM Session Manager works without one).",
        )

        # ---------------------------------------------------------------
        # VPC
        # ---------------------------------------------------------------
        vpc = ec2.Vpc(
            self, "WindowsSparkVpc",
            max_azs=3,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24,
                ),
            ],
        )

        # ---------------------------------------------------------------
        # Security Group
        # ---------------------------------------------------------------
        sg = ec2.SecurityGroup(
            self, "WindowsSparkClusterSg", vpc=vpc,
            description="Windows Spark inference cluster - master and worker traffic",
            allow_all_outbound=True,
        )
        sg.add_ingress_rule(sg, ec2.Port.all_traffic(), "Inter-node Spark traffic")

        admin_ui_rule_8080 = ec2.CfnSecurityGroupIngress(
            self, "AdminUi8080",
            group_id=sg.security_group_id,
            ip_protocol="tcp", from_port=SPARK_MASTER_UI_PORT, to_port=SPARK_MASTER_UI_PORT,
            cidr_ip=admin_cidr.value_as_string,
        )
        admin_ui_rule_8080.cfn_options.condition = _non_empty_condition(self, "HasAdminCidr8080", admin_cidr)

        admin_ui_rule_4040 = ec2.CfnSecurityGroupIngress(
            self, "AdminUi4040",
            group_id=sg.security_group_id,
            ip_protocol="tcp", from_port=SPARK_APP_UI_PORT, to_port=SPARK_APP_UI_PORT,
            cidr_ip=admin_cidr.value_as_string,
        )
        admin_ui_rule_4040.cfn_options.condition = _non_empty_condition(self, "HasAdminCidr4040", admin_cidr)

        admin_rdp_rule = ec2.CfnSecurityGroupIngress(
            self, "AdminRdp",
            group_id=sg.security_group_id,
            ip_protocol="tcp", from_port=SPARK_RDP_PORT, to_port=SPARK_RDP_PORT,
            cidr_ip=admin_cidr.value_as_string,
        )
        admin_rdp_rule.cfn_options.condition = _non_empty_condition(self, "HasAdminCidrRdp", admin_cidr)

        # ---------------------------------------------------------------
        # S3 Bucket
        # ---------------------------------------------------------------
        artifacts_bucket = s3.Bucket(
            self, "ArtifactsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
        )

        # ---------------------------------------------------------------
        # IAM Role
        # ---------------------------------------------------------------
        role = iam.Role(
            self, "WindowsSparkNodeRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="Windows Spark inference nodes - SSM, CloudWatch, S3, EC2 describe",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchAgentServerPolicy"),
            ],
        )
        artifacts_bucket.grant_read_write(role)
        role.add_to_policy(iam.PolicyStatement(
            actions=["cloudwatch:PutMetricData"],
            resources=["*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["ec2:DescribeInstances", "ec2:DescribeVolumes", "ec2:ModifyVolume"],
            resources=["*"],
        ))

        # ---------------------------------------------------------------
        # AMI - always-current SSM public parameter (avoids account-bound
        # MachineImage.lookup()/context-cache issues)
        # ---------------------------------------------------------------
        windows_ami = ec2.MachineImage.from_ssm_parameter(
            "/aws/service/ami-windows-latest/Windows_Server-2022-English-Full-Base",
            os=ec2.OperatingSystemType.WINDOWS,
        )

        # ---------------------------------------------------------------
        # Master Instance (CPU - also runs as Spark worker)
        # ---------------------------------------------------------------
        master_script = _common_script(
            artifacts_bucket.bucket_name, self.region, auto_shutdown_hours.value_as_string, is_gpu=False,
        ) + "\n" + _master_tail_script()

        master_user_data = ec2.UserData.for_windows()
        master_user_data.add_commands(*_script_writer_commands(master_script))

        master = ec2.Instance(
            self, "SparkMasterWindows",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType(master_instance_type.value_as_string),
            machine_image=windows_ami,
            security_group=sg,
            role=role,
            user_data=master_user_data,
            block_devices=[ec2.BlockDevice(
                device_name="/dev/sda1",
                volume=ec2.BlockDeviceVolume.ebs(100, volume_type=ec2.EbsDeviceVolumeType.GP3),
            )],
            associate_public_ip_address=True,
        )
        Tags.of(master).add("Name", "spark-master-windows")
        Tags.of(master).add("Role", "spark-master")
        master.instance.add_property_override(
            "KeyName", _if_non_empty(self, "HasKeyPairMaster", key_pair_name))

        # ---------------------------------------------------------------
        # GPU Worker Instance
        # ---------------------------------------------------------------
        gpu_worker_script = _common_script(
            artifacts_bucket.bucket_name, self.region, auto_shutdown_hours.value_as_string, is_gpu=True,
        ) + "\n" + _gpu_worker_tail_script(master.instance_private_ip)

        gpu_worker_user_data = ec2.UserData.for_windows()
        gpu_worker_user_data.add_commands(*_script_writer_commands(gpu_worker_script))

        gpu_worker = ec2.Instance(
            self, "SparkGpuWorkerWindows",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType(worker_instance_type.value_as_string),
            machine_image=windows_ami,
            security_group=sg,
            role=role,
            user_data=gpu_worker_user_data,
            block_devices=[ec2.BlockDevice(
                device_name="/dev/sda1",
                volume=ec2.BlockDeviceVolume.ebs(150, volume_type=ec2.EbsDeviceVolumeType.GP3),
            )],
            associate_public_ip_address=True,
        )
        Tags.of(gpu_worker).add("Name", "spark-gpu-worker-windows")
        Tags.of(gpu_worker).add("Role", "spark-worker")
        gpu_worker.instance.add_property_override(
            "KeyName", _if_non_empty(self, "HasKeyPairWorker", key_pair_name))

        # ---------------------------------------------------------------
        # CloudWatch Dashboard
        # ---------------------------------------------------------------
        dashboard = cloudwatch.Dashboard(self, "WindowsSparkClusterDashboard",
                                          dashboard_name="SparkInferenceCluster-Windows")
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown="# Windows Spark Inference Cluster - Master / GPU Worker / Benchmark",
                width=24, height=1,
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Host CPU %",
                left=[cloudwatch.Metric(namespace="CWAgent", metric_name="cpu_usage_active",
                                         statistic="Average", period=Duration.minutes(1))],
                width=8, height=6,
            ),
            cloudwatch.GraphWidget(
                title="Host Memory %",
                left=[cloudwatch.Metric(namespace="CWAgent", metric_name="mem_used_percent",
                                         statistic="Average", period=Duration.minutes(1))],
                width=8, height=6,
            ),
            cloudwatch.GraphWidget(
                title="Host Disk %",
                left=[cloudwatch.Metric(namespace="CWAgent", metric_name="disk_used_percent",
                                         statistic="Average", period=Duration.minutes(1))],
                width=8, height=6,
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Spark: Active Workers",
                left=[cloudwatch.Metric(namespace="SparkInference/Spark", metric_name="ActiveWorkers",
                                         statistic="Maximum", period=Duration.minutes(1))],
                width=8, height=6,
            ),
            cloudwatch.GraphWidget(
                title="Spark: Executor Active Tasks",
                left=[cloudwatch.Metric(namespace="SparkInference/Spark", metric_name="ExecutorActiveTasks",
                                         statistic="Sum", period=Duration.minutes(1))],
                width=8, height=6,
            ),
            cloudwatch.GraphWidget(
                title="Spark: Executor Completed Tasks",
                left=[cloudwatch.Metric(namespace="SparkInference/Spark", metric_name="ExecutorCompletedTasks",
                                         statistic="Sum", period=Duration.minutes(1))],
                width=8, height=6,
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="GPU Utilization %",
                left=[cloudwatch.Metric(namespace="SparkInference/GPU", metric_name="GpuUtilizationPercent",
                                         statistic="Average", period=Duration.minutes(1))],
                width=8, height=6,
            ),
            cloudwatch.GraphWidget(
                title="GPU Memory Used (MB)",
                left=[cloudwatch.Metric(namespace="SparkInference/GPU", metric_name="GpuMemoryUsedMb",
                                         statistic="Average", period=Duration.minutes(1))],
                width=8, height=6,
            ),
            cloudwatch.GraphWidget(
                title="GPU Temperature (C)",
                left=[cloudwatch.Metric(namespace="SparkInference/GPU", metric_name="GpuTemperatureC",
                                         statistic="Maximum", period=Duration.minutes(1))],
                width=8, height=6,
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Benchmark Throughput (samples/sec)",
                left=[cloudwatch.Metric(namespace="SparkInference/Benchmark",
                                         metric_name="ThroughputSamplesPerSec",
                                         statistic="Maximum", period=Duration.minutes(5))],
                width=12, height=6,
            ),
            cloudwatch.GraphWidget(
                title="Benchmark Elapsed Time (sec)",
                left=[cloudwatch.Metric(namespace="SparkInference/Benchmark",
                                         metric_name="ElapsedTimeSec",
                                         statistic="Maximum", period=Duration.minutes(5))],
                width=12, height=6,
            ),
        )

        # ---------------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------------
        CfnOutput(self, "MasterInstanceId", value=master.instance_id)
        CfnOutput(self, "MasterPublicIp", value=master.instance_public_ip)
        CfnOutput(self, "MasterPrivateIp", value=master.instance_private_ip)
        CfnOutput(self, "GpuWorkerInstanceId", value=gpu_worker.instance_id)
        CfnOutput(self, "GpuWorkerPublicIp", value=gpu_worker.instance_public_ip)
        CfnOutput(self, "GpuWorkerPrivateIp", value=gpu_worker.instance_private_ip)
        CfnOutput(self, "MasterUiUrl",
                  value=f"http://{master.instance_public_ip}:{SPARK_MASTER_UI_PORT}")
        CfnOutput(self, "ArtifactsBucketName", value=artifacts_bucket.bucket_name)
        CfnOutput(self, "DashboardUrl",
                  value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
                        f"?region={self.region}#dashboards:name=SparkInferenceCluster-Windows")
        CfnOutput(self, "SubmitJobExample",
                  value=f"python submit_job.py --model example_mlp --mode hybrid "
                        f"--master spark://{master.instance_private_ip}:7077")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _non_empty_condition(scope: Construct, cid: str, param: CfnParameter):
    from aws_cdk import CfnCondition
    return CfnCondition(scope, cid,
                        expression=Fn.condition_not(Fn.condition_equals(param.value_as_string, "")))


def _if_non_empty(scope: Construct, cid: str, param: CfnParameter):
    cond = _non_empty_condition(scope, cid, param)
    return Fn.condition_if(cond.logical_id, param.value_as_string, Fn.ref("AWS::NoValue"))


def _script_writer_commands(script_body: str, script_path: str = r"C:\spark\bootstrap.ps1"):
    """UserData commands that write `script_body` to disk as a single-quoted
    (literal, non-interpolating) here-string, register it as an "At startup"
    Scheduled Task, and run it once immediately so setup doesn't wait for a
    reboot. Re-running it after a reboot (e.g. post NVIDIA-driver-install) is
    what makes the GPU worker's bootstrap self-healing — the script's own
    idempotency checks (Test-Path / Get-Command) skip already-done steps.
    """
    return [
        r'New-Item -ItemType Directory -Force -Path C:\spark, C:\spark\logs | Out-Null',
        f"@'\n{script_body}\n'@ | Set-Content -Path '{script_path}' -Encoding UTF8",
        'schtasks /create /tn SparkBootstrap /sc onstart /ru SYSTEM /rl HIGHEST '
        f'/tr "powershell.exe -NoProfile -ExecutionPolicy Bypass -File {script_path}" /f',
        'schtasks /run /tn SparkBootstrap',
    ]


def _common_script(artifacts_bucket_name: str, region: str, auto_shutdown_hours: str, is_gpu: bool) -> str:
    """Idempotent PowerShell: Java, Python, Spark, project code, CloudWatch
    Agent, auto-shutdown. Safe to re-run (e.g. after a reboot) — every step
    checks whether it's already done first.
    """
    torch_line = (
        r'& $PythonExe -m pip install --quiet --force-reinstall torch==2.6.0 torchvision==0.21.0 '
        r'--index-url https://download.pytorch.org/whl/cu126'
        if is_gpu else ""
    )
    return f"""
$ErrorActionPreference = 'Continue'
$Bucket = "{artifacts_bucket_name}"
$Region = "{region}"
[Environment]::SetEnvironmentVariable('ARTIFACTS_BUCKET', $Bucket, 'Machine')
[Environment]::SetEnvironmentVariable('AWS_DEFAULT_REGION', $Region, 'Machine')
$env:ARTIFACTS_BUCKET = $Bucket
$env:AWS_DEFAULT_REGION = $Region

# --- Own private IP (IMDSv2) ---
$Token = Invoke-RestMethod -Method PUT -Uri http://169.254.169.254/latest/api/token -Headers @{{'X-aws-ec2-metadata-token-ttl-seconds'='21600'}}
$PrivateIp = Invoke-RestMethod -Uri http://169.254.169.254/latest/meta-data/local-ipv4 -Headers @{{'X-aws-ec2-metadata-token'=$Token}}

# --- Java 17 (Temurin) ---
# NOTE: Adoptium's /v3/binary/... API returns a .zip archive (PK signature),
# not an MSI, despite historically being fetched as one - extract it directly
# instead of msiexec'ing it (which fails with MSI error 2203/exit 1620,
# "cannot open database file", since the downloaded file isn't a valid MSI).
$JavaHome = 'C:\\java\\jdk-17'
if (-not (Test-Path "$JavaHome\\bin\\java.exe")) {{
    Invoke-WebRequest -Uri 'https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse' -OutFile C:\\temurin17.zip -UseBasicParsing
    Expand-Archive -Path C:\\temurin17.zip -DestinationPath C:\\temurin17_extract -Force
    $extractedDir = Get-ChildItem C:\\temurin17_extract -Directory | Select-Object -First 1
    if (Test-Path $JavaHome) {{ Remove-Item $JavaHome -Recurse -Force -ErrorAction SilentlyContinue }}
    New-Item -ItemType Directory -Force -Path (Split-Path $JavaHome -Parent) | Out-Null
    Move-Item $extractedDir.FullName $JavaHome -Force
}}
[Environment]::SetEnvironmentVariable('JAVA_HOME', $JavaHome, 'Machine')

# --- Python 3.12 ---
$PythonExe = 'C:\\Python312\\python.exe'
if (-not (Test-Path $PythonExe)) {{
    Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe' -OutFile C:\\python-installer.exe -UseBasicParsing
    Start-Process C:\\python-installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1 TargetDir=C:\\Python312' -Wait
}}
[Environment]::SetEnvironmentVariable('PYSPARK_PYTHON', $PythonExe, 'Machine')
[Environment]::SetEnvironmentVariable('PYSPARK_DRIVER_PYTHON', $PythonExe, 'Machine')

# --- AWS CLI v2 (NOT preinstalled on the Windows Server base AMI - needed
#     below for project.zip, and by the GPU worker for the NVIDIA driver) ---
$AwsCliExe = 'C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe'
if (-not (Test-Path $AwsCliExe)) {{
    Invoke-WebRequest -Uri 'https://awscli.amazonaws.com/AWSCLIV2.msi' -OutFile C:\\awscliv2.msi -UseBasicParsing
    Start-Process msiexec.exe -ArgumentList '/i C:\\awscliv2.msi /quiet /norestart' -Wait
}}
$env:Path = "C:\\Program Files\\Amazon\\AWSCLIV2;$env:Path"

# --- Spark {SPARK_VERSION} ---
$SparkHome = 'C:\\spark\\spark-{SPARK_VERSION}-bin-hadoop3'
if (-not (Test-Path "$SparkHome\\bin\\spark-class2.cmd")) {{
    if (Test-Path C:\\spark\\spark.tgz) {{ Remove-Item C:\\spark\\spark.tgz -Force -ErrorAction SilentlyContinue }}
    Invoke-WebRequest -Uri 'https://archive.apache.org/dist/spark/spark-{SPARK_VERSION}/spark-{SPARK_VERSION}-bin-hadoop3.tgz' -OutFile C:\\spark\\spark.tgz -UseBasicParsing
    tar -xzf C:\\spark\\spark.tgz -C C:\\spark
}}
[Environment]::SetEnvironmentVariable('SPARK_HOME', $SparkHome, 'Machine')
$env:SPARK_HOME = $SparkHome
$env:JAVA_HOME = $JavaHome
$env:Path = "$SparkHome\\bin;$JavaHome\\bin;C:\\Python312;C:\\Python312\\Scripts;$env:Path"

# --- Project code from S3 (re-run manually via SSM after updating project.zip to refresh) ---
aws s3 cp "s3://$Bucket/inference/project.zip" C:\\project.zip
if (Test-Path 'C:\\project.zip') {{
    Expand-Archive -Path C:\\project.zip -DestinationPath C:\\app -Force
    & $PythonExe -m pip install --quiet pyspark=={SPARK_VERSION} boto3
    & $PythonExe -m pip install --quiet -r C:\\app\\requirements.txt
    {torch_line}
}}

# --- CloudWatch Agent (Windows) ---
$CwAgentExe = 'C:\\Program Files\\Amazon\\AmazonCloudWatchAgent\\amazon-cloudwatch-agent.exe'
if (-not (Test-Path $CwAgentExe)) {{
    Invoke-WebRequest -Uri 'https://s3.amazonaws.com/amazoncloudwatch-agent/windows/amd64/latest/amazon-cloudwatch-agent.msi' -OutFile C:\\cwagent.msi -UseBasicParsing
    Start-Process msiexec.exe -ArgumentList '/i C:\\cwagent.msi /quiet /norestart' -Wait
    {_cwagent_config_windows()}
    & 'C:\\Program Files\\Amazon\\AmazonCloudWatchAgent\\amazon-cloudwatch-agent-ctl.ps1' -a fetch-config -m ec2 -s -c file:C:\\cwagent-config.json
}}

# --- Auto-shutdown safety net ---
if ("{auto_shutdown_hours}" -ne "0") {{
    $ShutdownTime = (Get-Date).AddHours({auto_shutdown_hours})
    schtasks /create /tn AutoShutdown /sc once /sd $ShutdownTime.ToString('MM/dd/yyyy') /st $ShutdownTime.ToString('HH:mm') /tr "shutdown /s /t 0" /ru SYSTEM /f 2>$null | Out-Null
}}
""".strip("\n")


def _master_tail_script() -> str:
    return r"""
# --- Start Spark master + a local CPU worker (idempotent) ---
$MasterRunning = Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'deploy\.master\.Master' }
if (-not $MasterRunning) {
    Start-Process -FilePath "$SparkHome\bin\spark-class2.cmd" `
        -ArgumentList 'org.apache.spark.deploy.master.Master','--host',$PrivateIp,'--port','7077' `
        -WindowStyle Hidden `
        -RedirectStandardOutput C:\spark\logs\master.log -RedirectStandardError C:\spark\logs\master.err.log
    Start-Sleep -Seconds 10
}

$WorkerRunning = Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'deploy\.worker\.Worker' }
if (-not $WorkerRunning) {
    Start-Process -FilePath "$SparkHome\bin\spark-class2.cmd" `
        -ArgumentList 'org.apache.spark.deploy.worker.Worker',"spark://${PrivateIp}:7077",'-c','2','-m','4g' `
        -WindowStyle Hidden `
        -RedirectStandardOutput C:\spark\logs\worker.log -RedirectStandardError C:\spark\logs\worker.err.log
}
""".strip("\n")


def _gpu_worker_tail_script(master_private_ip: str) -> str:
    return f"""
# --- NVIDIA driver (Tesla/datacenter, from AWS's public unauthenticated bucket) ---
$MasterIp = "{master_private_ip}"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {{
    $listing = aws s3 ls s3://ec2-windows-nvidia-drivers/latest/ --no-sign-request
    $installerName = ($listing -split "`r?`n" | Where-Object {{ $_ -match 'server2022|server2019' }} |
        Select-Object -First 1) -replace '^.*\\s(\\S+\\.exe)\\s*$', '$1'
    if ($installerName) {{
        aws s3 cp "s3://ec2-windows-nvidia-drivers/latest/$installerName" C:\\nvidia-driver.exe --no-sign-request
        Start-Process -FilePath C:\\nvidia-driver.exe -ArgumentList '-s -clean -noreboot' -Wait
    }}
    # Reboot to load the driver; the AtStartup SparkBootstrap task re-runs this
    # same script afterwards, finds nvidia-smi present, and continues below.
    Restart-Computer -Force
    exit
}}

# --- Start Spark GPU worker (idempotent) ---
$WorkerRunning = Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue |
    Where-Object {{ $_.CommandLine -match 'deploy\\.worker\\.Worker' }}
if (-not $WorkerRunning) {{
    Start-Process -FilePath "$SparkHome\\bin\\spark-class2.cmd" `
        -ArgumentList 'org.apache.spark.deploy.worker.Worker',"spark://${{MasterIp}}:7077",'-c','4','-m','12g' `
        -WindowStyle Hidden `
        -RedirectStandardOutput C:\\spark\\logs\\worker.log -RedirectStandardError C:\\spark\\logs\\worker.err.log
}}
""".strip("\n")


def _cwagent_config_windows() -> str:
    """Written inline (Set-Content) into the common script, not returned as
    JSON directly — metric names are renamed to match the Linux stack's
    CWAgent metric names (cpu_usage_active/mem_used_percent/disk_used_percent)
    so the same dashboard widget queries work against either cluster.

    Uses a plain multi-line single-quoted string literal (not a here-string
    @'...'@) because this text is itself embedded inside the outer @'...'@
    here-string that _script_writer_commands() uses to write the whole
    bootstrap script to disk — here-strings don't nest, so a second @'...'@
    in here would terminate the outer one early at runtime. A single-quoted
    string can safely span multiple lines and contains no single quotes of
    its own (the JSON only uses double quotes), so no escaping is needed.
    """
    return r"""
    '{
  "agent": {"metrics_collection_interval": 30},
  "metrics": {
    "namespace": "CWAgent",
    "append_dimensions": {"InstanceId": "${aws:InstanceId}"},
    "metrics_collected": {
      "Processor": {
        "measurement": [{"name": "% Processor Time", "rename": "cpu_usage_active", "unit": "Percent"}],
        "resources": ["_Total"]
      },
      "Memory": {
        "measurement": [{"name": "% Committed Bytes In Use", "rename": "mem_used_percent", "unit": "Percent"}]
      },
      "LogicalDisk": {
        "measurement": [{"name": "% Free Space", "rename": "disk_used_percent", "unit": "Percent"}],
        "resources": ["C:"]
      }
    }
  }
}' | Set-Content -Path C:\cwagent-config.json -Encoding UTF8
""".strip("\n")
