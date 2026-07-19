"""
SparkClusterStack - 2 EC2 instances:
  1. Master (CPU, t3.large) - Spark master + driver + CPU worker
  2. GPU Worker (g4dn.xlarge) - Spark worker with GPU

Both instances pull code from S3, build the Docker image, and start Spark.
Metrics are published to CloudWatch at host/Spark/GPU/benchmark levels.
"""
from aws_cdk import (
    Stack,
    CfnParameter,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Tags,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    aws_cloudwatch as cloudwatch,
)
from constructs import Construct

SPARK_MASTER_RPC_PORT = 7077
SPARK_MASTER_UI_PORT = 8080
SPARK_APP_UI_PORT = 4040


class SparkClusterStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------
        admin_cidr = CfnParameter(
            self, "AdminCidr",
            type="String",
            default="",
            description="CIDR allowed to reach Spark UIs (8080/4040), e.g. '203.0.113.5/32'.",
        )
        master_instance_type = CfnParameter(
            self, "MasterInstanceType", type="String", default="m5.xlarge",
            description="Instance type for Spark master (CPU). Also acts as a CPU worker.",
        )
        worker_instance_type = CfnParameter(
            self, "WorkerInstanceType", type="String", default="g4dn.xlarge",
            description="Instance type for the GPU worker.",
        )
        auto_shutdown_hours = CfnParameter(
            self, "AutoShutdownHours", type="Number", default=4, min_value=0,
            description="Safety net: instances shut down after this many hours (0 = disabled).",
        )

        # ---------------------------------------------------------------
        # VPC - multiple AZs for GPU availability
        # ---------------------------------------------------------------
        vpc = ec2.Vpc(
            self, "SparkVpc",
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
            self, "SparkClusterSg", vpc=vpc,
            description="Spark inference cluster - master and worker traffic",
            allow_all_outbound=True,
        )
        sg.add_ingress_rule(sg, ec2.Port.all_traffic(), "Inter-node Spark traffic")

        # Admin UI access (conditional)
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
            self, "SparkNodeRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="Spark inference nodes - SSM, CloudWatch, S3, EC2 describe",
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
            actions=["ec2:DescribeInstances", "ec2:DescribeVolumes",
                     "ec2:ModifyVolume", "autoscaling:DescribeAutoScalingGroups"],
            resources=["*"],
        ))

        # ---------------------------------------------------------------
        # Master Instance (CPU - also runs as Spark worker)
        # ---------------------------------------------------------------
        master_user_data = ec2.UserData.for_linux()
        master_user_data.add_commands(*_common_bootstrap(
            artifacts_bucket.bucket_name, self.region, auto_shutdown_hours.value_as_string))
        master_user_data.add_commands(*_master_bootstrap())

        master = ec2.Instance(
            self, "SparkMaster",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType(master_instance_type.value_as_string),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            security_group=sg,
            role=role,
            user_data=master_user_data,
            block_devices=[ec2.BlockDevice(
                device_name="/dev/xvda",
                volume=ec2.BlockDeviceVolume.ebs(100, volume_type=ec2.EbsDeviceVolumeType.GP3),
            )],
            associate_public_ip_address=True,
        )
        Tags.of(master).add("Name", "spark-master")
        Tags.of(master).add("Role", "spark-master")

        # ---------------------------------------------------------------
        # GPU Worker Instance
        # ---------------------------------------------------------------
        gpu_worker_user_data = ec2.UserData.for_linux()
        gpu_worker_user_data.add_commands(*_common_bootstrap(
            artifacts_bucket.bucket_name, self.region, auto_shutdown_hours.value_as_string))
        gpu_worker_user_data.add_commands(*_gpu_worker_bootstrap(master.instance_private_ip))

        gpu_worker = ec2.Instance(
            self, "SparkGpuWorker",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType(worker_instance_type.value_as_string),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            security_group=sg,
            role=role,
            user_data=gpu_worker_user_data,
            block_devices=[ec2.BlockDevice(
                device_name="/dev/xvda",
                volume=ec2.BlockDeviceVolume.ebs(150, volume_type=ec2.EbsDeviceVolumeType.GP3),
            )],
            associate_public_ip_address=True,
        )
        Tags.of(gpu_worker).add("Name", "spark-gpu-worker")
        Tags.of(gpu_worker).add("Role", "spark-worker")

        # ---------------------------------------------------------------
        # CloudWatch Dashboard
        # ---------------------------------------------------------------
        dashboard = cloudwatch.Dashboard(self, "SparkClusterDashboard",
                                          dashboard_name="SparkInferenceCluster")
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown="# Spark Inference Cluster - Master / GPU Worker / Benchmark",
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
                        f"?region={self.region}#dashboards:name=SparkInferenceCluster")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _non_empty_condition(scope: Construct, cid: str, param: CfnParameter):
    from aws_cdk import CfnCondition, Fn
    return CfnCondition(scope, cid,
                        expression=Fn.condition_not(Fn.condition_equals(param.value_as_string, "")))


def _if_non_empty(scope: Construct, cid: str, param: CfnParameter):
    from aws_cdk import Fn
    cond = _non_empty_condition(scope, cid, param)
    return Fn.condition_if(cond.logical_id, param.value_as_string, Fn.ref("AWS::NoValue"))


def _common_bootstrap(artifacts_bucket_name: str, region: str, auto_shutdown_hours: str):
    """Commands run on every node."""
    return [
        "set -eux",
        "export PKG_MGR=$(command -v dnf || command -v yum)",
        "$PKG_MGR install -y docker aws-cli jq unzip",
        "systemctl enable docker",
        "systemctl start docker",
        "usermod -aG docker ec2-user || true",
        # CloudWatch Agent
        "rpm -Uvh https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm || true",
        "mkdir -p /opt/aws/amazon-cloudwatch-agent/etc",
        f"cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'CWEOF'\n{_cwagent_config()}\nCWEOF",
        "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl "
        "-a fetch-config -m ec2 -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json",
        # Set bucket env
        f"export ARTIFACTS_BUCKET={artifacts_bucket_name}",
        f"echo 'ARTIFACTS_BUCKET={artifacts_bucket_name}' >> /etc/environment",
        f"export AWS_DEFAULT_REGION={region}",
        # Pull project zip and build Docker image
        "mkdir -p /opt/spark-inference/app",
        f"aws s3 cp s3://{artifacts_bucket_name}/inference/project.zip /opt/spark-inference/project.zip || "
        "echo 'WARNING: project.zip not in S3 yet'",
        "if [ -f /opt/spark-inference/project.zip ]; then "
        "unzip -o /opt/spark-inference/project.zip -d /opt/spark-inference/app && "
        "cd /opt/spark-inference/app && "
        "docker build -t multi-model-inference:latest -f deploy/Dockerfile . ; fi",
        # Also try loading a pre-built image if available
        f"aws s3 cp s3://{artifacts_bucket_name}/inference/multi-model-inference.tar /opt/spark-inference/image.tar && "
        "docker load < /opt/spark-inference/image.tar || true",
        # Safety net
        f"if [ '{auto_shutdown_hours}' != '0' ]; then "
        f"$PKG_MGR install -y at && systemctl enable atd && systemctl start atd && "
        f"echo 'shutdown -h now' | at now + {auto_shutdown_hours} hours; fi",
    ]


def _master_bootstrap():
    """Start Spark master + a local CPU worker on the master node."""
    return [
        "sleep 5",
        # Start Spark master
        "docker run -d --name spark-master --network host --restart unless-stopped "
        "multi-model-inference:latest "
        "bash -c \"start-master.sh && tail -f /opt/spark/logs/*master*\"",
        "sleep 10",
        # Also start a CPU worker on master (so master participates in computation)
        "docker run -d --name spark-cpu-worker --network host --restart unless-stopped "
        "multi-model-inference:latest "
        "bash -c \"start-worker.sh spark://$(hostname -I | awk '{print \\$1}'):7077 -c 2 -m 4g && "
        "tail -f /opt/spark/logs/*worker*\"",
    ]


def _gpu_worker_bootstrap(master_private_ip: str):
    """Install NVIDIA driver + start Spark GPU worker pointing to master."""
    return [
        # NVIDIA driver
        "if ! command -v nvidia-smi &>/dev/null; then "
        "$PKG_MGR install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r) gcc make dkms && "
        "curl -fsSL https://us.download.nvidia.com/tesla/535.183.01/NVIDIA-Linux-x86_64-535.183.01.run "
        "-o /tmp/nvidia-driver.run && "
        "bash /tmp/nvidia-driver.run --silent --dkms || true; fi",
        # nvidia-container-toolkit
        "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | "
        "tee /etc/yum.repos.d/nvidia-container-toolkit.repo",
        "$PKG_MGR install -y nvidia-container-toolkit",
        "nvidia-ctk runtime configure --runtime=docker",
        "systemctl restart docker",
        "sleep 5",
        # Start Spark GPU worker
        f"docker run -d --name spark-gpu-worker --network host --restart unless-stopped "
        f"--gpus all --shm-size=4g "
        f"multi-model-inference:latest "
        f"bash -c \"start-worker.sh spark://{master_private_ip}:7077 -c 4 -m 12g && "
        f"tail -f /opt/spark/logs/*worker*\"",
    ]


def _cwagent_config() -> str:
    import json
    config = {
        "agent": {"metrics_collection_interval": 30},
        "metrics": {
            "namespace": "CWAgent",
            "append_dimensions": {"InstanceId": "${aws:InstanceId}"},
            "metrics_collected": {
                "cpu": {"measurement": ["cpu_usage_active", "cpu_usage_iowait"], "totalcpu": True},
                "mem": {"measurement": ["mem_used_percent"]},
                "disk": {"measurement": ["disk_used_percent"], "resources": ["/"]},
                "netstat": {"measurement": ["tcp_established", "tcp_time_wait"]},
            },
        },
    }
    return json.dumps(config, indent=2)
