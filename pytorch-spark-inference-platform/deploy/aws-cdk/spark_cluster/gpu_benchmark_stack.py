"""
GPU Benchmark Stack — Single g4dn.xlarge instance with:
  - AWS Deep Learning AMI (NVIDIA drivers pre-installed)
  - Docker + nvidia-container-toolkit ready
  - SSM access for manual docker commands
  - S3 bucket for code upload / results download
  - Auto-shutdown after 4 hours (safety net)

Usage:
    cd deploy/aws-cdk
    pip install -r requirements.txt
    cdk deploy GpuBenchmarkStack --context region=us-east-1

Then SSM into the instance and run benchmarks manually.
"""
from aws_cdk import (
    Stack, CfnOutput, RemovalPolicy, Tags, Duration,
    aws_ec2 as ec2, aws_iam as iam, aws_s3 as s3,
)
from constructs import Construct


class GpuBenchmarkStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # VPC
        vpc = ec2.Vpc(self, "BenchVpc", max_azs=2, nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public",
                    subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24)
            ])

        # Security Group — only outbound (SSM access, no SSH needed)
        sg = ec2.SecurityGroup(self, "BenchSg", vpc=vpc,
            description="GPU benchmark - outbound only",
            allow_all_outbound=True)

        # S3 Bucket
        bucket = s3.Bucket(self, "ArtifactsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL)

        # IAM Role
        role = iam.Role(self, "BenchRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
            ])
        bucket.grant_read_write(role)
        role.add_to_policy(iam.PolicyStatement(
            actions=["ec2:DescribeInstances"], resources=["*"]))

        # Use AWS Deep Learning AMI — has NVIDIA drivers + Docker pre-installed
        # This guarantees nvidia-smi works out of the box
        dl_ami = ec2.MachineImage.lookup(
            name="Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *",
            owners=["amazon"],
        )

        # UserData — install docker, nvidia-container-toolkit, pull code
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "set -eux",
            # Docker (if not already installed)
            "apt-get update -y",
            "apt-get install -y docker.io awscli unzip jq",
            "systemctl enable docker && systemctl start docker",
            "usermod -aG docker ubuntu || true",
            # nvidia-container-toolkit
            "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
            "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
            "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
            "tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
            "apt-get update -y && apt-get install -y nvidia-container-toolkit",
            "nvidia-ctk runtime configure --runtime=docker",
            "systemctl restart docker",
            # Verify GPU
            "nvidia-smi",
            "docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi",
            # Workspace
            "mkdir -p /opt/benchmark/app /opt/benchmark/results",
            f"echo 'BUCKET={bucket.bucket_name}' >> /etc/environment",
            # Auto-shutdown after 4 hours
            "apt-get install -y at && systemctl enable atd && systemctl start atd",
            "echo 'shutdown -h now' | at now + 4 hours",
            "echo '=== GPU Instance Ready — SSM in and run benchmarks ==='",
        )

        # EC2 Instance — g4dn.xlarge (4 vCPU, 16GB, 1x T4 GPU)
        instance = ec2.Instance(self, "GpuBenchInstance",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType("g4dn.xlarge"),
            machine_image=dl_ami,
            security_group=sg,
            role=role,
            user_data=user_data,
            block_devices=[ec2.BlockDevice(
                device_name="/dev/sda1",
                volume=ec2.BlockDeviceVolume.ebs(150,
                    volume_type=ec2.EbsDeviceVolumeType.GP3),
            )],
            associate_public_ip_address=True,
        )
        Tags.of(instance).add("Name", "gpu-benchmark-instance")

        # Outputs
        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(self, "PublicIp", value=instance.instance_public_ip)
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "SSMCommand",
            value=f"aws ssm start-session --target {instance.instance_id} --region {self.region}")
        CfnOutput(self, "UploadCommand",
            value=f"aws s3 cp project.zip s3://{bucket.bucket_name}/project.zip --region {self.region}")
