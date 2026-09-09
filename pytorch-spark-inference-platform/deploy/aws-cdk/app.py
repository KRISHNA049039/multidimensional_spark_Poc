#!/usr/bin/env python3
"""
CDK App entrypoint — Spark inference cluster on EC2.

Stacks:
  1. SparkInferenceClusterStack — Full 2-node Linux/Docker cluster (master + GPU worker)
  2. GpuBenchmarkStack — Single GPU instance for manual benchmarking
  3. WindowsSparkClusterStack — Native (Docker-free) Windows Server cluster (master + GPU worker)

Usage:
    cdk deploy GpuBenchmarkStack --context region=us-east-1
    cdk deploy SparkInferenceClusterStack --context region=us-east-1
    cdk deploy WindowsSparkClusterStack --context region=us-east-1
"""
import os
import aws_cdk as cdk

from spark_cluster.spark_cluster_stack import SparkClusterStack
from spark_cluster.gpu_benchmark_stack import GpuBenchmarkStack
from spark_cluster.windows_spark_cluster_stack import WindowsSparkClusterStack

app = cdk.App()

region = app.node.try_get_context("region") or "us-east-1"
account = app.node.try_get_context("account") or os.environ.get("CDK_DEFAULT_ACCOUNT")

# If no account specified, use environment-agnostic (won't work for AMI lookups)
# So we require explicit account
env = cdk.Environment(account=account, region=region)

SparkClusterStack(app, "SparkInferenceClusterStack",
    description="Full Spark cluster: master + GPU worker",
    env=env)

GpuBenchmarkStack(app, "GpuBenchmarkStack",
    description="Single g4dn.xlarge with Deep Learning AMI for manual GPU benchmarks",
    env=env)

WindowsSparkClusterStack(app, "WindowsSparkClusterStack",
    description="Native (Docker-free) Windows Server Spark cluster: master + GPU worker",
    env=env)

app.synth()
