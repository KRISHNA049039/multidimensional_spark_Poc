#!/usr/bin/env python3
"""
CDK App entrypoint — Spark inference cluster on EC2.

Stacks:
  1. SparkInferenceClusterStack — Full 2-node cluster (master + GPU worker)
  2. GpuBenchmarkStack — Single GPU instance for manual benchmarking

Usage:
    cdk deploy GpuBenchmarkStack --context region=us-east-1
    cdk deploy SparkInferenceClusterStack --context region=us-east-1
"""
import aws_cdk as cdk

from spark_cluster.spark_cluster_stack import SparkClusterStack
from spark_cluster.gpu_benchmark_stack import GpuBenchmarkStack

app = cdk.App()

region = app.node.try_get_context("region") or "us-east-1"
env = cdk.Environment(
    account=app.node.try_get_context("account"),
    region=region,
)

SparkClusterStack(app, "SparkInferenceClusterStack",
    description="Full Spark cluster: master + GPU worker",
    env=env)

GpuBenchmarkStack(app, "GpuBenchmarkStack",
    description="Single g4dn.xlarge with Deep Learning AMI for manual GPU benchmarks",
    env=env)

app.synth()
