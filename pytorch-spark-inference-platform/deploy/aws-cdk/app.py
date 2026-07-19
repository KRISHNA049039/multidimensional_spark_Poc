#!/usr/bin/env python3
"""
CDK App entrypoint — Spark inference cluster on EC2 (1 master + N GPU workers).

Usage:
    cdk deploy --parameters KeyName=my-key --parameters SshCidr=1.2.3.4/32

See docs/AWS_CDK_DEPLOYMENT.md (in the platform root) for the full walkthrough.
"""
import aws_cdk as cdk

from spark_cluster.spark_cluster_stack import SparkClusterStack

app = cdk.App()

SparkClusterStack(
    app,
    "SparkInferenceClusterStack",
    description="1 master + N GPU worker EC2 cluster for pytorch-spark-inference-platform, "
                 "with CloudWatch metrics at host/Spark/GPU/benchmark level.",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
)

app.synth()
