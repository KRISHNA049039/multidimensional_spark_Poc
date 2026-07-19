"""
Shared CloudWatch publishing helper used by all three metrics publishers
(spark_metrics_publisher, gpu_metrics_publisher, benchmark_metrics_publisher).

Design:
- boto3 picks up credentials automatically from the EC2 instance role attached by
  the CDK stack (see deploy/aws-cdk/spark_cluster/spark_cluster_stack.py) — no
  access keys are ever stored in the image or containers.
- All metrics are dimensioned by NodeRole (master/worker) and InstanceId so the
  CloudWatch dashboard can break results down per-node.
- If boto3 or AWS credentials are unavailable (e.g. running locally in Docker
  Desktop without an instance role), publishing is skipped with a warning
  instead of crashing the benchmark/inference process.
"""
import logging
import os
import socket
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger("cloudwatch_publisher")

_METADATA_TOKEN_URL = "http://169.254.169.254/latest/api/token"
_METADATA_BASE_URL = "http://169.254.169.254/latest/meta-data"


def _fetch_imdsv2_token(timeout: float = 1.0) -> Optional[str]:
    try:
        req = urllib.request.Request(
            _METADATA_TOKEN_URL, method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode().strip()
    except Exception:
        return None


def _fetch_metadata(path: str, timeout: float = 1.0) -> Optional[str]:
    """Best-effort EC2 instance metadata fetch (IMDSv2, falls back to no-op off-EC2)."""
    token = _fetch_imdsv2_token(timeout=timeout)
    headers = {"X-aws-ec2-metadata-token": token} if token else {}
    try:
        req = urllib.request.Request(f"{_METADATA_BASE_URL}/{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode().strip()
    except Exception:
        return None


def get_instance_id() -> str:
    return _fetch_metadata("instance-id") or os.environ.get("HOSTNAME", socket.gethostname())


def get_region() -> str:
    az = _fetch_metadata("placement/availability-zone")
    if az:
        return az[:-1]
    return os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"


class CloudWatchPublisher:
    """Thin wrapper around boto3 CloudWatch put_metric_data with graceful degradation."""

    def __init__(self, namespace: str, node_role: str, extra_dimensions: Optional[Dict[str, str]] = None):
        self.namespace = namespace
        self.node_role = node_role
        self.instance_id = get_instance_id()
        self.extra_dimensions = extra_dimensions or {}
        self._client = None
        self._enabled = True

        try:
            import boto3
            self._client = boto3.client("cloudwatch", region_name=get_region())
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("CloudWatch publishing disabled (boto3/credentials unavailable): %s", exc)
            self._enabled = False

    def _dimensions(self, extra: Optional[Dict[str, str]] = None) -> List[dict]:
        dims = {
            "NodeRole": self.node_role,
            "InstanceId": self.instance_id,
        }
        dims.update(self.extra_dimensions)
        if extra:
            dims.update(extra)
        return [{"Name": k, "Value": str(v)} for k, v in dims.items()]

    def put_metric(self, metric_name: str, value: float, unit: str = "None",
                   extra_dimensions: Optional[Dict[str, str]] = None) -> None:
        """Publish a single metric datapoint. No-op (logged) if CloudWatch is unavailable."""
        if not self._enabled:
            logger.debug("[metrics-disabled] %s=%s %s", metric_name, value, unit)
            return
        try:
            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{
                    "MetricName": metric_name,
                    "Dimensions": self._dimensions(extra_dimensions),
                    "Value": float(value),
                    "Unit": unit,
                }],
            )
        except Exception as exc:  # pragma: no cover - network/AWS dependent
            logger.warning("Failed to publish metric %s: %s", metric_name, exc)

    def put_metrics(self, metrics: List[dict]) -> None:
        """
        Publish a batch of metrics in one API call (CloudWatch allows up to 1000
        per call; callers should chunk larger batches).

        Each item: {"metric_name": str, "value": float, "unit": str (optional),
                    "extra_dimensions": dict (optional)}
        """
        if not self._enabled:
            for m in metrics:
                logger.debug("[metrics-disabled] %s=%s", m["metric_name"], m["value"])
            return
        if not metrics:
            return
        try:
            data = [{
                "MetricName": m["metric_name"],
                "Dimensions": self._dimensions(m.get("extra_dimensions")),
                "Value": float(m["value"]),
                "Unit": m.get("unit", "None"),
            } for m in metrics]
            # CloudWatch caps at 1000 metrics per PutMetricData call
            for i in range(0, len(data), 1000):
                self._client.put_metric_data(Namespace=self.namespace, MetricData=data[i:i + 1000])
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to publish metric batch: %s", exc)
