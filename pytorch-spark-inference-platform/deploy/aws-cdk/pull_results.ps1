# =============================================================================
# pull_results.ps1 — Sync BYOM/benchmark results from S3 to a local directory
# =============================================================================
# Usage:
#   .\deploy\aws-cdk\pull_results.ps1 -BucketName <ArtifactsBucketName output>
#   .\deploy\aws-cdk\pull_results.ps1 -BucketName my-bucket -LocalDir .\results\aws
#
# Results land in S3 under s3://<bucket>/results/ whenever submit_job.py runs
# on a cluster node with the ARTIFACTS_BUCKET env var set (set automatically
# by the CDK stacks' bootstrap scripts).
# =============================================================================

param(
    [Parameter(Mandatory = $true)]
    [string]$BucketName,
    [string]$LocalDir = ".\results"
)

if (-not (Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
}

aws s3 sync "s3://$BucketName/results/" $LocalDir
