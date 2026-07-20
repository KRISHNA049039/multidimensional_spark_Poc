"""
Capture Spark UI statistics and save to JSON.

Run this WHILE a benchmark is still executing (port 4040 is only alive during a running app).

Usage (from master container):
    python benchmark/capture_spark_stats.py

Or from host:
    docker exec spark-master python benchmark/capture_spark_stats.py

Saves all data to results/spark_stats_<timestamp>.json
"""

import json
import os
import sys
import urllib.request
from datetime import datetime

BASE_URL = "http://localhost:4040/api/v1"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def fetch_json(url):
    """Fetch JSON from Spark REST API."""
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return None


def capture_all():
    print("\n" + "=" * 70)
    print("  SPARK UI STATS CAPTURE")
    print("=" * 70)

    # Get applications
    apps = fetch_json(f"{BASE_URL}/applications")
    if not apps:
        print("\n  ERROR: No Spark application found on port 4040.")
        print("  Make sure a benchmark is running or just finished.")
        print("  Port 4040 is only available while a Spark app is active.\n")
        sys.exit(1)

    app = apps[0]
    app_id = app["id"]
    app_name = app.get("name", "unknown")
    print(f"\n  App: {app_name} ({app_id})")

    stats = {
        "captured_at": datetime.now().isoformat(),
        "application": app,
    }

    # Jobs
    print("  Fetching jobs...")
    jobs = fetch_json(f"{BASE_URL}/applications/{app_id}/jobs")
    stats["jobs"] = jobs or []
    print(f"    {len(stats['jobs'])} jobs")

    # Stages
    print("  Fetching stages...")
    stages = fetch_json(f"{BASE_URL}/applications/{app_id}/stages")
    stats["stages"] = stages or []
    print(f"    {len(stats['stages'])} stages")

    # Stage details (tasks per stage)
    print("  Fetching stage details...")
    stage_details = []
    for stage in (stages or []):
        stage_id = stage.get("stageId")
        attempt = stage.get("attemptId", 0)
        detail = fetch_json(f"{BASE_URL}/applications/{app_id}/stages/{stage_id}/{attempt}")
        if detail:
            stage_details.append(detail)
    stats["stage_details"] = stage_details

    # Executors
    print("  Fetching executors...")
    executors = fetch_json(f"{BASE_URL}/applications/{app_id}/allexecutors")
    stats["executors"] = executors or []
    print(f"    {len(stats['executors'])} executors")

    # Environment
    print("  Fetching environment...")
    env = fetch_json(f"{BASE_URL}/applications/{app_id}/environment")
    stats["environment"] = env

    # Storage (RDDs)
    print("  Fetching storage/RDD info...")
    storage = fetch_json(f"{BASE_URL}/applications/{app_id}/storage/rdd")
    stats["storage_rdd"] = storage or []

    # Summary
    print(f"\n{'─' * 70}")
    print("  QUICK SUMMARY")
    print(f"{'─' * 70}")

    total_tasks = sum(s.get("numCompleteTasks", 0) + s.get("numActiveTasks", 0) for s in (stages or []))
    total_duration = sum(j.get("stageIds", [0]).__len__() for j in (jobs or []))

    print(f"  Application    : {app_name}")
    print(f"  Jobs           : {len(stats['jobs'])}")
    print(f"  Stages         : {len(stats['stages'])}")
    print(f"  Total Tasks    : {total_tasks}")
    print(f"  Executors      : {len(stats['executors'])}")

    if executors:
        print(f"\n  EXECUTOR DETAIL:")
        print(f"  {'ID':<6} {'Host':<30} {'Cores':<6} {'Tasks':<7} {'Duration':<12} {'Input':<10} {'Shuffle'}")
        print(f"  {'─'*6} {'─'*30} {'─'*6} {'─'*7} {'─'*12} {'─'*10} {'─'*10}")
        for ex in executors:
            host = ex.get("hostPort", "?")
            print(f"  {ex.get('id','?'):<6} {host:<30} "
                  f"{ex.get('totalCores',0):<6} {ex.get('completedTasks',0):<7} "
                  f"{ex.get('totalDuration',0)/1000:.1f}s{'':>5} "
                  f"{ex.get('totalInputBytes',0)/1e6:.1f}MB{'':>3} "
                  f"{ex.get('totalShuffleRead',0)/1e6:.1f}MB")

    if stages:
        print(f"\n  STAGES:")
        print(f"  {'ID':<4} {'Name':<40} {'Tasks':<7} {'Input':<10} {'Duration'}")
        print(f"  {'─'*4} {'─'*40} {'─'*7} {'─'*10} {'─'*10}")
        for s in stages[:20]:  # Limit to 20 stages
            name = s.get("name", "?")[:38]
            print(f"  {s.get('stageId',0):<4} {name:<40} "
                  f"{s.get('numCompleteTasks',0):<7} "
                  f"{s.get('inputBytes',0)/1e6:.1f}MB{'':>3} "
                  f"{s.get('executorRunTime',0)/1000:.1f}s")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_DIR, f"spark_stats_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n  Results saved: {filepath}")
    print("")

    return stats


if __name__ == "__main__":
    capture_all()
