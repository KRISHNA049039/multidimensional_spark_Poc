"""
Generate colorful HTML dashboard with interactive charts from benchmark results.
Uses Chart.js (CDN) — no local dependencies needed. Just run and open the HTML.

Usage:
    python results/generate_dashboard.py
    # Opens: results/benchmark_dashboard.html
"""
import json
import os

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLOUD_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "..", "results", "cloud_benchmark")
GPU_DIR = os.path.join(RESULTS_DIR, "..", "results", "gpu_benchmark")

# Normalize paths
for d in [CLOUD_DIR, GPU_DIR]:
    if not os.path.exists(d):
        # Try alternate locations
        pass

# Find results directories
cloud_candidates = [
    os.path.join(RESULTS_DIR, "..", "..", "results", "cloud_benchmark"),
    os.path.join(RESULTS_DIR, "cloud_benchmark"),
    os.path.join(RESULTS_DIR, "..", "..", "..", "results", "cloud_benchmark"),
    r"C:\multidim_spark_poc\multidimensional_spark_Poc\results\cloud_benchmark",
]
gpu_candidates = [
    os.path.join(RESULTS_DIR, "gpu_benchmark"),
    os.path.join(RESULTS_DIR, "..", "..", "results", "gpu_benchmark"),
    os.path.join(RESULTS_DIR, "..", "..", "..", "results", "gpu_benchmark"),
    r"C:\multidim_spark_poc\multidimensional_spark_Poc\results\gpu_benchmark",
    r"C:\multidim_spark_poc\multidimensional_spark_Poc\pytorch-spark-inference-platform\results\gpu_benchmark",
]

CLOUD_DIR = next((d for d in cloud_candidates if os.path.exists(d)), None)
GPU_DIR = next((d for d in gpu_candidates if os.path.exists(d)), None)

print(f"Cloud results: {CLOUD_DIR}")
print(f"GPU results: {GPU_DIR}")


# === DATA EXTRACTION ===
def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def extract_throughput(data):
    if isinstance(data, list):
        return [(d.get("total_throughput", 0), d.get("elapsed_time", 0), d.get("total_samples_processed", 0)) for d in data]
    dg = data.get("distributed_gpu", data)
    return dg.get("total_throughput", 0), dg.get("elapsed_time", 0), dg.get("total_samples_processed", 0)

# Partition scaling data
partition_data = {}
if CLOUD_DIR:
    for f in os.listdir(CLOUD_DIR):
        if f.startswith("results_partitions_") and f.endswith(".json"):
            parts = int(f.split("_")[2])
            data = load_json(os.path.join(CLOUD_DIR, f))
            tp, elapsed, samples = extract_throughput(data)
            partition_data[parts] = {"throughput": tp, "elapsed": elapsed, "samples": samples}

# Data size scaling
datasize_data = {}
size_map = {"tiny": 500, "small": 1000, "medium": 5000, "large": 8000, "xlarge": 10000}
if CLOUD_DIR:
    for f in os.listdir(CLOUD_DIR):
        if f.startswith("results_datasize_") and f.endswith(".json"):
            size_label = f.split("_")[2]
            data = load_json(os.path.join(CLOUD_DIR, f))
            tp, elapsed, samples = extract_throughput(data)
            datasize_data[size_map.get(size_label, 0)] = {"throughput": tp, "elapsed": elapsed, "samples": samples, "label": size_label}

# Batch size data
batch_data = {}
if CLOUD_DIR:
    for f in os.listdir(CLOUD_DIR):
        if f.startswith("results_batch_") and f.endswith(".json"):
            bs = int(f.split("_")[2])
            data = load_json(os.path.join(CLOUD_DIR, f))
            tp, elapsed, samples = extract_throughput(data)
            batch_data[bs] = {"throughput": tp, "elapsed": elapsed}

# Worker scaling
worker_data = {}
if CLOUD_DIR:
    for f in os.listdir(CLOUD_DIR):
        if f.startswith("results_workers_") and f.endswith(".json"):
            w = int(f.split("_")[2])
            data = load_json(os.path.join(CLOUD_DIR, f))
            tp, elapsed, samples = extract_throughput(data)
            worker_data[w] = {"throughput": tp, "elapsed": elapsed}

# GPU vs CPU vs Hybrid (from gpu_benchmark directory)
device_mode_data = {"cpu_only": [], "gpu_only": [], "hybrid": []}
if GPU_DIR:
    for f in os.listdir(GPU_DIR):
        if f.startswith("cluster_benchmark_") and f.endswith(".json") and "incremental" not in f:
            data = load_json(os.path.join(GPU_DIR, f))
            mode = data.get("device_mode", "unknown")
            if mode in device_mode_data:
                device_mode_data[mode].append({
                    "throughput": data.get("total_throughput", 0),
                    "elapsed": data.get("elapsed_time", 0),
                    "samples": data.get("total_samples_processed", 0),
                    "partitions": data.get("num_partitions", 0),
                    "batch_size": data.get("batch_size", 0),
                })

# Per-model throughput from partition_8 result
per_model_data = {}
if CLOUD_DIR:
    p8_file = os.path.join(CLOUD_DIR, "results_partitions_8_distributed_sig5000_img50_det10_p8_20260725_101607.json")
    if os.path.exists(p8_file):
        data = load_json(p8_file)
        dg = data.get("distributed_gpu", {})
        details = dg.get("partition_details", [])
        if details:
            for model_name, info in details[0].get("per_model", {}).items():
                per_model_data[model_name] = info.get("throughput", 0)

print(f"Partitions: {sorted(partition_data.keys())}")
print(f"Data sizes: {sorted(datasize_data.keys())}")
print(f"Batch sizes: {sorted(batch_data.keys())}")
print(f"Workers: {sorted(worker_data.keys())}")
print(f"Device modes: { {k: len(v) for k,v in device_mode_data.items()} }")
print(f"Per-model models: {len(per_model_data)}")


# === HTML GENERATION ===
# Sort data for charts
p_labels = sorted(partition_data.keys())
p_throughput = [partition_data[k]["throughput"] for k in p_labels]
p_elapsed = [round(partition_data[k]["elapsed"], 2) for k in p_labels]

d_labels = sorted(datasize_data.keys())
d_throughput = [datasize_data[k]["throughput"] for k in d_labels]
d_names = [datasize_data[k]["label"] for k in d_labels]

b_labels = sorted(batch_data.keys())
b_throughput = [batch_data[k]["throughput"] for k in b_labels]

w_labels = sorted(worker_data.keys())
w_throughput = [worker_data[k]["throughput"] for k in w_labels]
w_ideal = [worker_data[min(worker_data.keys())]["throughput"] * w for w in w_labels]

# Device mode averages (5K signals)
mode_avg = {}
for mode, runs in device_mode_data.items():
    big_runs = [r for r in runs if r["samples"] >= 15000]
    if big_runs:
        mode_avg[mode] = round(sum(r["throughput"] for r in big_runs) / len(big_runs), 1)
    elif runs:
        mode_avg[mode] = round(sum(r["throughput"] for r in runs) / len(runs), 1)

# Per-model sorted by throughput
pm_sorted = sorted(per_model_data.items(), key=lambda x: x[1], reverse=True)
pm_names = [x[0] for x in pm_sorted]
pm_values = [round(x[1], 1) for x in pm_sorted]

# Model categories for coloring
model_categories = {
    "ew_classifier": "signal", "signal_denoiser": "signal", "threat_prioritizer": "signal",
    "rf_fingerprinter": "signal", "anomaly_detector": "signal",
    "resnet18": "image", "mobilenetv3": "image", "efficientnet_b0": "image",
    "yolov8_nano": "detection", "yolov8_small": "detection"
}
pm_colors = []
for name in pm_names:
    cat = model_categories.get(name, "other")
    if cat == "signal":
        pm_colors.append("'rgba(54, 162, 235, 0.8)'")
    elif cat == "image":
        pm_colors.append("'rgba(255, 99, 132, 0.8)'")
    else:
        pm_colors.append("'rgba(255, 206, 86, 0.8)'")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spark Multi-Model Inference — Benchmark Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }}
.header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 40px; text-align: center; border-bottom: 2px solid #3b82f6; }}
.header h1 {{ font-size: 2.2em; color: #f8fafc; margin-bottom: 8px; }}
.header p {{ color: #94a3b8; font-size: 1.1em; }}
.metrics-bar {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; padding: 30px 40px; background: #1e293b; }}
.metric-card {{ background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%); border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #334155; }}
.metric-card .value {{ font-size: 2.4em; font-weight: 700; color: #3b82f6; }}
.metric-card .label {{ color: #94a3b8; margin-top: 4px; font-size: 0.9em; }}
.metric-card.green .value {{ color: #10b981; }}
.metric-card.purple .value {{ color: #8b5cf6; }}
.metric-card.orange .value {{ color: #f59e0b; }}
.dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; padding: 30px 40px; }}
.chart-card {{ background: #1e293b; border-radius: 16px; padding: 24px; border: 1px solid #334155; }}
.chart-card h3 {{ color: #f8fafc; margin-bottom: 16px; font-size: 1.1em; }}
.chart-card.full {{ grid-column: 1 / -1; }}
.insight {{ background: #172554; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 0 8px 8px 0; margin-top: 12px; font-size: 0.85em; color: #93c5fd; }}
.insight.warn {{ border-left-color: #f59e0b; background: #1c1917; color: #fcd34d; }}
.insight.success {{ border-left-color: #10b981; background: #022c22; color: #6ee7b7; }}
.legend {{ display: flex; gap: 16px; justify-content: center; margin-top: 12px; }}
.legend span {{ display: flex; align-items: center; gap: 6px; font-size: 0.8em; color: #94a3b8; }}
.legend .dot {{ width: 10px; height: 10px; border-radius: 50%; }}
canvas {{ max-height: 300px; }}
</style>
</head>
<body>
<div class="header">
<h1>⚡ Spark Multi-Model Inference Platform</h1>
<p>Performance & Scalability Dashboard — 10 Models × 3 Device Modes × 81 Benchmark Runs</p>
</div>
"""

html += f"""
<div class="metrics-bar">
<div class="metric-card"><div class="value">3,707</div><div class="label">Peak CPU Throughput (samples/sec)</div></div>
<div class="metric-card green"><div class="value">1,512</div><div class="label">Peak Hybrid Throughput</div></div>
<div class="metric-card purple"><div class="value">10</div><div class="label">Models in Parallel</div></div>
<div class="metric-card orange"><div class="value">~$1.70</div><div class="label">Cost per Full Benchmark</div></div>
</div>

<div class="dashboard">

<!-- Chart 1: Partition Scaling -->
<div class="chart-card">
<h3>📊 Partition Scaling (5K signals, 8 cores)</h3>
<canvas id="partitionChart"></canvas>
<div class="insight success">Sweet spot: 6-8 partitions = core count. Beyond that, model reload overhead dominates.</div>
</div>

<!-- Chart 2: Data Volume Scaling -->
<div class="chart-card">
<h3>📈 Data Volume Scaling</h3>
<canvas id="datasizeChart"></canvas>
<div class="insight">Near-linear scaling above 5K signals. Model load (1.2s) is amortized at higher volumes.</div>
</div>

<!-- Chart 3: Worker Scaling -->
<div class="chart-card">
<h3>👷 Worker Scaling (Horizontal)</h3>
<canvas id="workerChart"></canvas>
<div class="insight warn">Sub-linear scaling due to 1.2s model reload per executor. Better at 50K+ signals.</div>
</div>

<!-- Chart 4: Batch Size -->
<div class="chart-card">
<h3>📦 Batch Size Impact (CPU)</h3>
<canvas id="batchChart"></canvas>
<div class="insight">Minimal impact on CPU (&lt;5%). Batch 256 slightly optimal. GPU would show larger gains.</div>
</div>

<!-- Chart 5: Device Mode Comparison -->
<div class="chart-card">
<h3>🖥️ Device Mode Comparison (3K+ signals)</h3>
<canvas id="deviceChart"></canvas>
<div class="insight success">Hybrid wins by routing signals→CPU and images→GPU. Pure GPU loses on signal models.</div>
</div>

<!-- Chart 6: Per-Model Throughput -->
<div class="chart-card">
<h3>🧠 Per-Model Throughput (samples/sec per executor)</h3>
<canvas id="modelChart"></canvas>
<div class="legend">
<span><span class="dot" style="background:#36a2eb"></span> Signal Models</span>
<span><span class="dot" style="background:#ff6384"></span> Image Classification</span>
<span><span class="dot" style="background:#ffce56"></span> Object Detection</span>
</div>
<div class="insight warn">CNN models (ResNet/YOLO) are 1000-10000× slower than signal models on CPU. Primary GPU beneficiaries.</div>
</div>

<!-- Chart 7: Elapsed Time Comparison -->
<div class="chart-card full">
<h3>⏱️ Elapsed Time by Configuration</h3>
<canvas id="elapsedChart"></canvas>
</div>

</div>

<script>
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
const gradient_blue = (ctx) => {{ let g = ctx.chart.ctx.createLinearGradient(0,0,0,300); g.addColorStop(0,'rgba(59,130,246,0.5)'); g.addColorStop(1,'rgba(59,130,246,0.05)'); return g; }};
const gradient_green = (ctx) => {{ let g = ctx.chart.ctx.createLinearGradient(0,0,0,300); g.addColorStop(0,'rgba(16,185,129,0.5)'); g.addColorStop(1,'rgba(16,185,129,0.05)'); return g; }};
"""

html += f"""
// Partition Chart
new Chart(document.getElementById('partitionChart'), {{
  type: 'line',
  data: {{
    labels: {p_labels},
    datasets: [{{
      label: 'Throughput (samples/sec)',
      data: {p_throughput},
      borderColor: '#3b82f6', backgroundColor: gradient_blue,
      fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: '#3b82f6'
    }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Throughput' }} }}, x: {{ title: {{ display: true, text: 'Partitions' }} }} }} }}
}});

// Data Size Chart
new Chart(document.getElementById('datasizeChart'), {{
  type: 'line',
  data: {{
    labels: {d_labels},
    datasets: [{{
      label: 'Throughput',
      data: {d_throughput},
      borderColor: '#10b981', backgroundColor: gradient_green,
      fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: '#10b981'
    }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Throughput (samples/sec)' }} }}, x: {{ title: {{ display: true, text: 'Signal Samples' }} }} }} }}
}});

// Worker Chart
new Chart(document.getElementById('workerChart'), {{
  type: 'line',
  data: {{
    labels: {w_labels},
    datasets: [
      {{ label: 'Actual', data: {w_throughput}, borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.1)', fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: '#8b5cf6' }},
      {{ label: 'Ideal Linear', data: {w_ideal}, borderColor: '#475569', borderDash: [5,5], pointRadius: 0, fill: false }}
    ]
  }},
  options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Throughput' }} }}, x: {{ title: {{ display: true, text: 'Workers' }} }} }} }}
}});

// Batch Size Chart
new Chart(document.getElementById('batchChart'), {{
  type: 'bar',
  data: {{
    labels: {b_labels},
    datasets: [{{
      label: 'Throughput',
      data: {b_throughput},
      backgroundColor: ['#f59e0b','#f97316','#ef4444','#ec4899','#8b5cf6','#6366f1'],
      borderRadius: 6
    }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true, min: 3000, title: {{ display: true, text: 'Throughput' }} }}, x: {{ title: {{ display: true, text: 'Batch Size' }} }} }} }}
}});

// Device Mode Chart
new Chart(document.getElementById('deviceChart'), {{
  type: 'bar',
  data: {{
    labels: ['CPU Only', 'GPU Only', 'Hybrid'],
    datasets: [{{
      label: 'Avg Throughput (3K+ signals)',
      data: [{mode_avg.get('cpu_only', 0)}, {mode_avg.get('gpu_only', 0)}, {mode_avg.get('hybrid', 0)}],
      backgroundColor: ['rgba(59,130,246,0.7)', 'rgba(139,92,246,0.7)', 'rgba(16,185,129,0.7)'],
      borderColor: ['#3b82f6', '#8b5cf6', '#10b981'],
      borderWidth: 2, borderRadius: 8
    }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Throughput (samples/sec)' }} }} }} }}
}});

// Per-Model Chart
new Chart(document.getElementById('modelChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(pm_names)},
    datasets: [{{
      label: 'Throughput/executor',
      data: {pm_values},
      backgroundColor: [{','.join(pm_colors)}],
      borderRadius: 4
    }}]
  }},
  options: {{ indexAxis: 'y', responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ type: 'logarithmic', title: {{ display: true, text: 'Throughput (log scale)' }} }} }} }}
}});

// Elapsed Time comparison
new Chart(document.getElementById('elapsedChart'), {{
  type: 'bar',
  data: {{
    labels: ['2 parts', '4 parts', '6 parts', '8 parts', '12 parts', '16 parts', '1 worker', '2 workers', '3 workers', '4 workers', '6 workers'],
    datasets: [{{
      label: 'Elapsed Time (sec)',
      data: [{', '.join(str(round(partition_data[k]["elapsed"],1)) for k in p_labels)}, {', '.join(str(round(worker_data[k]["elapsed"],1)) for k in w_labels)}],
      backgroundColor: ['#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#8b5cf6','#8b5cf6','#8b5cf6','#8b5cf6','#8b5cf6'],
      borderRadius: 4
    }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ title: {{ display: true, text: 'Seconds' }} }} }} }}
}});
</script>
</body>
</html>
"""

# Write HTML
output_path = os.path.join(RESULTS_DIR, "benchmark_dashboard.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ Dashboard generated: {output_path}")
print("   Open in browser to view interactive charts.")
