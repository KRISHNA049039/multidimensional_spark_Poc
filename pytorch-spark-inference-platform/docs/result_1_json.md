{
  "system_info": {
    "platform": "Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35",
    "python": "3.11.15",
    "torch": "2.7.0+cu128",
    "cpu_cores": 20,
    "cuda": true,
    "gpu_name": "NVIDIA GeForce RTX 5060",
    "gpu_memory_gb": "8.5",
    "gpu_count": 1,
    "timestamp": "2026-07-16T13:55:41.837284"
  },
  "config": {
    "signal_samples": 10000,
    "image_samples": 500,
    "detection_samples": 100,
    "batch_size": 256,
    "partitions": 4
  },
  "models": {
    "ew_classifier": {
      "category": "signal",
      "memory_mb": 50
    },
    "signal_denoiser": {
      "category": "signal",
      "memory_mb": 100
    },
    "threat_prioritizer": {
      "category": "signal",
      "memory_mb": 350
    },
    "rf_fingerprinter": {
      "category": "signal",
      "memory_mb": 120
    },
    "anomaly_detector": {
      "category": "signal",
      "memory_mb": 100
    },
    "resnet18": {
      "category": "image_classification",
      "memory_mb": 300
    },
    "mobilenetv3": {
      "category": "image_classification",
      "memory_mb": 150
    },
    "efficientnet_b0": {
      "category": "image_classification",
      "memory_mb": 200
    },
    "yolov8_nano": {
      "category": "object_detection",
      "memory_mb": 200
    },
    "yolov8_small": {
      "category": "object_detection",
      "memory_mb": 400
    }
  },
  "single_gpu": {
    "mode": "single_gpu",
    "device": "cuda",
    "elapsed_time": 97.1828,
    "total_samples_processed": 51700,
    "total_throughput": 532.0,
    "per_model_processed": {
      "ew_classifier": 10000,
      "signal_denoiser": 10000,
      "threat_prioritizer": 10000,
      "rf_fingerprinter": 10000,
      "anomaly_detector": 10000,
      "resnet18": 500,
      "mobilenetv3": 500,
      "efficientnet_b0": 500,
      "yolov8_nano": 100,
      "yolov8_small": 100
    },
    "num_models": 10,
    "num_batches": 40,
    "batch_size": 256,
    "avg_batch_latency_ms": 2429.57,
    "p99_batch_latency_ms": 58960.62
  },
  "total_benchmark_time": 377.05
}
