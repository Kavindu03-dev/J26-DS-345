"""
Pipeline Stage 03: Aspect-Based Sentiment Analysis (ABSA).
Step 03: Multi-Model ABSA Benchmarking Across Standard & Adversarial Splits.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Fulfills Section 3.4 (Baseline Comparison) and Table 2 (Evaluation Metrics):
Compares VADER, Lexical ABSA, RoBERTa-base, and DeBERTa-v3 across:
  - Standard Benchmark (N=400)
  - 50 Hard Linguistic Challenges (N=50)
  - Combined Gold-Standard Benchmark (N=450)
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark_absa_comparison import ABSAModelBenchmark

if __name__ == "__main__":
    benchmark = ABSAModelBenchmark()
    benchmark.run_full_benchmark()
