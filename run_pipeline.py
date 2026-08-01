#!/usr/bin/env python3
"""
run_pipeline.py
===============
Runs all 4 ML pipeline steps sequentially, then starts the FastAPI server.
Usage:
    python run_pipeline.py --train    # Run ML pipeline (Parts 1-4)
    python run_pipeline.py --serve    # Start API server only
    python run_pipeline.py --all      # Train + Serve
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script_rel, label):
    script = os.path.join(BASE_DIR, script_rel)
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print('='*60)
    result = subprocess.run([sys.executable, script], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n❌ {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n✅ {label} complete.")


def train():
    run("ml/1_model_creation.py",  "PART 1 — Model Creation & Feature Engineering")
    run("ml/2_model_training.py",  "PART 2 — Model Training (RF + XGBoost Ensemble)")
    run("ml/3_model_testing.py",   "PART 3 — Model Testing on Hold-out Set")
    run("ml/4_model_evaluation.py","PART 4 — Full Evaluation & Future Forecast")
    print("\n🎉 Full ML pipeline complete! Models saved to saved_model/\n")


def serve():
    print("\n🚀 Starting FastAPI server...")
    print("   API docs:    http://localhost:8000/docs")
    print("   Frontend:    http://localhost:8000/")
    print("   Press Ctrl+C to stop.\n")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], cwd=BASE_DIR)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--all" in args:
        train()
        serve()
    elif "--train" in args:
        train()
    elif "--serve" in args:
        serve()
    else:
        print(__doc__)
        print("No flag provided. Defaulting to: --all (train + serve)\n")
        train()
        serve()
