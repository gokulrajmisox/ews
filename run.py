"""
run.py - Main CLI & Server Runner for SilentWindow.

Usage:
  python run.py                 # Starts web server and opens dashboard at http://127.0.0.1:8000
  python run.py --server        # Run server only
  python run.py --train         # Train model and calibrator
  python run.py --evaluate      # Run chronological evaluation on test set
  python run.py --ablation      # Run architectural ablation study
  python run.py --noise-test    # Run sensor noise stress test
  python run.py --test          # Run pytest unit test suite
  python run.py --all           # Execute full pipeline end-to-end and launch server
  python run.py --reproduce     # Run tests, train, evaluate, ablate, and stress-test without server
"""

import sys
import os
import argparse
import webbrowser
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="SilentWindow - Trust-Aware Early Warning System")
    parser.add_argument("--server", action="store_true", help="Launch FastAPI server and dashboard")
    parser.add_argument("--train", action="store_true", help="Train XGBoost model and calibrate probabilities")
    parser.add_argument("--evaluate", action="store_true", help="Run chronological test evaluation")
    parser.add_argument("--ablation", action="store_true", help="Run systematic ablation study")
    parser.add_argument("--noise-test", action="store_true", help="Run noise stress testing")
    parser.add_argument("--test", action="store_true", help="Run pytest unit test suite")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline end-to-end and launch server")
    parser.add_argument("--reproduce", action="store_true", help="Run the reproducible pipeline without launching the server")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default 127.0.0.1)")

    args = parser.parse_args()

    # If no specific action passed, launch server (or run pipeline if artifacts missing)
    if not (args.server or args.train or args.evaluate or args.ablation or args.noise_test or args.test or args.all or args.reproduce):
        args.server = True

    if args.reproduce:
        args.test = True
        args.train = True
        args.evaluate = True
        args.ablation = True
        args.noise_test = True

    if args.all:
        args.test = True
        args.train = True
        args.evaluate = True
        args.ablation = True
        args.noise_test = True
        args.server = True

    if args.test:
        print("\n[1/5] Running Pytest Unit Test Suite...")
        import pytest
        code = pytest.main(["tests/test_silentwindow.py", "-v"])
        if code != 0:
            print("Tests failed! Aborting pipeline.")
            sys.exit(code)

    if args.train:
        print("\n[2/5] Training XGBoost Model & Fitting Probability Calibration...")
        from ml.train import train_and_calibrate
        train_and_calibrate()

    if args.evaluate:
        print("\n[3/5] Running Chronological Cohort Evaluation...")
        from ml.evaluation import run_evaluation
        run_evaluation()

    if args.ablation:
        print("\n[4/5] Running Systematic Ablation Study...")
        from ml.ablation import run_ablation_study
        run_ablation_study()

    if args.noise_test:
        print("\n[5/5] Running Sensor Noise Stress Testing...")
        from ml.noise_test import NoiseStressTester
        tester = NoiseStressTester()
        tester.run_stress_test(intensity=0.5, max_patients=30)

    if args.server:
        # Check required artifacts exist
        if not os.path.exists("models/xgboost_model.joblib") or not os.path.exists("results/evaluation_summary.json"):
            print("Model artifacts or evaluation results missing. Training and evaluating first...")
            from ml.train import train_and_calibrate
            from ml.evaluation import run_evaluation
            train_and_calibrate()
            run_evaluation()

        url = f"http://{args.host}:{args.port}"
        print(f"\n=======================================================")
        print(f"  SILENTWINDOW CLINICAL CDS PROTOTYPE RUNNING")
        print(f"  Access Dashboard: {url}")
        print(f"  API Documentation: {url}/docs")
        print(f"  Press Ctrl+C to terminate.")
        print(f"=======================================================\n")
        
        try:
            webbrowser.open(url)
        except Exception:
            pass

        uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
