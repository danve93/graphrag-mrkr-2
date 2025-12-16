
import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variants", required=True)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    dataset_path = Path(args.dataset)
    
    # Check for config
    config_path = Path("evals/ragas/config.yaml")
    if not config_path.exists():
        print(f"Config not found at {config_path}, using example config.")
        config_path = Path("evals/ragas/config.example.yaml")
        if not config_path.exists():
             print("Error: No config.yaml or config.example.yaml found.")
             sys.exit(1)

    # Base runner script
    runner_script = Path("evals/ragas/ragas_runner.py")
    if not runner_script.exists():
         print(f"Error: Runner script not found at {runner_script}")
         sys.exit(1)

    print(f"Starting benchmark for {len(variants)} variants using dataset: {dataset_path}")

    success_count = 0
    final_exit_code = 0
    
    for variant in variants:
        print(f"\n>>> Running variant: {variant}")
        
        # Define output path
        out_path = Path(f"reports/ragas/run_{variant}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", str(runner_script),
            "--config", str(config_path),
            "--dataset", str(dataset_path),
            "--variant", variant,
            "--out", str(out_path)
        ]
        
        # Stream output
        try:
            # We use subprocess.run to block until complete
            res = subprocess.run(cmd, text=True, cwd=os.getcwd())
            if res.returncode == 0:
                print(f">>> Variant {variant} completed successfully.")
                success_count += 1
            else:
                 print(f">>> Variant {variant} failed with exit code {res.returncode}.")
                 final_exit_code = res.returncode
        except Exception as e:
             print(f"Failed to execute runner: {e}")
             final_exit_code = 1

    if success_count == len(variants):
        print("\nAll variants completed successfully.")
        sys.exit(0)
    else:
        print(f"\nCompleted {success_count}/{len(variants)} variants.")
        sys.exit(final_exit_code if final_exit_code != 0 else 1)

if __name__ == "__main__":
    main()
