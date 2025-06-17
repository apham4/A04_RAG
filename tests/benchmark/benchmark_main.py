import time
import argparse
import statistics
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from classes.utilities import delete_directory
from main import config

def run_benchmark(use_rag : bool):
    """
    This function orchestrates the benchmark by calling a subprocess
    and then performing cleanup.
    """
    
    # --- Cleanup from previous runs ---
    print("Cleaning up previous run directories...")
    delete_directory(config.get("vectordb_directory"))
    delete_directory(config.get("cleaned_text_directory"))
    delete_directory(config.get("embeddings_directory"))
    
    Path(config.get("cleaned_text_directory")).mkdir(exist_ok=True)
    Path(config.get("embeddings_directory")).mkdir(exist_ok=True)
    Path(config.get("vectordb_directory")).mkdir(exist_ok=True)

    # --- Run the pipeline in a separate process ---
    script_to_run = Path(__file__).parent / "benchmark_sub.py"
    
    start_time = time.perf_counter()
    
    # Use subprocess.run to execute the script and wait for it to complete.
    # This ensures the process is terminated and all file locks are released.
    result = subprocess.run(
        [sys.executable, str(script_to_run), "--use-rag", str(use_rag)],
        capture_output=True,
        text=True
    )
    
    end_time = time.perf_counter()
    
    # Check if the subprocess ran successfully
    if result.returncode != 0:
        print("--- Subprocess Error ---")
        print(result.stdout)
        print(result.stderr)
        print("------------------------")
        raise RuntimeError("Subprocess failed to execute.")
    else:
        print(result.stdout) # Print the output from the subprocess

    return end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark the RAG pipeline.")
    parser.add_argument("--use-rag", type=bool, required=True, help="Whether to use RAG to augment the query.")
    parser.add_argument("--runs", type=int, default=3, help="Number of times to run the benchmark.")
    args = parser.parse_args()
    
    if args.use_rag:
        print("Running benchmark with RAG enabled.")
    else:
        print("Running benchmark without RAG.")

    all_results = []
    print(f"Starting benchmark with {args.runs} runs...")

    for i in range(args.runs):
        print(f"\n--- Starting Run {i+1}/{args.runs} ---")
        total_time = run_benchmark(args.use_rag)
        all_results.append(total_time)
        print(f"Run {i+1} finished in {total_time:.2f}s")
        
    print("\n--- Benchmark Complete ---")
    print(f"Results for full pipeline execution (ingest, embed, store):")
    print(f"Mean:  {statistics.mean(all_results):.2f}s")
    if len(all_results) > 1:
        print(f"StDev: {statistics.stdev(all_results):.2f}s")
    print(f"Min:   {min(all_results):.2f}s")
    print(f"Max:   {max(all_results):.2f}s")

if __name__ == "__main__":
    main()