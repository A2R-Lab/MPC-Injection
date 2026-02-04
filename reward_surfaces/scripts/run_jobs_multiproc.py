"""Run multiple jobs in parallel using multiprocessing."""

import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm


def run_command(cmd: str) -> int:
    """Run a single command and return exit code."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode
    except Exception as e:
        print(f"Error running command: {cmd}")
        print(f"  {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Run jobs in parallel')
    parser.add_argument('jobs_file', type=str, help='Path to jobs.sh file')
    parser.add_argument('--num-cpus', type=int, default=4,
                       help='Number of parallel processes')
    
    args = parser.parse_args()
    
    jobs_file = Path(args.jobs_file)
    
    if not jobs_file.exists():
        raise FileNotFoundError(f"Jobs file not found: {jobs_file}")
    
    # Read all jobs
    with open(jobs_file, 'r') as f:
        jobs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Running {len(jobs)} jobs with {args.num_cpus} parallel workers...")
    
    # Run jobs in parallel with progress bar
    with mp.Pool(processes=args.num_cpus) as pool:
        results = list(tqdm(
            pool.imap(run_command, jobs),
            total=len(jobs),
            desc="Processing"
        ))
    
    # Check for failures
    failures = sum(1 for r in results if r != 0)
    
    print(f"\nCompleted: {len(jobs) - failures}/{len(jobs)} successful")
    
    if failures > 0:
        print(f"Warning: {failures} jobs failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
