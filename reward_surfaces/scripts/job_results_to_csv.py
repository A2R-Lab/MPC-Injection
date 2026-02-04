"""Convert individual job results to a single CSV file."""

import argparse
import json
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Convert job results to CSV')
    parser.add_argument('job_dir', type=str, help='Job directory with results/')
    
    args = parser.parse_args()
    
    job_dir = Path(args.job_dir)
    results_dir = job_dir / "results"
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Collect all JSON files
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON result files found in {results_dir}")
    
    print(f"Found {len(json_files)} result files")
    
    # Load all results
    all_results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by coordinates
    df = df.sort_values(['dim0', 'dim1'])
    
    # Save to CSV
    csv_path = job_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Saved {len(df)} results to {csv_path}")
    print(f"\nAvailable columns: {', '.join(df.columns)}")
    print(f"\nReward statistics:")
    print(f"  Mean: {df['episode_rewards'].mean():.2f}")
    print(f"  Std:  {df['episode_rewards'].std():.2f}")
    print(f"  Min:  {df['episode_rewards'].min():.2f}")
    print(f"  Max:  {df['episode_rewards'].max():.2f}")


if __name__ == "__main__":
    main()
