#!/usr/bin/env python3
import os
import subprocess
import argparse

def submit_all_slurm_scripts(folder_path):
    # Ensure the directory exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Get all .slurm files in the folder
    slurm_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".in")]
    )

    if not slurm_files:
        print("No .in files found in the specified folder.")
        return

    print(f"Found {len(slurm_files)} slurm scripts in '{folder_path}':")
    for f in slurm_files:
        print(f"  - {f}")

    # Submit each .slurm file
    for slurm_file in slurm_files:
        slurm_path = os.path.join(folder_path, slurm_file)
        print(f"\nSubmitting {slurm_file} ...")
        try:
            result = subprocess.run(
                ["sbatch", slurm_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error submitting {slurm_file}:")
            print(e.stderr.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit all SLURM scripts in a folder.")
    parser.add_argument("folder", help="Path to the folder containing .slurm files")
    args = parser.parse_args()

    submit_all_slurm_scripts(args.folder)
