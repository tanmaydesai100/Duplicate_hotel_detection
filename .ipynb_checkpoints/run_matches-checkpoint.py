#!/usr/bin/env python3
import subprocess
import sys

def run_sequence(ranges, output_file, script="addressComparison.py"):
    for start, end in ranges:
        cmd = [
            sys.executable, script,
            "--start", str(start),
            "--end",   str(end),
            "--output", output_file,
            "--skip_enrich"
        ]
        print(f"\n→ Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Command failed with exit code {e.returncode}. Stopping.")
            break

if __name__ == "__main__":
    # create 100-row chunks from 1800 up to 7000
    chunks = [(i, i+100) for i in range(60000,70000 ,100)]
    run_sequence(chunks, output_file="matches.csv")
    print("\n✅ All chunks complete — ready for next command.\n")
