#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path

def add_description(in_csv: Path, out_csv: Path):
    # 1) Load, preserving empty cells
    df = pd.read_csv(in_csv, dtype=str).fillna("")
    
    # 2) Ensure 'Name' exists
    if "Name" not in df.columns:
        df["Name"] = ""
    
    # 3) Ensure 'description' exists
    if "description" not in df.columns:
        # if you had any extra fields, you'd collapse them here,
        # but since there are none, just create an empty column
        df["description"] = ""
    
    # 4) Reorder (and drop any other stray columns)
    final_cols = ["Name", "lat", "lon", "address", "phone", "URLs", "description"]
    # add any missing cols as empty
    for c in final_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[final_cols]
    
    # 5) Write out
    df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Ensure cleaned CSV has Name and description columns."
    )
    p.add_argument("infile", type=Path, help="Path to existing cleaned_output.csv")
    p.add_argument("outfile", type=Path, help="Path to write cleaned_with_desc.csv")
    args = p.parse_args()

    add_description(args.infile, args.outfile)
