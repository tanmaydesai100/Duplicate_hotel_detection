#!/usr/bin/env python3
import argparse
import pandas as pd

def filter_empty_addressB(input_path: str, output_path: str) -> None:
    """
    Load a CSV, drop rows where addressB is empty or NaN,
    and write the cleaned data to a new CSV.
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(input_path)

    # Filter: keep rows where addressB is not null AND not the empty string
    mask = df['addressB'].notna() & (df['addressB'].astype(str).str.strip() != '')
    df_filtered = df[mask]

    # Write the filtered DataFrame to a new CSV (no index column)
    df_filtered.to_csv(output_path, index=False)
    print(f"Filtered data written to {output_path} ({len(df_filtered)} rows)")

def main():
    parser = argparse.ArgumentParser(
        description="Remove rows with empty addressB from a CSV."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "output_csv",
        help="Path where the filtered CSV will be written"
    )
    args = parser.parse_args()

    filter_empty_addressB(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()
