import json
import argparse
from pathlib import Path

def extract_types(obj, found_types=None):
    if found_types is None:
        found_types = set()
        
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "type":
                found_types.add(value)
            extract_types(value, found_types)
    elif isinstance(obj, list):
        for item in obj:
            extract_types(item, found_types)
    
    return found_types

def main():
    parser = argparse.ArgumentParser(
        description="Extract unique values for all 'type' keys in a JSON file."
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the JSON file to process"
    )
    args = parser.parse_args()

    # Load JSON from file
    with args.json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract and print unique 'type' values
    unique_types = extract_types(data)
    print("Unique 'type' values found:")
    for t in sorted(unique_types):
        print(f"  - {t}")

if __name__ == "__main__":
    main()
