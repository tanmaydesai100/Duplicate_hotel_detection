import json
import csv
import argparse
from pathlib import Path

ADDRESS_ORDER = [
    "addr:housenumber",
    "addr:street",
    "addr:city",
    "addr:county",
    "addr:postcode",
    "addr:country",
]

def build_address(tags):
    parts = []
    for key in ADDRESS_ORDER:
        val = tags.get(key)
        if val:
            parts.append(val)
    return ", ".join(parts)

def json_to_csv(infile: Path, outfile: Path):
    data = json.loads(infile.read_text(encoding="utf-8"))
    elements = data.get("elements", [])

    # first pass: collect all non-address tag keys to use as columns
    other_tag_keys = set()
    for elem in elements:
        for k in elem.get("tags", {}):
            if not k.startswith("addr:"):
                other_tag_keys.add(k)

    # define CSV columns
    columns = ["type", "id", "lat", "lon", "address"] + sorted(other_tag_keys)

    with outfile.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        for elem in elements:
            row = {
                "type": elem.get("type"),
                "id": elem.get("id"),
                # for nodes, lat/lon are top‐level; for ways/relations we assume we ran the conversion
                "lat": elem.get("lat", ""),
                "lon": elem.get("lon", ""),
            }

            tags = elem.get("tags", {})
            # build merged address string
            row["address"] = build_address(tags)

            # fill other tag columns
            for key in other_tag_keys:
                row[key] = tags.get(key, "")

            writer.writerow(row)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert flattened OSM nodes JSON to CSV, merging address fields."
    )
    p.add_argument("infile", type=Path, help="Input JSON file")
    p.add_argument("outfile", type=Path, help="Output CSV file")
    args = p.parse_args()

    json_to_csv(args.infile, args.outfile)
