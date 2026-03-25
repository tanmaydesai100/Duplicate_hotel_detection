import json
import argparse
from pathlib import Path

def make_node_from_feature(feat):
    """
    Given a 'way' or 'relation' with a 'bounds' block and 'tags',
    return a 'node' object at the midpoint of the bbox, carrying all tags.
    """
    b = feat["bounds"]
    lat = (b["minlat"] + b["maxlat"]) / 2
    lon = (b["minlon"] + b["maxlon"]) / 2

    return {
        "type": "node",
        "id": feat["id"],
        "lat": lat,
        "lon": lon,
        "tags": dict(feat.get("tags", {}))
    }

def process_elements(elements):
    """
    Walk the list of elements, converting each to a 'node' object.
    """
    output = []
    print(len(elements))
    for elem in elements:
        t = elem.get("type")
        if t == "node":
            # already has lat/lon and tags
            output.append({
                "type": "node",
                "id": elem["id"],
                "lat": elem["lat"],
                "lon": elem["lon"],
                "tags": dict(elem.get("tags", {}))
            })
        elif t in ("way", "relation"):
            # use the bbox midpoint
            output.append(make_node_from_feature(elem))
        else:
            # skip anything else
            continue
    return output

def main():
    parser = argparse.ArgumentParser(
        description="Convert OSM 'elements' (node/way/relation) into a flat list of nodes."
    )
    parser.add_argument(
        "infile", type=Path,
        help="Input JSON file containing a top‐level 'elements' array"
    )
    parser.add_argument(
        "-o", "--outfile", type=Path, default=None,
        help="If provided, write output JSON to this file (otherwise prints to stdout)"
    )
    args = parser.parse_args()

    # Load the entire JSON
    data = json.loads(args.infile.read_text(encoding="utf-8"))
    elems = data.get("elements", [])

    # Process into nodes
    nodes = process_elements(elems)

    # Prepare output structure
    out = {"elements": nodes}

    # Serialize
    out_str = json.dumps(out, indent=2)

    if args.outfile:
        args.outfile.write_text(out_str, encoding="utf-8")
        print(f"Wrote {len(nodes)} nodes to {args.outfile}")
    else:
        print(out_str)

if __name__ == "__main__":
    main()
