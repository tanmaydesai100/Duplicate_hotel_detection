import json
import requests
import time

# Configuration for Nominatim reverse geocoding
NOMINATIM_URL = 'https://nominatim.openstreetmap.org/reverse'
USER_AGENT = 'osm-address-enhancer/1.0 (desaitanmay200uk@gmail.com)'

# Address fields we want to extract
ADDRESS_FIELDS = {
    'city': ['city', 'town', 'village'],
    'country': ['country'],
    'postcode': ['postcode'],
    'street': ['road', 'pedestrian', 'path', 'footway']
}


def reverse_geocode(lat, lon, timeout=10):
    """
    Call Nominatim reverse geocoding to get address components for a given lat/lon.
    Returns a dict of address tags (addr:city, addr:country, etc.) if available.
    """
    params = {
        'format': 'json',
        'lat': lat,
        'lon': lon,
        'addressdetails': 1,
    }
    headers = {'User-Agent': USER_AGENT}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        address = data.get('address', {})
        tags = {}
        # Map Nominatim fields to OSM addr:* tags
        for osm_key, nom_keys in ADDRESS_FIELDS.items():
            for nom_key in nom_keys:
                if nom_key in address:
                    tags[f'addr:{osm_key}'] = address[nom_key]
                    break
        return tags
    except requests.RequestException as e:
        print(f"Warning: reverse_geocode({lat}, {lon}) failed: {e}")
        return {}


def enhance_file(input_path, output_path):
    # Load the Overpass JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data.get('elements', []))
    print(f"Processing {total} elements...")

    # Iterate elements
    for i, element in enumerate(data.get('elements', []), start=1):
        if element.get('type') == 'node':
            tags = element.setdefault('tags', {})
            missing = [k for k in ADDRESS_FIELDS.keys() if f'addr:{k}' not in tags]
            if missing:
                lat = element.get('lat')
                lon = element.get('lon')
                new_tags = reverse_geocode(lat, lon)
                if new_tags:
                    tags.update({k: v for k, v in new_tags.items() if k not in tags})
                print(f"[{i}/{total}] {lat:.6f},{lon:.6f} → added: {list(new_tags.keys())}")
                time.sleep(1)

    # Ensure output directory exists and write updated JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Done: enhanced data written to '{output_path}'")
    except Exception as e:
        print(f"Error writing output file '{output_path}': {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Enhance OSM Overpass JSON with address tags')
    parser.add_argument('input_file', help='Input Overpass JSON file')
    parser.add_argument('output_file', help='Output JSON file with enhanced tags')
    args = parser.parse_args()

    enhance_file(args.input_file, args.output_file)
