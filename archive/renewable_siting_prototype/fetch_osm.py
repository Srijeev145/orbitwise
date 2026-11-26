"""
Run this on a LAPTOP (the cloud sandbox cannot reach Overpass):
    pip install requests && python fetch_osm.py
Writes data/osm_plants.json, data/osm_substations.json, data/osm_boundary.json.
model / ml pick them up automatically:
  * existing solar + wind plants  -> training labels for the Random Forest (ml.py)
  * substations                   -> real grid-distance instead of the built-up proxy
"""
import json
import time
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
AREA = 'area["boundary"="administrative"]["admin_level"="4"]["name:en"="Karnataka"]->.k;'

QUERIES = {
    "osm_plants.json": f'[out:json][timeout:300];{AREA}('
                       'nwr["power"="plant"]["plant:source"~"solar|wind"](area.k);'
                       'nwr["power"="generator"]["generator:source"="solar"]["generator:output:electricity"~"MW"](area.k);'
                       'node["power"="generator"]["generator:source"="wind"](area.k););out center tags;',
    "osm_substations.json": f'[out:json][timeout:300];{AREA}('
                            'nwr["power"="substation"]["voltage"~"(66|110|132|220|400|765)000"](area.k););'
                            'out center tags;',
    "osm_boundary.json": '[out:json][timeout:300];rel["boundary"="administrative"]["admin_level"="4"]'
                         '["name:en"="Karnataka"];out geom;',
}

for fname, q in QUERIES.items():
    for url in URLS:
        try:
            r = requests.post(url, data={"data": q}, timeout=360)
            r.raise_for_status()
            js = r.json()
            json.dump(js, open(DATA / fname, "w"))
            print(f"{fname}: {len(js.get('elements', []))} elements")
            break
        except Exception as e:
            print(f"{fname}: {url} failed ({e}); trying next")
            time.sleep(5)
