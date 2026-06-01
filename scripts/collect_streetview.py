"""
scripts/collect_streetview.py — Automatic collection of urban photos via Google Street View Static API.
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import pickle
import sys
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles (cp1251 can't encode Cyrillic -> UnicodeEncodeError)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

try:
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import unary_union
except ImportError:
    print("Please install shapely: pip install shapely>=2.0.0")
    sys.exit(1)

# Constants
COST_PER_IMAGE = 0.007
METADATA_API_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_API_URL = "https://maps.googleapis.com/maps/api/streetview"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Cities Priority and BBox  [min_lat, min_lon, max_lat, max_lon]
# Target = number of IMAGES (4 headings per point => point budget = target/4).
CITIES = {
    # Главный класс — Киев: широкий bbox всего города, цель 10k городских фото.
    "kyiv":     {"priority": 1, "target": 10000, "bbox": [50.213, 30.239, 50.590, 30.825]},
    # Европейские столицы — bbox-ы сужены к историческим центрам, где сосредоточены
    # достопримечательности, памятники и плотная застройка. Цель ~5k фото каждый.
    "warsaw":   {"priority": 2, "target": 5000, "bbox": [52.200, 20.950, 52.290, 21.080]},
    "prague":   {"priority": 3, "target": 5000, "bbox": [50.040, 14.360, 50.110, 14.480]},
    "budapest": {"priority": 4, "target": 5000, "bbox": [47.460, 19.010, 47.540, 19.100]},
}

# Logger setup
log_dir = Path("results/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "streetview_collection.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fetch_osm_urban_polygons(city_name: str, bbox: list, cache_dir: Path):
    """
    Fetch urban polygons via Overpass API and cache them locally.
    Uses 'out geom' to directly parse coordinates without complex node resolving.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{city_name}_urban_polygons.pkl"
    
    if cache_path.exists():
        logger.info(f"Loading {city_name} OSM polygons from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    min_lat, min_lon, max_lat, max_lon = bbox
    logger.info(f"Fetching {city_name} OSM polygons via Overpass API...")
    
    # Query to fetch both include (urban + landmarks) and exclude (nature) areas
    query = f"""
    [out:json][timeout:120];
    (
      way["landuse"~"residential|commercial|industrial|retail|brownfield|construction"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["highway"~"residential|primary|secondary|tertiary|pedestrian|living_street"]({min_lat},{min_lon},{max_lat},{max_lon});

      // Достопримечательности / памятники / историческая застройка (дома, монументы)
      way["tourism"~"attraction|museum"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["historic"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["building"~"cathedral|church|castle|palace|monument"]({min_lat},{min_lon},{max_lat},{max_lon});

      way["landuse"~"forest|meadow|farmland|grass"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["natural"~"wood|water"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["leisure"~"park"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """
    
    response = requests.post(
        OVERPASS_URL, 
        data={"data": query}, 
        headers={"User-Agent": "diploma_script/1.0", "Accept": "*/*"},
        timeout=90
    )
    response.raise_for_status()
    data = response.json()
    
    include_geoms = []
    exclude_geoms = []
    
    for element in data.get("elements", []):
        if element["type"] == "way" and "geometry" in element:
            coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
            tags = element.get("tags", {})
            
            # Categorize
            is_exclude = any(k in tags and v in ["forest", "meadow", "farmland", "grass", "wood", "water", "park"] 
                             for k, v in [("landuse", tags.get("landuse")), ("natural", tags.get("natural")), ("leisure", tags.get("leisure"))])
            
            geom = None
            if len(coords) >= 3 and coords[0] == coords[-1]: # Closed way -> Polygon
                geom = Polygon(coords)
            else:
                geom = LineString(coords).buffer(0.0004) # Buffer ~40m for highways/open ways
                
            if is_exclude:
                exclude_geoms.append(geom)
            else:
                include_geoms.append(geom)

    logger.info("Merging polygons (this may take a minute)...")
    urban_area = unary_union(include_geoms)
    if exclude_geoms:
        nature_area = unary_union(exclude_geoms)
        urban_area = urban_area.difference(nature_area)
        
    with open(cache_path, "wb") as f:
        pickle.dump(urban_area, f)
        
    logger.info(f"Saved {city_name} urban polygons to cache.")
    return urban_area

def generate_urban_grid(bbox: list, urban_polygon, step_meters: int = 120):
    """
    Generate grid of points with step_meters and filter out non-urban points.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # Approx degrees conversions
    lat_step = step_meters / 111320.0
    
    points = []
    lat = min_lat
    while lat <= max_lat:
        # Longitude scaling depends on latitude
        lon_step = step_meters / (111320.0 * math.cos(math.radians(lat)))
        lon = min_lon
        while lon <= max_lon:
            if urban_polygon.contains(Point(lon, lat)):
                points.append((round(lat, 6), round(lon, 6)))
            lon += lon_step
        lat += lat_step
        
    return points

def check_metadata(lat: float, lon: float, api_key: str):
    """
    Check Metadata API to ensure panorama exists and is >= 2018.
    """
    params = {
        "location": f"{lat},{lon}",
        "key": api_key
    }
    try:
        r = requests.get(METADATA_API_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if data.get("status") == "OK":
            date_str = data.get("date", "")
            if date_str:
                year = int(date_str.split("-")[0])
                if year >= 2018:
                    return True, date_str
    except Exception as e:
        logger.debug(f"Metadata check failed for {lat},{lon}: {e}")
        
    return False, ""

def download_point_images(lat: float, lon: float, date_str: str, api_key: str, out_dir: Path, city: str):
    """
    Download 4 headings for a single point.
    """
    headings = [0, 90, 180, 270]
    results = []
    
    for h in headings:
        filename = f"{lat}_{lon}_h{h}.jpg"
        out_path = out_dir / filename
        
        # Avoid re-downloading if file exists and has size
        if out_path.exists() and out_path.stat().st_size > 1000:
            results.append((out_path, h))
            continue
            
        params = {
            "size": "640x640",
            "location": f"{lat},{lon}",
            "heading": h,
            "fov": 90,
            "pitch": 0,
            "return_error_code": "true",
            "key": api_key
        }
        
        try:
            r = requests.get(IMAGE_API_URL, params=params, stream=True, timeout=15)
            r.raise_for_status()
            
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    
            if out_path.stat().st_size > 1000:
                results.append((out_path, h))
            else:
                out_path.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Error downloading {filename}: {e}")
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Collect urban street view photos")
    parser.add_argument("--api-key", type=str, default=os.environ.get("STREETVIEW_API_KEY", ""), help="Google Street View API Key")
    parser.add_argument("--max-budget-usd", type=float, default=1.90, help="Maximum budget in USD to spend")
    parser.add_argument("--cities", nargs="+", choices=list(CITIES.keys()), default=["kyiv"], help="Cities to collect")
    parser.add_argument("--step-meters", type=int, default=120, help="Grid step in meters")
    parser.add_argument("--output", type=str, default="data/images/", help="Output base directory")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted collection")
    parser.add_argument("--dry-run", action="store_true", help="Calculate points without downloading")
    
    args = parser.parse_args()
    
    if not args.api_key and not args.dry_run:
        logger.error("API Key must be provided via --api-key or STREETVIEW_API_KEY env var.")
        sys.exit(1)
        
    out_base = Path(args.output)
    osm_cache = Path("data/osm_cache")
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = splits_dir / "streetview_manifest.csv"
    
    # Sort cities by priority
    target_cities = sorted(args.cities, key=lambda c: CITIES[c]["priority"])
    
    spent_usd = 0.0
    downloaded_total = 0
    
    # Load manifest state for resume
    processed_locations = set()
    if args.resume and manifest_path.exists():
        df_manifest = pd.read_csv(manifest_path)
        if "lat" in df_manifest.columns and "lon" in df_manifest.columns:
            for _, row in df_manifest.iterrows():
                processed_locations.add((round(float(row["lat"]), 6), round(float(row["lon"]), 6)))
        logger.info(f"Resuming: Loaded {len(processed_locations)} processed points (approx {len(processed_locations) * 4} images).")

    manifest_exists = manifest_path.exists()
    
    for city in target_cities:
        if spent_usd >= args.max_budget_usd:
            logger.info("Maximum budget reached. Stopping collection.")
            break
            
        city_info = CITIES[city]
        city_dir = out_base / city
        if not args.dry_run:
            city_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"=== Processing {city.upper()} ===")
        
        # 1. Fetch OSM polygons
        urban_polygon = fetch_osm_urban_polygons(city, city_info["bbox"], osm_cache)
        
        # 2. Generate and filter grid
        logger.info(f"Generating grid with {args.step_meters}m step...")
        points = generate_urban_grid(city_info["bbox"], urban_polygon, args.step_meters)
        logger.info(f"Generated {len(points)} urban points for {city}.")
        
        if args.dry_run:
            continue
            
        # 3. Shuffle points
        random.seed(42)
        random.shuffle(points)
        
        # 4. Download Process
        city_target = city_info["target"]
        city_downloaded = 0
        
        pbar = tqdm(total=city_target, desc=f"[{city.capitalize()}]")
        
        with open(manifest_path, "a", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            if not manifest_exists:
                writer.writerow(["filename", "city", "lat", "lon", "heading", "date_collected", "source"])
                manifest_exists = True
                
            for lat, lon in points:
                if city_downloaded >= city_target:
                    logger.info(f"[{city.upper()}] Reached target of {city_target} photos.")
                    break
                if spent_usd >= args.max_budget_usd:
                    break
                    
                if args.resume and (lat, lon) in processed_locations:
                    continue
                    
                # 5. Metadata check (Free)
                is_valid, date_str = check_metadata(lat, lon, args.api_key)
                if not is_valid:
                    continue
                    
                # 6. Download Images (Paid)
                # Check budget for 4 images: 4 * 0.007 = 0.028
                if spent_usd + (4 * COST_PER_IMAGE) > args.max_budget_usd:
                    logger.warning(f"Not enough budget for 4 more images. Spent: ${spent_usd:.2f}, Limit: ${args.max_budget_usd:.2f}")
                    break
                    
                results = download_point_images(lat, lon, date_str, args.api_key, city_dir, city)
                
                # Update counters and manifest
                for path, heading in results:
                    writer.writerow([str(path).replace("\\", "/"), city, lat, lon, heading, date_str, "streetview"])
                    
                images_got = len(results)
                city_downloaded += images_got
                downloaded_total += images_got
                spent_usd += images_got * COST_PER_IMAGE
                
                f_csv.flush()
                processed_locations.add((lat, lon))
                pbar.update(images_got)
                
                # 7. Log every 100 requests (approx)
                if city_downloaded % 100 < images_got: 
                    tqdm.write(f"[{city.capitalize()}] {city_downloaded}/{city_target} фото | Потрачено: ${spent_usd:.2f} | Осталось бюджета: ${args.max_budget_usd - spent_usd:.2f}")
                    
        pbar.close()

    logger.info(f"=== Collection Summary ===")
    logger.info(f"Total downloaded: {downloaded_total} images")
    logger.info(f"Total spent: ${spent_usd:.2f} (Limit: ${args.max_budget_usd:.2f})")
    logger.info(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
