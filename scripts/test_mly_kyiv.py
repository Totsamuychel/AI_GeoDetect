import requests, os, sys
sys.stdout.reconfigure(encoding="utf-8")

token = os.environ.get("MAPILLARY_API_KEY")

# Use small tile (0.01deg) like fast_download_mapillary does
resp = requests.get(
    "https://graph.mapillary.com/images",
    params={
        "access_token": token,
        "fields": "id,geometry,captured_at,thumb_256_url,thumb_1024_url,thumb_2048_url",
        "bbox": "30.50,50.44,30.51,50.45",   # 0.01deg tile in central Kyiv
        "limit": 20,
    },
    timeout=15
)
print("Status:", resp.status_code)
data = resp.json()
if resp.status_code != 200:
    print("Error:", data)
else:
    items = data.get("data", [])
    print("Images:", len(items))
    for img in items[:5]:
        has_thumb = bool(img.get("thumb_256_url") or img.get("thumb_1024_url") or img.get("thumb_2048_url"))
        print(f"  {img['id']}  has_thumb:{has_thumb}  ts:{img.get('captured_at')}")
        if has_thumb:
            print("  -->", str(img.get("thumb_256_url", ""))[:100])
