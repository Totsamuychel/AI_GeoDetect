"""
Аналіз географічного розкиду датасету для оцінки можливості
передбачення координат (а не лише класу-міста).
"""
import numpy as np
import pandas as pd
from pathlib import Path

R = 6371.0
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

root = Path("dataset/manifests")
frames = []
for s in ["train", "val", "test"]:
    df = pd.read_csv(root / f"{s}.csv"); df["split"] = s; frames.append(df)
full = pd.concat(frames, ignore_index=True)

print(f"Всього зображень: {len(full)}\n")
print(f"{'city':10s} {'N':>5s} {'bbox_км':>10s} {'центроїд→точка':>16s} {'медіана':>9s}")
print("-" * 60)

centroids = {}
for city, g in full.groupby("city"):
    clat, clon = g["lat"].mean(), g["lon"].mean()
    centroids[city] = (clat, clon)
    # розмір міста: діагональ bbox
    diag = haversine(g["lat"].min(), g["lon"].min(), g["lat"].max(), g["lon"].max())
    # помилка якщо завжди передбачати центроїд
    d = haversine(g["lat"].values, g["lon"].values, clat, clon)
    print(f"{city:10s} {len(g):5d} {diag:8.1f}км {d.mean():12.2f}км {np.median(d):7.2f}км")

print()
print("Відстані між центроїдами міст (км):")
cities = list(centroids.keys())
for i in range(len(cities)):
    for j in range(i+1, len(cities)):
        a, b = cities[i], cities[j]
        dd = haversine(*centroids[a], *centroids[b])
        print(f"  {a:10s} ↔ {b:10s}: {dd:7.1f} км")

# Що дає чистий "центроїд правильного міста" (стеля без міжміської плутанини)
all_d = []
for city, g in full.groupby("city"):
    clat, clon = centroids[city]
    all_d.append(haversine(g["lat"].values, g["lon"].values, clat, clon))
all_d = np.concatenate(all_d)
print()
print("=== Якщо КЛАСИФІКАЦІЯ ідеальна (100%), координата = центроїд міста ===")
print(f"  mean Haversine error : {all_d.mean():.2f} км")
print(f"  median               : {np.median(all_d):.2f} км")
print(f"  90-й перцентиль      : {np.percentile(all_d, 90):.2f} км")
print(f"  < 5 км               : {(all_d < 5).mean()*100:.1f}%")
print(f"  < 10 км              : {(all_d < 10).mean()*100:.1f}%")

# Щільність покриття
print()
print("=== Щільність покриття (фото на км²) ===")
for city, g in full.groupby("city"):
    lat_km = (g["lat"].max() - g["lat"].min()) * 111.0
    lon_km = (g["lon"].max() - g["lon"].min()) * 111.0 * np.cos(np.radians(g["lat"].mean()))
    area = max(lat_km * lon_km, 1.0)
    print(f"  {city:10s}: ~{area:7.0f} км², {len(g)/area:5.2f} фото/км²")
