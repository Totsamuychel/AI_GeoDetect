Fix the following issues in the AI_GeoDetect project manifests and configs:

1. **Fix Windows paths in CSV manifests** (CRITICAL):
   In dataset/manifests/train.csv, val.csv, test.csv — replace all backslashes (\) 
   in the `filepath` column with forward slashes (/).
   Use Python: df['filepath'] = df['filepath'].str.replace('\\', '/', regex=False)
   Save files back in place.

2. **Fill empty `city` column** (CRITICAL):
   The `city` column is empty for all rows. Fill it with the `country` value 
   as a fallback (so the model has at least country-level classes).
   Then fill `region` with country value too if empty.

3. **Fix config paths** (configs/*.yaml):
   Update in ALL yaml files (baseline.yaml, geoclip.yaml, streetclip.yaml, 
   baseline_fast.yaml, geoclip_fast.yaml, streetclip_fast.yaml):
   - manifest_path: "dataset/manifests/train.csv"
   - image_root: "dataset/raw/osv5m/images"
   
   Do NOT change val/test paths — train.py likely handles splits internally 
   or reads train manifest and splits from it.

4. After fixing, print statistics:
   - Sample filepaths (verify forward slashes)
   - City value counts per split
   - Confirm all 3 CSVs updated