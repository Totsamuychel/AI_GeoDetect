import re
from pathlib import Path

def update_config(file_path):
    print(f"Updating {file_path}...")
    content = file_path.read_text(encoding="utf-8")
    
    # Update manifest_path
    content = re.sub(
        r'(manifest_path:\s*)[\'\"]?.*[\'\"]?', 
        r'\1"dataset/manifests/train.csv"', 
        content
    )
    
    # Update image_root
    content = re.sub(
        r'(image_root:\s*)[\'\"]?.*[\'\"]?', 
        r'\1"dataset/raw/osv5m/images"', 
        content
    )
    
    file_path.write_text(content, encoding="utf-8")

def main():
    configs_dir = Path("configs")
    for yaml_file in configs_dir.glob("*.yaml"):
        update_config(yaml_file)
    print("All configs updated.")

if __name__ == "__main__":
    main()
