import os
from huggingface_hub import hf_hub_download, HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "NUS-UAL/global-streetscapes"

# Let's find one actual file path in the repo
api = HfApi()
try:
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    # Filter for image files to see their naming convention
    img_files = [f for f in files if "img/" in f][:5]
    print(f"Sample files in repo: {img_files}")
    
    if img_files:
        test_file = img_files[0]
        print(f"Attempting to download: {test_file}")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=test_file,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"Success! Downloaded to: {path}")
    else:
        print("No image files found in repo.")
except Exception as e:
    print(f"Error: {e}")
