from datasets import load_dataset
import os

DATASETS_TO_DOWNLOAD = {
    "hotpot_qa": "distractor",
    "trivia_qa": "unfiltered.nocontext",
    "nq_open": None
}
CACHE_DIR = "./huggingface_cache"

def download():
    print("🚀 Starting to download and cache datasets...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    for name, subset in DATASETS_TO_DOWNLOAD.items():
        print(f"\nDownloading {name} ({subset or 'default'})...")
        try:
            load_dataset(name, subset, cache_dir=CACHE_DIR)
            print(f"✅ {name} downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
            print("Please check your network connection or dataset name.")

if __name__ == "__main__":
    download()
