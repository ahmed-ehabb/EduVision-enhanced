"""
Download required ML models for Student Module
Run this script before starting the student module for the first time
"""

import os
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "util" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "shape_predictor_68_face_landmarks.dat": {
        "url": "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat",
        "size": "95 MB",
        "description": "Dlib 68-point facial landmark detector"
    }
}

def download_file(url: str, destination: Path):
    """Download file with progress"""
    print(f"Downloading to {destination}...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")

    urllib.request.urlretrieve(url, destination, progress_hook)
    print()  # New line after progress

def main():
    """Download all required models"""
    print("=" * 60)
    print("Student Module - Model Download Script")
    print("=" * 60)

    for filename, info in MODELS.items():
        model_path = MODEL_DIR / filename

        if model_path.exists():
            print(f"\n✓ {filename} already exists")
            continue

        print(f"\n📥 Downloading {filename}")
        print(f"   Size: {info['size']}")
        print(f"   Description: {info['description']}")
        print(f"   URL: {info['url']}")

        try:
            download_file(info['url'], model_path)
            print(f"✓ Successfully downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            print("   You may need to download manually from:")
            print(f"   {info['url']}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
