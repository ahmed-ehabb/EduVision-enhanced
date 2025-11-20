"""
Download Quiz Model - Standalone Script

Downloads the merged Psychology Quiz model with progress monitoring.
This ensures the model is cached before running tests.

Model: ahmedhugging12/Llama-3.2-3B-Psychology-Merged
Size: ~6GB
"""

import os
import time
from pathlib import Path

print("="*80)
print("  Quiz Model Download Script")
print("="*80)
print()

# Model configuration
MODEL_NAME = "ahmedhugging12/Llama-3.2-3B-Psychology-Merged"
CACHE_DIR = Path.cwd() / ".model_cache"

print(f"Model: {MODEL_NAME}")
print(f"Cache Directory: {CACHE_DIR}")
print(f"Estimated Size: ~6GB")
print()

# Create cache directory
CACHE_DIR.mkdir(exist_ok=True)
print(f"[OK] Cache directory created: {CACHE_DIR}")
print()

# Import libraries
print("[INFO] Importing libraries...")
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[OK] Libraries imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import libraries: {e}")
    print("Please install: pip install torch transformers")
    exit(1)

print()

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[OK] GPU detected: {gpu_name}")
    print(f"[OK] VRAM: {total_vram:.2f} GB")
else:
    print("[WARNING] No GPU detected - model will download but loading may fail")

print()
print("="*80)
print("  DOWNLOADING MODEL FILES")
print("="*80)
print()
print("This will download 2 large files (~5GB + ~1.5GB)")
print("The download may take 10-30 minutes depending on your connection.")
print("DO NOT interrupt this process.")
print()

input("Press ENTER to start download...")

start_time = time.time()

try:
    # Step 1: Download tokenizer (small, fast)
    print("\n" + "="*80)
    print("[STEP 1/2] Downloading Tokenizer...")
    print("="*80)

    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True
    )
    tokenizer_time = time.time() - tokenizer_start

    print(f"[OK] Tokenizer downloaded in {tokenizer_time:.2f}s")

    # Step 2: Download model weights (large, slow)
    print("\n" + "="*80)
    print("[STEP 2/2] Downloading Model Weights (this will take time)...")
    print("="*80)
    print()
    print("Downloading model-00001-of-00002.safetensors (~5GB)...")
    print("Downloading model-00002-of-00002.safetensors (~1.5GB)...")
    print()
    print("Progress will be shown below:")
    print("-"*80)

    model_start = time.time()

    # Download without loading to GPU (just cache the files)
    from transformers import AutoConfig

    # First download config
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True
    )

    # Then download model files (this triggers the download)
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        resume_download=True,  # Resume if interrupted
        local_files_only=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Skip unnecessary files
    )

    model_time = time.time() - model_start

    print("-"*80)
    print(f"\n[OK] Model weights downloaded in {model_time:.2f}s ({model_time/60:.1f} minutes)")
    print(f"[OK] Model cached at: {model_path}")

    # Summary
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("  DOWNLOAD COMPLETE!")
    print("="*80)
    print()
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Cache directory: {CACHE_DIR}")
    print()
    print("Model files cached successfully!")
    print("You can now run the Quiz Generator tests.")
    print()
    print("="*80)

except KeyboardInterrupt:
    print("\n\n[!] Download interrupted by user")
    print("[INFO] You can resume the download by running this script again")
    exit(1)

except Exception as e:
    print(f"\n\n[ERROR] Download failed: {e}")
    print()
    print("Common issues:")
    print("1. Network timeout - Try again with a stable connection")
    print("2. Disk space - Ensure you have at least 10GB free space")
    print("3. Firewall - Check if HuggingFace is blocked")
    print()
    print("You can retry by running this script again.")
    import traceback
    traceback.print_exc()
    exit(1)
