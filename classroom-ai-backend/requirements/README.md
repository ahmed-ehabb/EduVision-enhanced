# Requirements Files

This folder contains organized dependency files for different use cases.

## File Structure

```
requirements/
├── base.txt       # Core API, database, utilities
├── phase1.txt     # Phase 1 testing (minimal ML setup)
├── ml.txt         # Full ML stack (includes base.txt)
├── dev.txt        # Development tools (includes ml.txt)
└── optional.txt   # Optional features
```

## Usage

### Phase 1: Model Testing (Recommended for now)
```bash
pip install -r requirements/phase1.txt
```

**Includes**:
- FastAPI, SQLAlchemy (core)
- PyTorch with CUDA support
- Transformers, bitsandbytes, optimum
- Audio processing (soundfile, librosa)
- Basic NLP tools

**Does NOT include**:
- TensorFlow/Keras
- Face recognition packages
- Heavy video processing

**Size**: ~5-10 GB download

---

### Full ML Stack
```bash
pip install -r requirements/ml.txt
```

**Includes everything in phase1.txt plus**:
- TensorFlow/Keras
- Face recognition (dlib, face-recognition)
- Video processing (av, mediapipe)
- Full NLP stack (spacy, textblob)
- DeepFace, FER

**Size**: ~15-20 GB download

---

### Development Environment
```bash
pip install -r requirements/dev.txt
```

**Includes everything in ml.txt plus**:
- pytest, pytest-asyncio, pytest-cov
- black, flake8, mypy
- ipython, jupyter

**Size**: ~15-20 GB + dev tools

---

### Base Only (minimal)
```bash
pip install -r requirements/base.txt
```

**Includes**:
- FastAPI, Uvicorn
- SQLAlchemy, Alembic
- Authentication (JWT, bcrypt)
- Basic utilities

**Size**: ~500 MB

---

## Dependency Chain

```
base.txt
  ↓
phase1.txt  (base + minimal ML for testing)

base.txt
  ↓
ml.txt  (base + full ML stack)
  ↓
dev.txt  (ml + development tools)
```

## Installation Tips

1. **For Phase 1 Testing**: Use `phase1.txt` to save time and space
2. **For Full Development**: Use `dev.txt` to get everything
3. **For Production**: Use `ml.txt` (no dev tools needed)

## GPU Support

PyTorch packages in `phase1.txt` and `ml.txt` are configured for CUDA 12.1 (RTX 3050 compatible).

If you need CPU-only or different CUDA version:
```bash
# Edit the torch lines in phase1.txt or ml.txt
# For CPU-only:
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1

# For CUDA 11.8:
--index-url https://download.pytorch.org/whl/cu118
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
```

## Troubleshooting

### Windows Issues

Some packages may have issues on Windows:

- **dlib**: Requires Visual C++ Build Tools
- **pyaudio**: May need manual installation
- **face-recognition**: Requires dlib

**Solution**: Use `phase1.txt` which excludes these packages, or install Visual C++ Build Tools first.

### CUDA Version Mismatch

Check your CUDA version:
```bash
nvidia-smi
```

If CUDA 11.8, change PyTorch index URL from `cu121` to `cu118`.

### Package Conflicts

If you get version conflicts:
```bash
# Create fresh virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac

# Try again
pip install -r requirements/phase1.txt
```

## What's Next?

After installing dependencies:
1. Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
2. See [../testing/QUICK_START.md](../testing/QUICK_START.md) for Phase 1 testing
