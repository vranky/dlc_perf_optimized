# Troubleshooting Guide - Mac M3 Performance Issues

## Issue: "Frame processor face_swapper_optimized not found"

### Quick Fix
Use the standard launcher with M3 optimizations instead:

```bash
# Instead of:
python run_optimized.py --use-optimized --batch-size 4

# Use:
python run_m3_optimized.py --batch-size 4
```

### Root Cause
The optimized frame processor requires all dependencies to be installed and properly configured.

## Issue: Application stuck at "Apple Silicon M3 detected - optimizations enabled"

### Immediate Solution
1. **Use the standard launcher with optimizations:**
   ```bash
   python run_m3_optimized.py
   ```

2. **Or run debug to identify issues:**
   ```bash
   python debug_issue.py
   ```

3. **Install missing dependencies:**
   ```bash
   # Activate virtual environment first
   source venv/bin/activate

   # Install all dependencies
   pip install -r requirements.txt

   # For Apple Silicon specifically:
   pip uninstall onnxruntime onnxruntime-silicon -y
   pip install onnxruntime-silicon==1.16.3
   ```

### Dependency Issues

If you see missing dependencies, install them step by step:

```bash
# Core dependencies
pip install numpy opencv-python torch torchvision
pip install insightface customtkinter pillow

# Apple Silicon specific
pip install onnxruntime-silicon==1.16.3

# For M3 specifically, ensure CoreML support
pip install coremltools
```

### Model Files Missing

Download the required model files:

```bash
# Create models directory
mkdir -p models

# Download models manually:
# 1. GFPGANv1.4.pth from: https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth
# 2. inswapper_128_fp16.onnx from: https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx

# Place both files in the models/ directory
```

## Working Commands for Mac M3

### Option 1: Simple M3 Optimized (Recommended)
```bash
python run_m3_optimized.py
```

### Option 2: Standard with CoreML
```bash
python run.py --execution-provider coreml
```

### Option 3: Standard CPU Mode
```bash
python run.py --execution-provider cpu
```

## Performance Tuning on M3

### Maximum Performance
```bash
python run_m3_optimized.py --execution-provider coreml --execution-threads 8
```

### Balanced Performance
```bash
python run_m3_optimized.py --execution-provider coreml --max-memory 8
```

### Debug Mode
```bash
python debug_issue.py
```

## Common Issues and Solutions

### 1. "Module not found" errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### 2. "gettext" import errors
Fixed in the codebase - the gettext.py module has been renamed to avoid conflicts.

### 3. "typing" import errors
Fixed in the codebase - the typing.py module has been renamed to face_types.py.

### 4. CoreML execution provider not available
```bash
# Install CoreML support
pip install coremltools
pip install onnxruntime-silicon==1.16.3
```

### 5. Low FPS performance
```bash
# Check thermal state
pmset -g thermlog

# Reduce memory usage
python run_m3_optimized.py --max-memory 6

# Use performance mode
python run_m3_optimized.py --video-quality 25
```

### 6. Camera not detected
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

## System Requirements Check

### Minimum for M3
- macOS 12.0+
- 8GB unified memory
- Python 3.10 (exactly)
- 5GB free space

### Verify Installation
```bash
# Check Python version (must be 3.10)
python --version

# Check if on Apple Silicon
python -c "import platform; print(f'Apple Silicon: {platform.processor() == \"arm\"}')"

# Run full diagnostic
python debug_issue.py
```

## Performance Expectations

### M3 Base (8GB)
- Live camera: 20-25 FPS @ 960x540
- Video processing: 15-20 FPS @ 1080p

### M3 Pro (18GB)
- Live camera: 25-35 FPS @ 960x540
- Video processing: 20-30 FPS @ 1080p

### M3 Max (36GB)
- Live camera: 30-45 FPS @ 960x540
- Video processing: 25-35 FPS @ 1080p

## Getting Help

1. **Run diagnostics first:**
   ```bash
   python debug_issue.py
   ```

2. **Include this information when reporting issues:**
   - Mac model (M3/Pro/Max)
   - Memory amount
   - macOS version
   - Python version
   - Full error message
   - Output from debug script

3. **Test with minimal command:**
   ```bash
   python run_m3_optimized.py
   ```

## Alternative Installation Method

If you continue having issues, try the conservative approach:

```bash
# Start fresh
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate

# Install minimal dependencies first
pip install numpy opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Test basic functionality
python -c "import cv2, torch; print('Basic imports work')"

# Then install remaining dependencies
pip install insightface customtkinter pillow
pip install onnxruntime-silicon==1.16.3

# Finally test the application
python run_m3_optimized.py
```