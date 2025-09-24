# Deep Live Cam - Mac M3 Performance Optimizations

## Overview

This document describes the performance optimizations implemented for Mac M3 to achieve >20 FPS in real-time face swapping. The optimizations leverage Apple Silicon's unique architecture including the Neural Engine, GPU cores, and unified memory.

## Performance Targets

- **Primary Goal**: >20 FPS sustained performance
- **Stretch Goal**: >30 FPS for high-performance scenarios
- **Quality**: Maintain visual quality while achieving performance targets

## Key Optimizations

### 1. Apple Silicon Specific Optimizations

#### Neural Engine Utilization
- Automatic detection of M3, M3 Pro, and M3 Max variants
- Optimized execution providers with CoreML prioritization
- Neural Engine TOPS utilization for inference acceleration

#### Metal Performance Shaders (MPS)
- GPU compute acceleration for image processing
- Hardware-accelerated video encoding with VideoToolbox
- Optimized memory management for unified memory architecture

#### CPU Core Optimization
- Performance core prioritization for compute-intensive tasks
- Efficiency core utilization for background operations
- Dynamic thread allocation based on thermal state

### 2. Memory Optimization

#### Frame Buffer Pooling
- Pre-allocated frame buffers to reduce allocation overhead
- Queue-based buffer management
- Memory usage monitoring and adaptive scaling

#### Unified Memory Management
- Optimized for Apple Silicon's unified memory architecture
- Reduced memory copying between CPU and GPU
- Intelligent memory pressure handling

### 3. Processing Pipeline Optimizations

#### Batch Processing
- Multi-frame batch processing for improved throughput
- Configurable batch sizes based on available memory
- Parallel processing of face operations

#### Asynchronous Processing
- Non-blocking frame capture and processing
- Producer-consumer pattern with frame queues
- Separate threads for capture, processing, and display

#### Adaptive Quality Control
- Dynamic quality adjustment based on FPS performance
- Frame skipping during thermal throttling
- Resolution scaling for performance optimization

### 4. Face Processing Optimizations

#### Face Cache
- Embedding-based face caching to reduce repeated computations
- Temporal coherence exploitation
- LRU cache with configurable size limits

#### Optimized Face Swapping
- Streamlined face swap pipeline
- Reduced intermediate buffer allocations
- Hardware-accelerated color space conversions

## Implementation Details

### Core Components

1. **performance_optimizer.py**
   - FPS monitoring and metrics collection
   - Frame buffer pool management
   - Apple Silicon hardware detection

2. **apple_silicon_config.py**
   - Hardware-specific configuration
   - Thermal state monitoring
   - Optimal settings recommendation

3. **face_swapper_optimized.py**
   - Optimized face swapping pipeline
   - Batch processing implementation
   - Performance monitoring integration

4. **ui_optimized.py**
   - Real-time FPS display
   - Adaptive quality controls
   - Threaded video processing

5. **core_optimized.py**
   - Main application entry point
   - Optimized argument parsing
   - Resource management

### Performance Monitoring

The system includes comprehensive performance monitoring:

- Real-time FPS calculation with moving averages
- Frame processing time measurement
- Memory usage tracking
- Thermal state monitoring
- GPU utilization metrics (where available)

## Usage

### Quick Start

```bash
# Run with optimizations (default)
python run_optimized.py

# Live camera with monitoring
python run_optimized.py --use-optimized --enable-monitoring

# Process video file
python run_optimized.py -s face.jpg -t input.mp4 -o output.mp4 --use-optimized

# Benchmark performance
python benchmark_performance.py
```

### Configuration Options

#### Basic Options
- `--use-optimized`: Enable optimizations (default: True)
- `--enable-monitoring`: Show real-time FPS (default: True)
- `--batch-size N`: Processing batch size (default: 4)

#### Quality Settings
- `--video-encoder`: Use hevc_videotoolbox for hardware acceleration
- `--video-quality`: Video quality (lower = better quality)
- Performance mode: 640x480 @ 60 FPS target
- Balanced mode: 960x540 @ 30 FPS target
- Quality mode: 1280x720 @ 24 FPS target

#### Advanced Options
- `--max-memory`: Memory limit in GB
- `--execution-threads`: Thread count (0 = auto)
- `--execution-provider`: CoreML, CPU providers

## Benchmarking

### Running Benchmarks

```bash
# Basic benchmark
python benchmark_performance.py

# Extended benchmark with custom settings
python benchmark_performance.py --duration 60 --frames 200 --resolution 1080p

# Save results
python benchmark_performance.py --save benchmark_results.json
```

### Benchmark Metrics

The benchmark tests:
- OpenCV operations (resize, color conversion, blur)
- Memory operations (allocation, copying)
- Face detection simulation
- Multi-threading performance
- Overall FPS estimation

### Expected Performance

On Mac M3 hardware:
- **M3**: 20-25 FPS @ 960x540
- **M3 Pro**: 25-35 FPS @ 960x540
- **M3 Max**: 30-45 FPS @ 960x540

Higher resolutions will reduce FPS proportionally.

## Troubleshooting

### Common Issues

1. **Low FPS Performance**
   - Check thermal state: `pmset -g thermlog`
   - Reduce batch size or resolution
   - Enable performance mode
   - Close other applications

2. **Memory Issues**
   - Reduce `--max-memory` setting
   - Lower frame buffer pool size
   - Use smaller batch sizes

3. **Model Loading Errors**
   - Ensure models directory exists
   - Check CoreML provider availability
   - Verify ONNX model compatibility

### Performance Tuning

1. **For Maximum FPS**:
   ```bash
   python run_optimized.py --batch-size 8 --video-quality 25
   ```

2. **For Balanced Performance**:
   ```bash
   python run_optimized.py --batch-size 4 --video-quality 20
   ```

3. **For Maximum Quality**:
   ```bash
   python run_optimized.py --batch-size 2 --video-quality 15
   ```

### Thermal Management

The system automatically adjusts performance based on thermal state:
- **Nominal**: Full performance
- **Fair**: 90% performance, 80% batch size
- **Serious**: 70% performance, 60% batch size
- **Critical**: 50% performance, 40% batch size

## Technical Architecture

### Processing Pipeline

1. **Frame Capture**
   - Camera/video input
   - Format conversion
   - Frame queuing

2. **Face Detection**
   - Optimized face detection
   - Face embedding extraction
   - Cache lookup/update

3. **Face Swapping**
   - Source-target face mapping
   - Hardware-accelerated blending
   - Batch processing

4. **Output Rendering**
   - Color space conversion
   - Display/encoding
   - FPS monitoring

### Memory Management

```
[Camera] → [Input Queue] → [Processing] → [Output Queue] → [Display]
                ↓              ↓             ↓
           [Buffer Pool]  [Face Cache]  [Frame Metrics]
```

### Threading Model

- **Main Thread**: UI and coordination
- **Capture Thread**: Camera input
- **Processing Thread**: Face operations
- **Display Thread**: Output rendering
- **Monitoring Thread**: Performance metrics

## Dependencies

### Required Packages
- opencv-python (4.10.0.84)
- onnxruntime-silicon (1.16.3) - for Apple Silicon
- torch (2.5.1) - MPS support
- insightface (0.7.3)
- customtkinter (5.2.2)

### System Requirements
- macOS 12+ with Apple Silicon (M1/M2/M3)
- 8GB+ unified memory recommended
- Python 3.9+

## Future Improvements

### Planned Optimizations
1. **Model Optimization**
   - CoreML model conversion
   - Quantization for faster inference
   - Model pruning

2. **Advanced Caching**
   - Persistent face cache across sessions
   - Predictive face loading
   - Smart cache eviction

3. **Hardware Utilization**
   - Direct Metal compute shaders
   - Advanced Neural Engine integration
   - Custom BNNS operations

4. **Quality Improvements**
   - Temporal consistency enhancement
   - Advanced color matching
   - Motion-aware processing

## Contributing

When contributing performance improvements:

1. **Benchmark First**: Always run benchmarks before/after changes
2. **Profile Changes**: Use appropriate profiling tools
3. **Test Thoroughly**: Test on different M3 variants if possible
4. **Document Impact**: Include performance impact in commit messages

## Support

For performance-related issues:
1. Run the benchmark: `python benchmark_performance.py`
2. Check system info and thermal state
3. Include benchmark results in issue reports
4. Specify M3 variant (M3/Pro/Max) and memory configuration