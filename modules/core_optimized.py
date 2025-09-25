"""
Optimized core module for Mac M3 with performance improvements
"""

import os
import sys
import platform
import warnings
import argparse
import signal
import shutil
import time
from typing import List, Optional
import threading
import multiprocessing as mp

# Optimize thread settings for Apple Silicon
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    # Optimal settings for M3
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(mp.cpu_count())
    # Enable Metal Performance Shaders
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
else:
    # Single thread for CUDA
    if any(arg.startswith('--execution-provider') for arg in sys.argv):
        os.environ['OMP_NUM_THREADS'] = '1'

# Reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import onnxruntime
import tensorflow
import cv2

import modules.globals
import modules.metadata
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension, is_image, is_video, detect_fps,
    create_video, extract_frames, get_temp_frame_paths,
    restore_audio, create_temp, move_temp, clean_temp,
    normalize_output_path
)
from modules.performance_optimizer import (
    get_recommended_settings,
    AppleSiliconOptimizer,
    FPSMonitor,
    optimize_opencv_settings
)

# Check platform
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.processor() == 'arm'

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Performance monitoring
global_fps_monitor = FPSMonitor(window_size=100)


def parse_args() -> None:
    """Parse command line arguments with optimized defaults"""
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()

    # Basic arguments
    program.add_argument('-s', '--source', help='select source image', dest='source_path')
    program.add_argument('-t', '--target', help='select target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')

    # Frame processors
    program.add_argument('--frame-processor', help='pipeline of frame processors',
                        dest='frame_processor', default=['face_swapper'],
                        choices=['face_swapper', 'face_swapper_optimized', 'face_enhancer'],
                        nargs='+')

    # Performance options
    program.add_argument('--use-optimized', help='use optimized processing',
                        dest='use_optimized', action='store_true', default=True)
    program.add_argument('--batch-size', help='batch size for processing',
                        dest='batch_size', type=int, default=4)
    program.add_argument('--enable-monitoring', help='enable FPS monitoring',
                        dest='enable_monitoring', action='store_true', default=True)

    # Video options
    program.add_argument('--keep-fps', help='keep original fps',
                        dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio',
                        dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames',
                        dest='keep_frames', action='store_true', default=False)

    # Face options
    program.add_argument('--many-faces', help='process every face',
                        dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter NSFW content',
                        dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces',
                        dest='map_faces', action='store_true', default=False)

    # Video encoding
    program.add_argument('--video-encoder', help='video encoder',
                        dest='video_encoder', default='libx264',
                        choices=['libx264', 'libx265', 'libvpx-vp9', 'hevc_videotoolbox'])
    program.add_argument('--video-quality', help='video quality',
                        dest='video_quality', type=int, default=18,
                        choices=range(52), metavar='[0-51]')

    # UI options
    program.add_argument('-l', '--lang', help='UI language', default="en")
    program.add_argument('--live-mirror', help='mirror live camera',
                        dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='resizable live camera',
                        dest='live_resizable', action='store_true', default=True)

    # Performance options
    program.add_argument('--max-memory', help='maximum RAM in GB',
                        dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider',
                        dest='execution_provider', default=suggest_default_providers(),
                        choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='execution threads',
                        dest='execution_threads', type=int, default=suggest_execution_threads())

    # Version
    program.add_argument('-v', '--version', action='version',
                        version=f'{modules.metadata.name} {modules.metadata.version} (Optimized)')

    args = program.parse_args()

    # Apply optimized settings
    apply_args(args)


def apply_args(args) -> None:
    """Apply parsed arguments with optimizations"""
    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(
        modules.globals.source_path,
        modules.globals.target_path,
        args.output_path
    )

    # Use optimized processor when requested
    if args.use_optimized:
        # Replace face_swapper with optimized version
        optimized_processors = []
        for processor in args.frame_processor:
            if processor == 'face_swapper':
                optimized_processors.append('face_swapper_optimized')
            else:
                optimized_processors.append(processor)
        modules.globals.frame_processors = optimized_processors
    else:
        modules.globals.frame_processors = args.frame_processor

    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces

    # Use hardware accelerated encoder on Mac
    if IS_APPLE_SILICON and args.video_encoder == 'libx264':
        modules.globals.video_encoder = 'hevc_videotoolbox'  # Hardware accelerated
    else:
        modules.globals.video_encoder = args.video_encoder

    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang

    # Store optimization settings
    modules.globals.use_optimized = args.use_optimized
    modules.globals.batch_size = args.batch_size
    modules.globals.enable_monitoring = args.enable_monitoring

    # Enhancer settings
    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False


def suggest_max_memory() -> int:
    """Suggest max memory based on platform"""
    if IS_APPLE_SILICON:
        # M3 has unified memory, be conservative
        return 8
    elif platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_default_providers() -> List[str]:
    """Suggest default execution providers"""
    if IS_APPLE_SILICON:
        return ['coreml', 'cpu']
    return ['cpu']


def suggest_execution_providers() -> List[str]:
    """Get available execution providers"""
    providers = onnxruntime.get_available_providers()
    encoded = []

    for provider in providers:
        name = provider.replace('ExecutionProvider', '').lower()
        encoded.append(name)

    return encoded


def suggest_execution_threads() -> int:
    """Suggest optimal thread count"""
    if IS_APPLE_SILICON:
        # Use all performance cores on M3
        return mp.cpu_count()
    elif 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    elif 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    """Decode execution provider names"""
    available = onnxruntime.get_available_providers()
    decoded = []

    for provider in available:
        provider_name = provider.replace('ExecutionProvider', '').lower()
        if any(ep in provider_name for ep in execution_providers):
            decoded.append(provider)

    return decoded


def limit_resources() -> None:
    """Limit and optimize resource usage"""
    # Configure tensorflow
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

    # Configure memory limits
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3

        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            try:
                # Get current limits to avoid exceeding maximum
                soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_DATA)
                
                # Use the smaller of requested memory or hard limit
                if hard_limit != resource.RLIM_INFINITY:
                    memory = min(memory, hard_limit)
                
                resource.setrlimit(resource.RLIMIT_DATA, (memory, hard_limit))
            except (OverflowError, OSError, ValueError):
                # Skip memory limit if it fails - common on macOS with unified memory
                if IS_APPLE_SILICON:
                    update_status('Skipping memory limits on Apple Silicon unified memory system')
                pass


def release_resources() -> None:
    """Release GPU/memory resources"""
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()

    # Clear OpenCV buffers
    cv2.destroyAllWindows()


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    """Update status with timestamp"""
    timestamp = time.strftime('%H:%M:%S')
    print(f'[{timestamp}][{scope}] {message}')

    if not modules.globals.headless:
        try:
            import modules.ui as ui
            ui.update_status(message)
        except:
            pass


def pre_check() -> bool:
    """Pre-flight checks"""
    if sys.version_info < (3, 9):
        update_status('Python version not supported - requires 3.9+')
        return False

    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed')
        return False

    if IS_APPLE_SILICON:
        update_status('Apple Silicon M3 detected - optimizations enabled')

    return True


def start_optimized() -> None:
    """Start optimized processing"""
    # Import optimized face swapper
    from modules.processors.frame import face_swapper_optimized

    # Initialize with optimizations
    optimize_opencv_settings()

    # Check processors
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if hasattr(frame_processor, 'pre_start') and not frame_processor.pre_start():
            return

    update_status('Starting optimized processing...')

    # Monitor performance if enabled
    if modules.globals.enable_monitoring:
        global_fps_monitor.start_frame()

    # Process based on target type
    if has_image_extension(modules.globals.target_path):
        process_image_optimized()
    else:
        process_video_optimized()

    # Show performance metrics
    if modules.globals.enable_monitoring:
        global_fps_monitor.end_frame()
        metrics = global_fps_monitor.get_metrics()
        update_status(f'Performance: {metrics.fps:.1f} FPS, {metrics.frame_time:.1f}ms per frame')


def process_image_optimized() -> None:
    """Process image with optimizations"""
    import modules.ui as ui

    if modules.globals.nsfw_filter:
        if ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return

    try:
        shutil.copy2(modules.globals.target_path, modules.globals.output_path)
    except Exception as e:
        update_status(f'Error copying file: {e}')
        return

    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Processing...', frame_processor.NAME)
        if hasattr(frame_processor, 'process_image'):
            frame_processor.process_image(
                modules.globals.source_path,
                modules.globals.output_path,
                modules.globals.output_path
            )
        release_resources()

    if is_image(modules.globals.target_path):
        update_status('Image processing completed!')
    else:
        update_status('Image processing failed!')


def process_video_optimized() -> None:
    """Process video with optimizations"""
    import modules.ui as ui

    if modules.globals.nsfw_filter:
        if ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return

    if not modules.globals.map_faces:
        update_status('Creating temporary resources...')
        create_temp(modules.globals.target_path)
        update_status('Extracting frames...')
        extract_frames(modules.globals.target_path)

    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)

    # Process with each frame processor
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Processing frames...', frame_processor.NAME)
        if hasattr(frame_processor, 'process_video_optimized'):
            # Use optimized method if available
            frame_processor.process_video_optimized(
                modules.globals.source_path,
                temp_frame_paths
            )
        elif hasattr(frame_processor, 'process_video'):
            frame_processor.process_video(
                modules.globals.source_path,
                temp_frame_paths
            )
        release_resources()

    # Create output video
    if modules.globals.keep_fps:
        update_status('Detecting FPS...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creating video with {fps} FPS...')
        create_video(modules.globals.target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')
        create_video(modules.globals.target_path)

    # Handle audio
    if modules.globals.keep_audio:
        update_status('Restoring audio...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)

    # Cleanup
    clean_temp(modules.globals.target_path)

    if is_video(modules.globals.target_path):
        update_status('Video processing completed!')
    else:
        update_status('Video processing failed!')


def destroy(to_quit=True) -> None:
    """Cleanup and exit"""
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    release_resources()
    if to_quit:
        quit()


def run() -> None:
    """Main entry point with optimizations"""
    parse_args()

    if not pre_check():
        return

    # Check frame processors
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if hasattr(frame_processor, 'pre_check') and not frame_processor.pre_check():
            return

    # Apply resource limits
    limit_resources()

    # Run processing
    if modules.globals.headless:
        if modules.globals.use_optimized:
            start_optimized()
        else:
            # Fallback to standard processing
            from modules.core import start
            start()
    else:
        # Use optimized UI
        if modules.globals.use_optimized:
            from modules.ui_optimized import create_optimized_ui
            app = create_optimized_ui()
            app.run()
        else:
            # Fallback to standard UI
            import modules.ui as ui
            window = ui.init(start_optimized if modules.globals.use_optimized else start,
                           destroy, modules.globals.lang)
            window.mainloop()


if __name__ == '__main__':
    run()