<h1 align="center">Deep-Live-Cam</h1>

<p align="center">
  Real-time face swap and video deepfake with a single click and only a single image.
</p>

<p align="center">
<a href="https://trendshift.io/repositories/11395" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11395" alt="hacksider%2FDeep-Live-Cam | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

##  Disclaimer

This deepfake software is designed to be a productive tool for the AI-generated media industry. It can assist artists in animating custom characters, creating engaging content, and even using models for clothing design.

We are aware of the potential for unethical applications and are committed to preventative measures. A built-in check prevents the program from processing inappropriate media (nudity, graphic content, sensitive material like war footage, etc.). We will continue to develop this project responsibly, adhering to the law and ethics. We may shut down the project or add watermarks if legally required.

- Ethical Use: Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online.

- Content Restrictions: The software includes built-in checks to prevent processing inappropriate media, such as nudity, graphic content, or sensitive material.

- Legal Compliance: We adhere to all relevant laws and ethical guidelines. If legally required, we may shut down the project or add watermarks to the output.

- User Responsibility: We are not responsible for end-user actions. Users must ensure their use of the software aligns with ethical standards and legal requirements.

By using this software, you agree to these terms and commit to using it in a manner that respects the rights and dignity of others.

Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online. We are not responsible for end-user actions.

## Exclusive v2.0 Quick Start - Pre-built (Windows)

  <a href="https://deeplivecam.net/index.php/quickstart"> <img src="media/Download.png" width="285" height="77" />

##### This is the fastest build you can get if you have a discrete NVIDIA or AMD GPU.
 
###### These Pre-builts are perfect for non-technical users or those who don't have time to, or can't manually install all the requirements. Just a heads-up: this is an open-source project, so you can also install it manually. This will be 60 days ahead on the open source version.

## TLDR; Live Deepfake in just 3 Clicks
![easysteps](https://github.com/user-attachments/assets/af825228-852c-411b-b787-ffd9aac72fc6)
1. Select a face
2. Select which camera to use
3. Press live!

## Features & Uses - Everything is in real-time

### Mouth Mask

**Retain your original mouth for accurate movement using Mouth Mask**

<p align="center">
  <img src="media/ludwig.gif" alt="resizable-gif">
</p>

### Face Mapping

**Use different faces on multiple subjects simultaneously**

<p align="center">
  <img src="media/streamers.gif" alt="face_mapping_source">
</p>

### Your Movie, Your Face

**Watch movies with any face in real-time**

<p align="center">
  <img src="media/movie.gif" alt="movie">
</p>

### Live Show

**Run Live shows and performances**

<p align="center">
  <img src="media/live_show.gif" alt="show">
</p>

### Memes

**Create Your Most Viral Meme Yet**

<p align="center">
  <img src="media/meme.gif" alt="show" width="450"> 
  <br>
  <sub>Created using Many Faces feature in Deep-Live-Cam</sub>
</p>

### Omegle

**Surprise people on Omegle**

<p align="center">
  <video src="https://github.com/user-attachments/assets/2e9b9b82-fa04-4b70-9f56-b1f68e7672d0" width="450" controls></video>
</p>

## Installation (Manual)

**Please be aware that the installation requires technical skills and is not for beginners. Consider downloading the prebuilt version.**

### Quick Start Guide for Mac M1/M2/M3 Users

<details>
<summary>🍎 Click here for Apple Silicon (M1/M2/M3) Quick Setup</summary>

#### Complete Setup in 5 Steps

```bash
# Step 1: Install prerequisites
brew install python@3.10 python-tk@3.10 ffmpeg cmake pkg-config

# Step 2: Clone the repository
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam

# Step 3: Set up Python environment
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Step 4: Install dependencies
pip install -r requirements.txt
pip uninstall onnxruntime onnxruntime-silicon -y
pip install onnxruntime-silicon==1.13.1

# Step 5: Download models to 'models' folder
# Download these files manually and place in the 'models' folder:
# - GFPGANv1.4.pth from: https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth
# - inswapper_128_fp16.onnx from: https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx
```

**Run the application:**
```bash
python3.10 run.py --execution-provider coreml
```

**Common Issues & Solutions:**
- **"Module _tkinter not found"**: Run `brew reinstall python-tk@3.10`
- **"Python version error"**: Ensure you're using `python3.10`, not `python` or `python3`
- **"Model not found"**: Check that both model files are in the `models` folder
- **Performance issues**: Close other apps, use ≤1080p videos, ensure good ventilation

</details>

<details>
<summary>Click to see the detailed installation process</summary>

### Installation

This is more likely to work on your computer but will be slower as it utilizes the CPU.

**1. Set up Your Platform**

-   Python (3.10 recommended)
-   pip
-   git
-   [ffmpeg](https://www.youtube.com/watch?v=OlNWCpFdVMA) - ```iex (irm ffmpeg.tc.ht)```
-   [Visual Studio 2022 Runtimes (Windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**2. Clone the Repository**

```bash
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
```

**3. Download the Models**

1. [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
2. [inswapper\_128\_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)

Place these files in the "**models**" folder.

**4. Install Dependencies**

We highly recommend using a `venv` to avoid issues.


For Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
For Linux:
```bash
# Ensure you use the installed Python 3.10
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**For macOS (Intel and Apple Silicon):**

### macOS Setup for Apple Silicon (M1/M2/M3)

Apple Silicon requires a specific setup due to ARM64 architecture. Follow these steps carefully:

#### Prerequisites

```bash
# 1. Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python 3.10 (specific version is critical for compatibility)
brew install python@3.10

# 3. Install tkinter package (required for the GUI)
brew install python-tk@3.10

# 4. Install ffmpeg (required for video processing)
brew install ffmpeg

# 5. Install additional dependencies that may be needed
brew install cmake pkg-config
```

#### Virtual Environment Setup

```bash
# Create and activate virtual environment with Python 3.10
python3.10 -m venv venv
source venv/bin/activate

# Verify you're using the correct Python version
python --version  # Should show Python 3.10.x

# Upgrade pip to the latest version
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Troubleshooting Common M1/M2/M3 Issues

1. **Python Version Conflicts:**
   ```bash
   # List all installed Python versions
   brew list | grep python
   
   # If you have conflicts, uninstall other versions (optional)
   brew uninstall --ignore-dependencies python@3.11 python@3.12 python@3.13
   
   # Ensure Python 3.10 is properly linked
   brew link python@3.10
   ```

2. **Missing tkinter Module:**
   ```bash
   # If you get "_tkinter" module not found error
   brew reinstall python-tk@3.10
   ```

3. **ARM64 vs x86_64 Architecture Issues:**
   ```bash
   # Check your architecture
   uname -m  # Should show arm64 for M1/M2/M3
   
   # If needed, install Rosetta 2 for compatibility
   softwareupdate --install-rosetta --agree-to-license
   ```

4. **Virtual Environment Issues:**
   ```bash
   # If venv creation fails, try with explicit path
   /opt/homebrew/bin/python3.10 -m venv venv
   ```

### macOS Setup for Intel-based Macs

```bash
# Install Python 3.10
brew install python@3.10

# Install tkinter and ffmpeg
brew install python-tk@3.10 ffmpeg

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

** In case something goes wrong and you need to reinstall the virtual environment **

```bash
# Deactivate the virtual environment
rm -rf venv

# Reinstall the virtual environment
python -m venv venv
source venv/bin/activate

# install the dependencies again
pip install -r requirements.txt
```

**Run:** If you don't have a GPU, you can run Deep-Live-Cam using `python run.py`. Note that initial execution will download models (~300MB).

### GPU Acceleration

**CUDA Execution Provider (Nvidia)**

1. Install [CUDA Toolkit 11.8.0](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3
```

3. Usage:

```bash
python run.py --execution-provider cuda
```

**CoreML Execution Provider (Apple Silicon M1/M2/M3)**

Apple Silicon Macs can leverage the Neural Engine for accelerated performance. Follow these steps:

1. **Prerequisites:** Complete the macOS Apple Silicon setup above using Python 3.10.

2. **Install CoreML Runtime:**
```bash
# First, ensure you're in your virtual environment
source venv/bin/activate

# Uninstall any existing onnxruntime versions
pip uninstall onnxruntime onnxruntime-silicon onnxruntime-gpu -y

# Install the Silicon-optimized version
pip install onnxruntime-silicon==1.13.1
```

3. **Download Models:** Ensure you have the required models in the `models` folder:
   - [GFPGANv1.4.pth](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
   - [inswapper_128_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)

4. **Run with CoreML Acceleration:**
```bash
# Always use python3.10 explicitly
python3.10 run.py --execution-provider coreml
```

5. **Performance Tips for M1/M2/M3:**
   - Close other resource-intensive applications
   - For best performance, use images/videos with resolution ≤ 1080p
   - The first run may be slower as models are optimized for your specific chip
   - M2 and M3 chips will see better performance than M1 due to improved Neural Engine

**Important Notes for Apple Silicon:**
- **Python Version:** You MUST use Python 3.10. Versions 3.11+ are not compatible with the current dependencies
- **Command Usage:** Always use `python3.10` instead of `python` to ensure the correct version
- **Memory:** M1 with 8GB RAM may experience limitations with high-resolution videos. M1 Pro/Max/Ultra and M2/M3 variants perform better
- **Thermal Management:** Extended use may cause thermal throttling. Consider using a laptop stand for better cooling
- **Rosetta 2:** Some dependencies may still require Rosetta 2 for x86_64 emulation:
  ```bash
  # Install if not already present
  softwareupdate --install-rosetta --agree-to-license
  ```

**CoreML Execution Provider (Apple Legacy)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-coreml
pip install onnxruntime-coreml==1.13.1
```

2. Usage:

```bash
python run.py --execution-provider coreml
```

**DirectML Execution Provider (Windows)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-directml
pip install onnxruntime-directml==1.15.1
```

2. Usage:

```bash
python run.py --execution-provider directml
```

**OpenVINO™ Execution Provider (Intel)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino==1.15.0
```

2. Usage:

```bash
python run.py --execution-provider openvino
```
</details>

## Usage

**1. Image/Video Mode**

-   Execute `python run.py`.
-   Choose a source face image and a target image/video.
-   Click "Start".
-   The output will be saved in a directory named after the target video.

**2. Webcam Mode**

-   Execute `python run.py`.
-   Select a source face image.
-   Click "Live".
-   Wait for the preview to appear (10-30 seconds).
-   Use a screen capture tool like OBS to stream.
-   To change the face, select a new source image.

## Tips and Tricks

Check out these helpful guides to get the most out of Deep-Live-Cam:

- [Unlocking the Secrets to the Perfect Deepfake Image](https://deeplivecam.net/index.php/blog/tips-and-tricks/unlocking-the-secrets-to-the-perfect-deepfake-image) - Learn how to create the best deepfake with full head coverage
- [Video Call with DeepLiveCam](https://deeplivecam.net/index.php/blog/tips-and-tricks/video-call-with-deeplivecam) - Make your meetings livelier by using DeepLiveCam with OBS and meeting software
- [Have a Special Guest!](https://deeplivecam.net/index.php/blog/tips-and-tricks/have-a-special-guest) - Tutorial on how to use face mapping to add special guests to your stream
- [Watch Deepfake Movies in Realtime](https://deeplivecam.net/index.php/blog/tips-and-tricks/watch-deepfake-movies-in-realtime) - See yourself star in any video without processing the video
- [Better Quality without Sacrificing Speed](https://deeplivecam.net/index.php/blog/tips-and-tricks/better-quality-without-sacrificing-speed) - Tips for achieving better results without impacting performance
- [Instant Vtuber!](https://deeplivecam.net/index.php/blog/tips-and-tricks/instant-vtuber) - Create a new persona/vtuber easily using Metahuman Creator

Visit our [official blog](https://deeplivecam.net/index.php/blog/tips-and-tricks) for more tips and tutorials.

## Command Line Arguments (Unmaintained)

```
options:
  -h, --help                                               show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH                     select a source image
  -t TARGET_PATH, --target TARGET_PATH                     select a target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH                     select output file or directory
  --frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]  frame processors (choices: face_swapper, face_enhancer, ...)
  --keep-fps                                               keep original fps
  --keep-audio                                             keep original audio
  --keep-frames                                            keep temporary frames
  --many-faces                                             process every face
  --map-faces                                              map source target faces
  --mouth-mask                                             mask the mouth region
  --video-encoder {libx264,libx265,libvpx-vp9}             adjust output video encoder
  --video-quality [0-51]                                   adjust output video quality
  --live-mirror                                            the live camera display as you see it in the front-facing camera frame
  --live-resizable                                         the live camera frame is resizable
  --max-memory MAX_MEMORY                                  maximum amount of RAM in GB
  --execution-provider {cpu} [{cpu} ...]                   available execution provider (choices: cpu, ...)
  --execution-threads EXECUTION_THREADS                    number of execution threads
  -v, --version                                            show program's version number and exit
```

Looking for a CLI mode? Using the -s/--source argument will make the run program in cli mode.

## Press

**We are always open to criticism and are ready to improve, that's why we didn't cherry-pick anything.**

 - [*"Deep-Live-Cam goes viral, allowing anyone to become a digital doppelganger"*](https://arstechnica.com/information-technology/2024/08/new-ai-tool-enables-real-time-face-swapping-on-webcams-raising-fraud-concerns/) - Ars Technica
 - [*"Thanks Deep Live Cam, shapeshifters are among us now"*](https://dataconomy.com/2024/08/15/what-is-deep-live-cam-github-deepfake/) - Dataconomy
 - [*"This free AI tool lets you become anyone during video-calls"*](https://www.newsbytesapp.com/news/science/deep-live-cam-ai-impersonation-tool-goes-viral/story) - NewsBytes
 - [*"OK, this viral AI live stream software is truly terrifying"*](https://www.creativebloq.com/ai/ok-this-viral-ai-live-stream-software-is-truly-terrifying) - Creative Bloq
 - [*"Deepfake AI Tool Lets You Become Anyone in a Video Call With Single Photo"*](https://petapixel.com/2024/08/14/deep-live-cam-deepfake-ai-tool-lets-you-become-anyone-in-a-video-call-with-single-photo-mark-zuckerberg-jd-vance-elon-musk/) - PetaPixel
 - [*"Deep-Live-Cam Uses AI to Transform Your Face in Real-Time, Celebrities Included"*](https://www.techeblog.com/deep-live-cam-ai-transform-face/) - TechEBlog
 - [*"An AI tool that "makes you look like anyone" during a video call is going viral online"*](https://telegrafi.com/en/a-tool-that-makes-you-look-like-anyone-during-a-video-call-is-going-viral-on-the-Internet/) - Telegrafi
 - [*"This Deepfake Tool Turning Images Into Livestreams is Topping the GitHub Charts"*](https://decrypt.co/244565/this-deepfake-tool-turning-images-into-livestreams-is-topping-the-github-charts) - Emerge
 - [*"New Real-Time Face-Swapping AI Allows Anyone to Mimic Famous Faces"*](https://www.digitalmusicnews.com/2024/08/15/face-swapping-ai-real-time-mimic/) - Digital Music News
 - [*"This real-time webcam deepfake tool raises alarms about the future of identity theft"*](https://www.diyphotography.net/this-real-time-webcam-deepfake-tool-raises-alarms-about-the-future-of-identity-theft/) - DIYPhotography
 - [*"That's Crazy, Oh God. That's Fucking Freaky Dude... That's So Wild Dude"*](https://www.youtube.com/watch?time_continue=1074&v=py4Tc-Y8BcY) - SomeOrdinaryGamers
 - [*"Alright look look look, now look chat, we can do any face we want to look like chat"*](https://www.youtube.com/live/mFsCe7AIxq8?feature=shared&t=2686) - IShowSpeed

## Credits

-   [ffmpeg](https://ffmpeg.org/): for making video-related operations easy
-   [deepinsight](https://github.com/deepinsight): for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models. Please be reminded that the [use of the model is for non-commercial research purposes only](https://github.com/deepinsight/insightface?tab=readme-ov-file#license).
-   [havok2-htwo](https://github.com/havok2-htwo): for sharing the code for webcam
-   [GosuDRM](https://github.com/GosuDRM): for the open version of roop
-   [pereiraroland26](https://github.com/pereiraroland26): Multiple faces support
-   [vic4key](https://github.com/vic4key): For supporting/contributing to this project
-   [kier007](https://github.com/kier007): for improving the user experience
-   [qitianai](https://github.com/qitianai): for multi-lingual support
-   and [all developers](https://github.com/hacksider/Deep-Live-Cam/graphs/contributors) behind libraries used in this project.
-   Footnote: Please be informed that the base author of the code is [s0md3v](https://github.com/s0md3v/roop)
-   All the wonderful users who helped make this project go viral by starring the repo ❤️

[![Stargazers](https://reporoster.com/stars/hacksider/Deep-Live-Cam)](https://github.com/hacksider/Deep-Live-Cam/stargazers)

## Contributions

![Alt](https://repobeats.axiom.co/api/embed/fec8e29c45dfdb9c5916f3a7830e1249308d20e1.svg "Repobeats analytics image")

## Stars to the Moon 🚀

<a href="https://star-history.com/#hacksider/deep-live-cam&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
 </picture>
</a>
