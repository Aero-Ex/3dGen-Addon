# 3D-Gen - Advanced 3D Generation for Blender

![Version](https://img.shields.io/badge/version-2.0.0--exp-blue.svg)
![Blender](https://img.shields.io/badge/Blender-4.5+-orange.svg)
![Python](https://img.shields.io/badge/Python-3.11-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

**3D-Gen** is a powerful Blender addon that brings state-of-the-art AI-powered 3D generation directly into your workflow. Generate high-quality 3D models from text descriptions, images, or multiple images using **TRELLIS** and **Direct3D-S2** architectures.

> **⚡ Experimental Branch**: This branch includes the Direct3D-S2 pipeline for high-resolution mesh refinement and upscaling!

## ✨ Features

### Core Features
- **🎨 Text-to-3D Generation**: Create 3D models from natural language descriptions (TRELLIS)
- **🖼️ Image-to-3D Generation**: Convert single images into full 3D meshes (TRELLIS & Direct3D-S2)
- **📸 Multi-Image-to-3D**: Generate models from multiple reference images (TRELLIS)
- **🔥 NEW: Direct3D-S2 Pipeline**: High-resolution mesh refinement and upscaling
  - 512³ resolution support (works on 6GB VRAM)
  - 1024³ resolution support (requires 12GB+ VRAM)
  - Mesh upscaling from existing models
  - Advanced sparse attention optimization

### Advanced Options
- **⚡ GPU Acceleration**: CUDA-optimized for NVIDIA GPUs (RTX 20/30/40 series)
- **🔧 Customizable Parameters**:
  - Seed control for reproducible results
  - Diffusion steps (5-30, affects quality vs. speed)
  - Guidance scale (3-12, controls adherence to input)
  - Mesh simplification with adjustable ratios
- **💾 Export Options**: GLB, OBJ, PLY formats with embedded textures
- **🚀 Automatic Setup**: One-click installation of all dependencies
- **📊 Real-time Progress**: Visual feedback during generation process
- **🖥️ Console Scripts**: Standalone generation and upscaling without Blender UI

## 🎯 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support
  - **For 512³ Direct3D-S2**: 6GB+ VRAM (RTX 3060, RTX 4050, etc.)
  - **For 1024³ Direct3D-S2**: 12GB+ VRAM (RTX 3080, RTX 4070+, etc.)
  - **TRELLIS Only**: 6GB+ VRAM sufficient
  - Tested on: RTX 4050 Laptop GPU (6GB)
  - Supported: RTX 20/30/40 series
- **RAM**: 16GB+ recommended (32GB for 1024³)
- **Storage**: 15GB+ free space for dependencies and model weights

### Software
- **Blender**: 4.5 or newer
- **Operating System**: Windows 10/11 (Primary support)
- **Internet Connection**: Required for initial setup
- **C Compiler** (for Direct3D-S2):
  - **Option 1**: Use pre-compiled Triton cache (RTX 40 series only, included in repo)
  - **Option 2**: Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    - Required for Triton kernel compilation
    - One-time ~5 minute install
    - RTX 30 series users must install this

## 📦 Installation

### Method 1: Clone Experimental Branch (Recommended)

1. **Download the addon**
   ```bash
   # Clone the experimental branch with Direct3D-S2
   git clone -b 3dGen-Addon_Exp https://github.com/Aero-Ex/3dGen-Addon.git
   ```
   Or download ZIP: [3dGen-Addon_Exp branch](https://github.com/Aero-Ex/3dGen-Addon/tree/3dGen-Addon_Exp)

2. **Install in Blender**
   - Open Blender
   - Go to `Edit > Preferences > Add-ons`
   - Click `Install...`
   - Navigate to the downloaded folder and select the addon folder
   - Enable the addon by checking the box next to "3D-Gen"

3. **Install MSVC (RTX 30 series users only)**
   - RTX 40 series can skip this (pre-compiled cache included)
   - Download [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Install with "Desktop development with C++" workload
   - Restart your computer

4. **Initial Setup**
   - Find the addon in the sidebar: `View3D > Sidebar > 3D-Gen`
   - Click `Install Dependencies` (one-time setup, takes 10-15 minutes)
   - Wait for installation to complete (60+ packages will be installed)
   - First Direct3D-S2 run may compile Triton kernels (1-2 minutes, RTX 30 users)

  #### Custom wheel mirror

  The installer downloads CUDA-specific wheels (flash-attn, gsplat, nvdiffrast, etc.) from HuggingFace. By default it points to `https://huggingface.co/SumitMathur8956/myaddon_cuda14/resolve/main`. To use your own mirror, set the environment variable before launching Blender/Install Dependencies:

  ```powershell
  setx TRELLIS_WHEELS_BASE_URL "https://huggingface.co/<user>/<repo>/resolve/main"
  ```

  Place the `.whl` files inside that repo (or in `Documents\wheels`) and the installer will pull them automatically.

### Method 2: Development Installation

```bash
# Clone the experimental branch
git clone -b 3dGen-Addon_Exp https://github.com/Aero-Ex/3dGen-Addon.git

# Link to Blender addons folder (Windows)
mklink /D "C:\Users\YourUsername\AppData\Roaming\Blender Foundation\Blender\4.5\scripts\addons\3dGen-Addon" "path\to\cloned\repo"

# Or copy directly
xcopy /E /I "path\to\cloned\repo" "C:\Users\YourUsername\AppData\Roaming\Blender Foundation\Blender\4.5\scripts\addons\3dGen-Addon"
```

## 🚀 Quick Start

### Text-to-3D Generation

1. Open the **3D-Gen** panel in the 3D Viewport sidebar
2. Select **Text to 3D** mode
3. Enter your prompt (e.g., "a wooden chair", "futuristic robot")
4. Click **Generate 3D Model**
5. Wait for generation to complete (~1-2 minutes)
6. Your model will be imported into the scene automatically

### Image-to-3D Generation

1. Select **Image to 3D** mode
2. Click the folder icon to select an input image
3. (Optional) Adjust generation parameters
4. Click **Generate 3D Model**
5. The generated model will appear in your scene

### Multi-Image Generation

1. Select **Multi-Image to 3D** mode
2. Add multiple reference images using the list interface
3. Configure parameters as needed
4. Click **Generate 3D Model**

### Direct3D-S2 Generation (Experimental)

1. Select **Direct3D-S2** mode in the pipeline dropdown
2. Select an input image
3. Choose resolution:
   - **512³**: Works on 6GB VRAM (recommended for RTX 3060/4050)
   - **1024³**: Requires 12GB+ VRAM (RTX 3080/4070+)
4. Adjust refinement parameters:
   - **Steps** (5-30): Lower preserves input, higher refines more
   - **Guidance** (3-12): Lower preserves mesh, higher follows image
5. Click **Generate 3D Model**
6. Model will be generated at high resolution with sparse optimization

### Upscaling Existing Meshes (Console)

For upscaling existing 3D models outside of Blender:

```bash
# Navigate to addon directory
cd "C:\Users\YourUsername\AppData\Roaming\Blender Foundation\Blender\4.5\scripts\addons\3dGen-Addon"

# Upscale a mesh with Direct3D-S2
python upscale_in_console.py input_mesh.obj reference_image.png --resolution 512

# With custom parameters
python upscale_in_console.py mesh.obj image.png --resolution 512 --steps 15 --guidance 7.0 --seed 42

# Preserve mesh structure (low steps, low guidance)
python upscale_in_console.py mesh.obj image.png --steps 8 --guidance 4.0

# High quality refinement
python upscale_in_console.py mesh.obj image.png --steps 25 --guidance 10.0 --resolution 1024
```

## ⚙️ Configuration

### TRELLIS Generation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **Seed** | Random seed for reproducible results | 0 | 0 - 2147483647 |
| **Sparse Structure Steps** | Detail level for initial structure | 12 | 1 - 20 |
| **SLAT Steps** | Refinement iterations | 12 | 1 - 20 |
| **Sparse Structure CFG** | Guidance scale for structure | 7.5 | 1.0 - 15.0 |
| **SLAT CFG** | Guidance scale for refinement | 3.0 | 1.0 - 10.0 |

### Direct3D-S2 Parameters

| Parameter | Description | Default | Range | Notes |
|-----------|-------------|---------|-------|-------|
| **Resolution** | Output resolution | 512³ | 512³, 1024³ | 512³=6GB VRAM, 1024³=12GB+ |
| **Steps** | Diffusion sampling steps | 15 | 5 - 30 | Lower=faster/preserve, Higher=quality |
| **Guidance Scale** | Adherence to reference image | 7.0 | 3.0 - 12.0 | Lower=preserve mesh, Higher=follow image |
| **Seed** | Random seed | 42 | -1 to 2³¹ | -1 for random |
| **Simplify Ratio** | Mesh simplification | 0.95 | 0.1 - 1.0 | 1.0=no simplification |

### Post-Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| **Simplify Mesh** | Reduce polygon count | Enabled |
| **Target Faces** | Face count after simplification | 50,000 |
| **Bake Texture** | Optimize and bake textures | Enabled |
| **Texture Resolution** | Output texture size | 1024 |
| **Baking Iterations** | Quality vs. speed tradeoff | 1000 |

## 📂 Output

Generated models are saved to:
```
C:\Users\YourUsername\Documents\TRELLIS_Output\
```

Files are named with timestamp:
```
trellis_YYYYMMDD_HHMMSS.glb
```

## 🔧 Advanced Usage

### Custom Virtual Environment

The addon creates a virtual environment at:
```
C:\Users\YourUsername\Documents\TRELLIS_venv\
```

You can manually manage this environment if needed:
```powershell
# Activate the environment
C:\Users\YourUsername\Documents\TRELLIS_venv\Scripts\Activate.ps1

# Update packages
python -m pip install --upgrade package-name

# Check installed packages
python -m pip list
```

### Dependency Management

Installed packages include:
- **PyTorch 2.2.2** (CUDA 11.8)
- **xformers 0.0.24**
- **NumPy 1.26.4** (locked for compatibility)
- **trimesh**, **open3d**, **pyvista** (mesh processing)
- **transformers**, **diffusers** (AI models)
- **kaolin**, **spconv-cu118** (3D operations)
- And 40+ other supporting libraries

## 🐛 Troubleshooting

### Installation Issues

**Problem**: Dependencies fail to install
- **Solution**: Ensure stable internet connection and sufficient disk space (15GB+)
- Check console logs in Blender for specific errors
- Try deleting `C:\Users\YourUsername\Documents\TRELLIS_venv` and reinstalling

**Problem**: CUDA not detected
- **Solution**: Update NVIDIA drivers to latest version
- Verify CUDA is available: `nvidia-smi` in terminal
- Ensure you have an NVIDIA GPU (AMD/Intel not supported)

**Problem**: "Failed to find C compiler" (Direct3D-S2)
- **Solution for RTX 40 series**: Pre-compiled cache should work automatically
- **Solution for RTX 30 series**: Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Select "Desktop development with C++" workload
  - Restart computer after installation
- **Verification**: First Direct3D-S2 run will compile Triton kernels (1-2 min wait is normal)

### Generation Issues

**Problem**: "Out of memory" errors
- **Solution**: 
  - Reduce sampling steps (try 8 instead of 12)
  - Close other GPU-intensive applications
  - Lower texture resolution

**Problem**: Generated model looks incorrect
- **Solution**:
  - Try different seeds (randomize or set specific values)
  - Adjust guidance scales (higher = more adherence to prompt)
  - Use more descriptive prompts

**Problem**: Slow generation
- **Solution**:
  - First generation is slower (model loading)
  - Subsequent generations are faster (~1 minute)
  - Reduce baking iterations for faster texture processing
  - Direct3D-S2 at 512³: ~8-10 seconds per generation
  - Direct3D-S2 at 1024³: ~30-40 seconds per generation

**Problem**: Direct3D-S2 generation fails or looks wrong
- **Solution**:
  - Ensure input image has clear subject on white/transparent background
  - Try different seeds (randomize or set specific values)
  - Adjust guidance: Lower (4-5) preserves mesh, Higher (8-10) follows image more
  - Start with 512³ before trying 1024³
  - Check VRAM usage (6GB minimum for 512³)

### Known Issues

- **spconv cumm DLL warning**: Harmless warning that doesn't affect generation
- **UI lag on first open**: Path setup runs once per session (normal behavior)
- **Triton compilation on first run**: RTX 30 users will see kernel compilation messages (normal, ~1-2 min)
- **Console output during generation**: Direct3D-S2 prints progress to console (expected behavior)

## 📊 Performance Benchmarks

Tested on **RTX 4050 Laptop GPU** (6GB VRAM):

### TRELLIS Pipeline

| Operation | Time | Notes |
|-----------|------|-------|
| First Generation | 1m 30s | Includes model loading |
| Subsequent Generations | 1m 10s | Models cached |
| Sparse Structure (12 steps) | ~8s | Initial structure |
| SLAT Generation (12 steps) | ~6s | Refinement |
| Mesh Decimation | ~2s | 542K → 50K faces |
| Texture Baking (1000 iter) | ~45s | 1024x1024 |

### Direct3D-S2 Pipeline

| Operation | Time (512³) | Time (1024³) | VRAM Used | Notes |
|-----------|-------------|--------------|-----------|-------|
| First Generation | ~1m 30s | N/A | ~5.8GB | Includes loading + Triton compilation (RTX 30) |
| Subsequent (15 steps) | ~8-10s | ~30-40s | ~5.5GB / ~10GB | Cached models |
| Dense Sampling (15 steps) | ~8s | ~25s | ~3GB / ~8GB | Image encoding + diffusion |
| Sparse Sampling (15 steps) | ~2-3s | ~8-10s | ~5GB / ~10GB | Sparse attention optimization |
| Mesh Post-processing | ~1-2s | ~3-5s | ~2GB | Remeshing + simplification |
| **Total (with remesh)** | **~10-12s** | **~35-45s** | **~5.8GB** | After first run |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Aero-Ex/3d_Gen-Addon.git

# Create development branch
git checkout -b feature/your-feature-name

# Make your changes
# Test thoroughly in Blender

# Submit pull request
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft TRELLIS**: Based on the TRELLIS architecture ([GitHub](https://github.com/microsoft/TRELLIS))
- **Direct3D-S2**: High-resolution 3D generation pipeline ([Research](https://github.com/VAST-AI-Research/Direct3D-S2))
- **OpenAI Triton**: GPU kernel optimization framework
- **PyTorch Team**: For the deep learning framework
- **Blender Foundation**: For the amazing 3D creation suite
- **Community Contributors**: For testing, feedback, and improvements

## 📮 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Aero-Ex/3d_Gen-Addon/issues)
- **Documentation**: Check this README for common questions
- **Author**: AeroX

## 🔗 Links

- **GitHub Repository**: [https://github.com/Aero-Ex/3dGen-Addon](https://github.com/Aero-Ex/3dGen-Addon)
- **Experimental Branch**: [3dGen-Addon_Exp](https://github.com/Aero-Ex/3dGen-Addon/tree/3dGen-Addon_Exp)
- **TRELLIS**: [Microsoft Research](https://github.com/microsoft/TRELLIS)
- **Direct3D-S2**: [VAST AI Research](https://github.com/VAST-AI-Research/Direct3D-S2)
- **Blender**: [https://www.blender.org/](https://www.blender.org/)

---

**Made with ❤️ by AeroX**

*Experimental Branch - Direct3D-S2 Integration*
*Last Updated: November 22, 2025*
