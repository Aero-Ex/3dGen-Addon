# 3D-Gen - Advanced 3D Generation for Blender

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Blender](https://img.shields.io/badge/Blender-4.5+-orange.svg)
![Python](https://img.shields.io/badge/Python-3.11-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

**3D-Gen** is a powerful Blender addon that brings state-of-the-art AI-powered 3D generation directly into your workflow. Generate high-quality 3D models from text descriptions, images, or multiple images using the TRELLIS architecture.

## ✨ Features

- **🎨 Text-to-3D Generation**: Create 3D models from natural language descriptions
- **🖼️ Image-to-3D Generation**: Convert single images into full 3D meshes
- **📸 Multi-Image-to-3D**: Generate models from multiple reference images
- **⚡ GPU Acceleration**: CUDA-optimized for NVIDIA GPUs
- **🔧 Advanced Options**: 
  - Customizable generation parameters (seed, sampling steps, guidance scale)
  - Mesh simplification with adjustable target face count
  - Texture baking with configurable resolution and iterations
- **💾 Export Options**: GLB format with embedded textures
- **🚀 Automatic Setup**: One-click installation of all dependencies
- **📊 Real-time Progress**: Visual feedback during generation process

## 🎯 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (6GB+ VRAM recommended)
  - Tested on: RTX 4050 Laptop GPU
  - Minimum: GTX 1060 6GB or equivalent
- **RAM**: 16GB+ recommended
- **Storage**: 16GB+ free space for dependencies

### Software
- **Blender**: 4.5 or newer
- **Operating System**: Windows 10/11 (Primary support)
- **Internet Connection**: Required for initial setup

## 📦 Installation

### Method 1: Manual Installation (Recommended)

1. **Download the addon**
   ```
   Clone or download this repository
   ```

2. **Install in Blender**
   - Open Blender
   - Go to `Edit > Preferences > Add-ons`
   - Click `Install...`
   - Navigate to the downloaded folder and select the addon folder
   - Enable the addon by checking the box next to "3D-Gen"

3. **Initial Setup**
   - Find the addon in the sidebar: `View3D > Sidebar > 3D-Gen`
   - Click `Install Dependencies` (one-time setup, takes 5-10 minutes)
   - Wait for installation to complete (50 packages will be installed)

### Method 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/Aero-Ex/3d_Gen-Addon.git

# Link to Blender addons folder
mklink /D "C:\Users\YourUsername\AppData\Roaming\Blender Foundation\Blender\4.5\scripts\addons\3d_Gen-Addon" "path\to\cloned\repo"
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

## ⚙️ Configuration

### Generation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **Seed** | Random seed for reproducible results | 0 | 0 - 2147483647 |
| **Sparse Structure Steps** | Detail level for initial structure | 12 | 1 - 20 |
| **SLAT Steps** | Refinement iterations | 12 | 1 - 20 |
| **Sparse Structure CFG** | Guidance scale for structure | 7.5 | 1.0 - 15.0 |
| **SLAT CFG** | Guidance scale for refinement | 3.0 | 1.0 - 10.0 |

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
- **Solution**: Ensure stable internet connection and sufficient disk space
- Check console logs in Blender for specific errors

**Problem**: CUDA not detected
- **Solution**: Update NVIDIA drivers to latest version
- Verify CUDA is available: `nvidia-smi` in terminal

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

### Known Issues

- **spconv cumm DLL warning**: Harmless warning that doesn't affect generation
- **UI lag on first open**: Path setup runs once per session (normal behavior)

## 📊 Performance Benchmarks

Tested on **RTX 4050 Laptop GPU** (6GB VRAM):

| Operation | Time | Notes |
|-----------|------|-------|
| First Generation | 1m 30s | Includes model loading |
| Subsequent Generations | 1m 10s | Models cached |
| Sparse Structure (12 steps) | ~8s | Initial structure |
| SLAT Generation (12 steps) | ~6s | Refinement |
| Mesh Decimation | ~2s | 542K → 50K faces |
| Texture Baking (1000 iter) | ~45s | 1024x1024 |

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
- **PyTorch Team**: For the deep learning framework
- **Blender Foundation**: For the amazing 3D creation suite

## 📮 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Aero-Ex/3d_Gen-Addon/issues)
- **Documentation**: Check this README for common questions
- **Author**: AeroX

## 🔗 Links

- **GitHub Repository**: [https://github.com/Aero-Ex/3d_Gen-Addon](https://github.com/Aero-Ex/3d_Gen-Addon)
- **TRELLIS Paper**: [Microsoft Research](https://github.com/microsoft/TRELLIS)
- **Blender**: [https://www.blender.org/](https://www.blender.org/)

---

**Made with ❤️ by AeroX**

*Last Updated: November 18, 2025*
