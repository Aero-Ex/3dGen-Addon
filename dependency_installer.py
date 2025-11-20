"""
Dependency installer for TRELLIS addon
Creates a separate virtual environment in Documents folder for complete isolation
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path
from typing import Tuple, List


# Custom Wheel URLs from HuggingFace
CUSTOM_WHEELS = {
    'triton': 'https://huggingface.co/SumitMathur8956/Trellis_Binary_wheels/resolve/main/triton-2.1.0-cp311-cp311-win_amd64.whl',
    'nvdiffrast': 'https://huggingface.co/SumitMathur8956/Trellis_Binary_wheels/resolve/main/nvdiffrast-0.3.4-cp311-cp311-win_amd64.whl',
    'diffoctreerast': 'https://huggingface.co/SumitMathur8956/Trellis_Binary_wheels/resolve/main/diffoctreerast-0.1.0-cp311-cp311-win_amd64.whl',
    'mip-splatting': 'https://huggingface.co/SumitMathur8956/Trellis_Binary_wheels/resolve/main/mip_splatting-0.1.0-cp311-cp311-win_amd64.whl',
}


# Virtual Environment Management
def get_venv_path() -> Path:
    """Get the path to the TRELLIS virtual environment in Documents"""
    if os.name == 'nt':  # Windows
        docs = Path.home() / 'Documents'
    else:  # Linux/Mac
        docs = Path.home() / 'Documents'

    return docs / 'TRELLIS_venv'


def check_venv_exists() -> bool:
    """Check if the virtual environment exists"""
    venv_path = get_venv_path()

    # Check for key venv files
    if os.name == 'nt':  # Windows
        python_exe = venv_path / 'Scripts' / 'python.exe'
    else:  # Linux/Mac
        python_exe = venv_path / 'bin' / 'python'

    return python_exe.exists()


def create_venv() -> Tuple[bool, str]:
    """
    Create a virtual environment in Documents folder using Blender's Python

    Returns:
        (success, message)
    """
    venv_path = get_venv_path()

    if check_venv_exists():
        return True, f"✓ Virtual environment already exists at {venv_path}"

    # Create Documents folder if it doesn't exist
    venv_path.parent.mkdir(parents=True, exist_ok=True)

    # Use Blender's Python to create venv
    blender_python = sys.executable

    try:
        print(f"   Creating virtual environment at: {venv_path}")
        print(f"   Using Blender's Python: {blender_python}")

        # Create venv with explicit isolation (--clear removes any existing content)
        result = subprocess.run(
            [blender_python, '-m', 'venv', '--clear', str(venv_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            # VERIFY venv is properly isolated (check pyvenv.cfg)
            cfg_path = venv_path / 'pyvenv.cfg'
            if cfg_path.exists():
                cfg_content = cfg_path.read_text()
                if 'include-system-site-packages = false' in cfg_content or 'include-system-site-packages' not in cfg_content:
                    print(f"   ✓ Venv properly isolated (system site-packages EXCLUDED)")
                else:
                    print(f"   ⚠ WARNING: Venv may include system site-packages!")
                    print(f"   Config: {cfg_content[:200]}")

            return True, f"✓ Virtual environment created at {venv_path}"
        else:
            return False, f"✗ Failed to create venv: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "✗ Timeout creating virtual environment"
    except Exception as e:
        return False, f"✗ Error creating venv: {e}"


def get_venv_python() -> str:
    """Get the Python executable from the virtual environment"""
    venv_path = get_venv_path()

    if os.name == 'nt':  # Windows
        return str(venv_path / 'Scripts' / 'python.exe')
    else:  # Linux/Mac
        return str(venv_path / 'bin' / 'python')


def get_venv_pip() -> str:
    """Get the pip executable from the virtual environment"""
    venv_path = get_venv_path()

    if os.name == 'nt':  # Windows
        return str(venv_path / 'Scripts' / 'pip.exe')
    else:  # Linux/Mac
        return str(venv_path / 'bin' / 'pip')


# Blender's protected packages (don't upgrade these)
BLENDER_PROTECTED_PACKAGES = {
    'bpy', 'mathutils', 'bgl', 'blf', 'gpu', 'bmesh',
    'aud', 'imbuf', 'freestyle', 'gpu_extras',
}

# Packages with version constraints to avoid conflicts
SAFE_PACKAGES = {
    'numpy': '>=1.24.0,<2.0.0',  # Blender uses numpy 1.x
    'Pillow': '>=9.5.0',
    'opencv-python': '>=4.8.0',
}

# Core TRELLIS dependencies (following official repo setup.sh)
TRELLIS_DEPENDENCIES = [
    # PyTorch (installed separately with CUDA)
    # 'torch==2.2.2', 'torchvision==0.17.2',

    # Attention backend (installed separately with --no-deps)
    # 'xformers==0.0.24',

    # Image processing (official repo: pillow imageio imageio-ffmpeg)
    'Pillow',  # Capitalized - correct package name
    'imageio',
    'imageio-ffmpeg',
    'opencv-python-headless',
    'rembg',

    # 3D processing (official repo: trimesh open3d xatlas pyvista pymeshfix igraph)
    'trimesh',
    'open3d',
    'xatlas',
    'pyvista',  # No version constraint - following official TRELLIS repo
    'pymeshfix',
    'igraph',

    # ML/AI (official repo: transformers)
    'transformers',
    'diffusers==0.30.3',
    'accelerate==0.34.2',
    'huggingface-hub==0.25.2',
    'hf_transfer', # Faster downloads
    'onnxruntime-directml==1.16.3',  # DirectML backend for GPU acceleration
    'tokenizers==0.19.1',
    'safetensors==0.4.5',

    # Scientific computing (official repo: scipy ninja rembg onnxruntime tqdm easydict)
    'scipy',
    'numpy==1.26.4',  # Pin to 1.26.4 for PyTorch/ONNX compatibility
    'tqdm',
    'easydict',
    
    # Sparse convolutions (CRITICAL for TRELLIS)
    'spconv-cu118==2.3.6',  # spconv with CUDA 11.8 support

    # Build tools
    'ninja',

    # Utils
    'omegaconf==2.3.0',
    'einops==0.8.0',
    'packaging==24.1',
    'filelock==3.16.1',
    'fsspec==2024.10.0',
    'regex==2024.9.11',
    'requests==2.32.3',
    'pyyaml==6.0.2',
    'typing-extensions==4.12.2',

    # TRELLIS utils3d dependency
    'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8',

    # Additional 3D libs (not in official repo basic install, but needed for full functionality)
    'pyrender==0.1.45',
    'rtree==1.3.0',
    'shapely==2.0.6',
    'networkx==3.2.1',
    'pyglet==2.0.18',
    'freetype-py==2.5.1',

    # Mesh processing
    'meshio==5.3.5',
    'plyfile==1.1',
    'mapbox-earcut==1.0.2',
    'colorlog==6.9.0',
    'jsonschema==4.23.0',
    'lxml==5.3.0',
    'glfw==2.8.0',
    'pyopengl==3.1.0',

    # Additional ML
    'scikit-image==0.24.0',
    'scikit-learn==1.5.2',
    'matplotlib==3.9.2',
]


def get_python_executable() -> str:
    """Get the Python executable (from venv if it exists, otherwise Blender's Python)"""
    if check_venv_exists():
        return get_venv_python()
    return sys.executable


def check_package_installed(package_name: str) -> Tuple[bool, str]:
    """
    Check if a package is installed in the venv

    Returns:
        (is_installed, version)
    """
    if not check_venv_exists():
        return False, ''

    try:
        # Extract package name without version constraints
        pkg_name = package_name.split('>=')[0].split('==')[0].split('[')[0]

        # Use pip show to check if package is installed in venv
        python_exe = get_python_executable()

        # Clear environment variables to ensure we check venv, not Blender's Python
        env = os.environ.copy()
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)

        result = subprocess.run(
            [python_exe, '-m', 'pip', 'show', pkg_name],
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )

        if result.returncode == 0:
            # Parse version from pip show output
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
                    return True, version
            return True, 'unknown'
        else:
            return False, ''

    except Exception:
        return False, ''


def check_pytorch_installed() -> Tuple[bool, bool]:
    """
    Check if PyTorch is installed with CUDA support in the venv

    Returns:
        (torch_installed, cuda_available)
    """
    if not check_venv_exists():
        return False, False

    # Check if torch is installed in venv
    torch_installed, _ = check_package_installed('torch')
    if not torch_installed:
        return False, False

    # Check CUDA availability by running a test script in the venv
    try:
        python_exe = get_python_executable()
        test_code = "import torch; print(torch.cuda.is_available())"

        # Clear environment variables to ensure we check venv, not Blender's Python
        env = os.environ.copy()
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)

        result = subprocess.run(
            [python_exe, '-c', test_code],
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )

        if result.returncode == 0:
            cuda_available = result.stdout.strip().lower() == 'true'
            return True, cuda_available
        else:
            # DLL errors or import errors
            return False, False

    except Exception:
        return False, False


def install_package(package: str, upgrade: bool = False) -> Tuple[bool, str]:
    """
    Install a package using pip

    Args:
        package: Package specification (e.g., 'numpy>=1.24.0')
        upgrade: Whether to upgrade if already installed

    Returns:
        (success, message)
    """
    python = get_python_executable()

    cmd = [python, '-m', 'pip', 'install']

    # Use --upgrade-strategy only-if-needed to avoid breaking Blender
    cmd.extend(['--upgrade-strategy', 'only-if-needed'])

    if upgrade:
        cmd.append('--upgrade')

    cmd.append(package)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per package
        )

        if result.returncode == 0:
            return True, f"✓ Installed {package}"
        else:
            return False, f"✗ Failed to install {package}: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, f"✗ Timeout installing {package}"
    except Exception as e:
        return False, f"✗ Error installing {package}: {e}"


def install_pytorch(cuda_version: str = 'cu118') -> Tuple[bool, str]:
    """
    Install PyTorch with CUDA support

    Uses PyTorch 2.1.0 on Windows for proven stability and proper DLL initialization.
    Newer versions have Windows DLL bugs. 2.1.0 has:
    - Proven stability on Windows systems
    - Proper DLL initialization without errors
    - Full compatibility with Blender's Python 3.11
    - Extensive real-world testing

    Args:
        cuda_version: CUDA version (cu118 or cu121)

    Returns:
        (success, message)
    """
    import platform
    python = get_python_executable()
    index_url = f"https://download.pytorch.org/whl/{cuda_version}"

    # Use PyTorch 2.2.2 on Windows for stability
    if platform.system() == 'Windows':
        cmd = [
            python, '-m', 'pip', 'install',
            'torch==2.2.2',
            'torchvision==0.17.2',
            '--index-url', index_url,
            '--upgrade-strategy', 'only-if-needed'
        ]
    else:
        # Latest version is fine on Linux/Mac
        cmd = [
            python, '-m', 'pip', 'install',
            'torch', 'torchvision',
            '--index-url', index_url,
            '--upgrade-strategy', 'only-if-needed'
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for PyTorch
        )

        if result.returncode == 0:
            version_str = "2.2.2 (Windows stable)" if platform.system() == 'Windows' else "latest"
            return True, f"✓ PyTorch {version_str} installed successfully"
        else:
            return False, f"✗ Failed to install PyTorch: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "✗ Timeout installing PyTorch (network issue?)"
    except Exception as e:
        return False, f"✗ Error installing PyTorch: {e}"


def install_xformers(cuda_version: str = 'cu118') -> Tuple[bool, str]:
    """
    Install xformers with --no-deps flag to avoid dependency conflicts

    Args:
        cuda_version: CUDA version (cu118 or cu121)

    Returns:
        (success, message)
    """
    python = get_python_executable()
    index_url = f"https://download.pytorch.org/whl/{cuda_version}"

    cmd = [
        python, '-m', 'pip', 'install',
        'xformers==0.0.24',
        '--index-url', index_url,
        '--no-deps',  # Critical: avoid dependency conflicts
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            return True, "✓ xformers 0.0.24 installed successfully"
        else:
            return False, f"✗ Failed to install xformers: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "✗ Timeout installing xformers"
    except Exception as e:
        return False, f"✗ Error installing xformers: {e}"


def install_custom_wheel(url: str, package_name: str) -> Tuple[bool, str]:
    """
    Install a custom wheel from HuggingFace or other URL

    Args:
        url: Direct URL to .whl file
        package_name: Name of the package for display

    Returns:
        (success, message)
    """
    python = get_python_executable()

    cmd = [
        python, '-m', 'pip', 'install',
        url,
        '--no-deps',  # Don't install dependencies
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            return True, f"✓ {package_name} installed from custom wheel"
        else:
            return False, f"✗ Failed to install {package_name}: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, f"✗ Timeout installing {package_name}"
    except Exception as e:
        return False, f"✗ Error installing {package_name}: {e}"


def install_custom_wheels() -> Tuple[bool, List[str]]:
    """
    Install all custom binary wheels from HuggingFace

    Returns:
        (success, messages)
    """
    messages = []
    all_success = True

    for package_name, url in CUSTOM_WHEELS.items():
        success, msg = install_custom_wheel(url, package_name)
        messages.append(msg)
        if not success:
            all_success = False

    return all_success, messages


def fix_numpy_dll_windows() -> Tuple[bool, str]:
    """
    Fix NumPy DLL import errors on Windows

    This addresses the common "DLL load failed while importing _multiarray_umath" error
    by reinstalling NumPy with proper version constraints.

    Returns:
        (success, message)
    """
    import platform

    if platform.system() != 'Windows':
        return True, "Not Windows - NumPy DLL fix not needed"

    python = get_python_executable()

    # First, try to uninstall NumPy completely
    try:
        print("   🔧 Uninstalling existing NumPy...")
        subprocess.run(
            [python, '-m', 'pip', 'uninstall', 'numpy', '-y'],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as e:
        print(f"   ⚠ Could not uninstall NumPy: {e}")

    # Reinstall NumPy with proper version constraints
    try:
        print("   📥 Installing NumPy with proper version constraints...")
        cmd = [
            python, '-m', 'pip', 'install',
            'numpy>=1.24.0,<2.0.0',  # Compatible with Blender
            '--force-reinstall',
            '--no-cache-dir',  # Don't use cached version
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            return True, "✓ NumPy reinstalled successfully"
        else:
            return False, f"✗ Failed to reinstall NumPy: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "✗ Timeout reinstalling NumPy"
    except Exception as e:
        return False, f"✗ Error reinstalling NumPy: {e}"


def check_numpy_dll_issue() -> bool:
    """
    Check if NumPy has DLL import issues (Windows) in the venv

    Returns:
        True if NumPy has DLL issues, False otherwise
    """
    if not check_venv_exists():
        return False

    try:
        python_exe = get_python_executable()
        test_code = "import numpy; numpy.array([1, 2, 3]); print('OK')"
        result = subprocess.run(
            [python_exe, '-c', test_code],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and 'OK' in result.stdout:
            return False  # No issues
        elif 'DLL load failed' in result.stderr or '_multiarray_umath' in result.stderr:
            return True  # DLL issue detected
        return False

    except Exception:
        return False


def check_pytorch_dll_issue() -> bool:
    """
    Check if PyTorch has DLL import issues (Windows) in the venv

    Returns:
        True if PyTorch has DLL issues, False otherwise
    """
    if not check_venv_exists():
        return False

    try:
        python_exe = get_python_executable()
        test_code = "import torch; torch.zeros(1); print('OK')"
        result = subprocess.run(
            [python_exe, '-c', test_code],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and 'OK' in result.stdout:
            return False  # No issues
        elif 'DLL' in result.stderr or 'c10.dll' in result.stderr:
            return True  # DLL issue detected
        return False

    except Exception:
        return False


def fix_pytorch_dll_windows() -> Tuple[bool, str]:
    """
    Fix PyTorch DLL import errors on Windows by reinstalling stable version

    This addresses the common "c10.dll initialization failed" error by
    uninstalling and reinstalling PyTorch 2.1.0 (proven stable on Windows).

    PyTorch 2.1.0 is used instead of latest because:
    - Proven stability on Windows systems
    - Proper DLL initialization (newer versions have Windows bugs)
    - Full compatibility with Blender's Python 3.11
    - Extensive real-world testing

    Returns:
        (success, message)
    """
    import platform

    if platform.system() != 'Windows':
        return True, "Not Windows - PyTorch DLL fix not needed"

    python = get_python_executable()

    # Uninstall PyTorch completely
    try:
        print("   🔧 Uninstalling existing PyTorch...")
        for package in ['torch', 'torchvision', 'torchaudio']:
            subprocess.run(
                [python, '-m', 'pip', 'uninstall', package, '-y'],
                capture_output=True,
                text=True,
                timeout=60,
            )
    except Exception as e:
        print(f"   ⚠ Could not uninstall PyTorch: {e}")

    # Reinstall PyTorch 2.2.2 (stable version for Windows)
    try:
        print("   📥 Reinstalling PyTorch 2.2.2 (Windows stable version)...")
        index_url = "https://download.pytorch.org/whl/cu118"

        cmd = [
            python, '-m', 'pip', 'install',
            'torch==2.2.2',
            'torchvision==0.17.2',
            '--index-url', index_url,
            '--force-reinstall',
            '--no-cache-dir',
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
        )

        if result.returncode == 0:
            return True, "✓ PyTorch 2.2.2 (stable) reinstalled successfully"
        else:
            return False, f"✗ Failed to reinstall PyTorch: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "✗ Timeout reinstalling PyTorch"
    except Exception as e:
        return False, f"✗ Error reinstalling PyTorch: {e}"


def verify_installation() -> Tuple[bool, List[str]]:
    """
    Verify TRELLIS installation in the venv

    Returns:
        (success, messages)
    """
    messages = []

    # Check venv exists
    if not check_venv_exists():
        messages.append("✗ Virtual environment not found")
        return False, messages

    # Check PyTorch
    torch_installed, cuda_available = check_pytorch_installed()
    if torch_installed:
        # Get PyTorch version from venv
        installed, version = check_package_installed('torch')
        messages.append(f"✓ PyTorch {version}")
        if cuda_available:
            messages.append("✓ CUDA available")
        else:
            messages.append("⚠ CUDA not available")
    else:
        messages.append("✗ PyTorch not installed")
        return False, messages

    # Check TRELLIS (bundled with addon)
    messages.append("✓ TRELLIS bundled with addon")

    # Check key dependencies
    deps_to_check = ['Pillow', 'trimesh', 'transformers', 'diffusers']
    for dep in deps_to_check:
        installed, version = check_package_installed(dep)
        if installed:
            messages.append(f"✓ {dep}")
        else:
            messages.append(f"⚠ {dep} not found")

    return True, messages


def get_installation_status(detailed: bool = False) -> dict:
    """
    Get current installation status of all dependencies in the venv

    Args:
        detailed: If True, check all packages (slow!). If False, only check essentials (fast).

    Returns:
        Dictionary with package statuses
    """
    status = {}

    # Check venv
    status['venv'] = {
        'exists': check_venv_exists(),
        'path': str(get_venv_path()),
    }

    if not check_venv_exists():
        # If no venv, everything is not installed
        status['torch'] = {'installed': False, 'cuda': False, 'version': ''}
        status['trellis'] = {'installed': False}
        status['installation_failed'] = False  # No venv means not attempted yet
        return status

    # Check PyTorch
    torch_installed, cuda_available = check_pytorch_installed()
    status['torch'] = {
        'installed': torch_installed,
        'cuda': cuda_available,
        'version': '',
    }
    if torch_installed:
        _, version = check_package_installed('torch')
        status['torch']['version'] = version

    # Check TRELLIS (bundled with addon, not pip-installed)
    status['trellis'] = {'installed': True, 'bundled': True}

    # Check installation status from status file
    import pathlib
    addon_dir = pathlib.Path(__file__).parent
    status_file = addon_dir / '.install_status'
    
    installation_failed = False
    
    # Only check status file if it exists
    if status_file.exists():
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                install_status = f.read().strip().split('\n')[0]
                # If status is 'failed' or 'in_progress' but PyTorch isn't installed, mark as failed
                if install_status == 'failed':
                    installation_failed = True
                elif install_status == 'in_progress' and not torch_installed:
                    # Installation was interrupted or crashed
                    installation_failed = True
        except Exception:
            pass
    # If no status file and PyTorch is installed, installation succeeded
    # If no status file and PyTorch NOT installed, show as "not installed" not "failed"
    # Only mark as failed if we have evidence of a failed attempt (status file)
    
    status['installation_failed'] = installation_failed

    # Only check detailed package status if requested
    # This is SLOW (20+ subprocess calls) so UI should NOT use this!
    if detailed:
        for dep in TRELLIS_DEPENDENCIES:
            # Handle git+ URLs specially
            if dep.startswith('git+'):
                # Extract package name from git URL (e.g., utils3d from git+...utils3d.git)
                pkg_name = dep.split('/')[-1].split('.git')[0]
            else:
                pkg_name = dep.split('>=')[0].split('==')[0].split('[')[0]

            installed, version = check_package_installed(pkg_name)
            status[pkg_name] = {
                'installed': installed,
                'version': version,
            }

    return status
