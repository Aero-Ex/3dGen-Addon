"""
Installation script that runs in a separate console window
Shows all installation output with professional formatting
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Import professional logging system
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from console_logger import ConsoleLogger

# Initialize professional logger
logger = ConsoleLogger("TRELLIS_install")


def format_cuda_variant_label(cuda_variant: str) -> str:
    """Convert cuXYZ style variant strings to human-friendly CUDA labels."""
    if cuda_variant.startswith("cu") and len(cuda_variant) > 3:
        digits = cuda_variant[2:]
        if digits.isdigit():
            major = digits[:-1] or digits
            minor = digits[-1]
            return f"CUDA {major}.{minor}"
    return cuda_variant.upper()


def write_install_status(status: str, error_msg: str = ""):
    """Write installation status to file for UI to read
    
    Args:
        status: 'in_progress', 'success', or 'failed'
        error_msg: Optional error message if status is 'failed'
    """
    try:
        status_file = script_dir / '.install_status'
        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(f"{status}\n")
            if error_msg:
                f.write(f"{error_msg}\n")
    except Exception as e:
        logger.warning(f"Could not write status file: {e}")


def check_trellis_path(path):
    """Check if TRELLIS path is valid"""
    if not path or not os.path.exists(path):
        return False, "Path does not exist"

    # Check for trellis directory (main package)
    trellis_dir = os.path.join(path, 'trellis')
    if not os.path.exists(trellis_dir):
        return False, "Missing trellis/ directory"

    if not os.path.isdir(trellis_dir):
        return False, "trellis/ is not a directory"

    # Check for __init__.py to confirm it's a valid Python package
    init_file = os.path.join(trellis_dir, '__init__.py')
    if not os.path.exists(init_file):
        return False, "Missing trellis/__init__.py"

    # Check for key subdirectories
    required_dirs = ['pipelines', 'models', 'utils']
    for dir_name in required_dirs:
        dir_path = os.path.join(trellis_dir, dir_name)
        if not os.path.exists(dir_path):
            return False, f"Missing trellis/{dir_name}/"

    return True, "Valid TRELLIS repository"


def install_package(python_exe, package, show_output=True):
    """Install a package and show live output or capture error details"""
    # CRITICAL: Clear environment variables that might interfere with venv
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)  # Remove PYTHONHOME - causes venv issues
    env.pop('PYTHONPATH', None)  # Remove PYTHONPATH - causes venv issues

    cmd = [
        python_exe, '-m', 'pip', 'install',
        package,
        '--isolated',  # Ignore environment variables and user configuration
        '--no-user',   # Don't do user install, force into venv
        '--prefer-binary',  # Prefer binary wheels over source
        '--timeout', '600',  # 10 minute timeout for large packages
    ]

    if show_output:
        # Show live output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env  # Use cleaned environment
        )

        for line in process.stdout:
            print(f"    {line}", end='')

        process.wait()
        return (process.returncode == 0, None)
    else:
        # Silent install but capture errors
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            if result.returncode != 0:
                # Extract meaningful error from stderr
                error_lines = result.stderr.strip().split('\n')
                # Get last few lines which usually contain the actual error
                error_msg = '\n'.join(error_lines[-5:]) if error_lines else "Unknown error"
                return (False, error_msg)
            return (True, None)
        except subprocess.TimeoutExpired:
            return (False, "Installation timed out after 10 minutes")
        except Exception as e:
            return (False, str(e))


def main():
    """Main installation workflow"""
    logger.header("TRELLIS Dependency Installer")
    
    # Mark installation as in progress (clears any previous failure)
    write_install_status('in_progress')
    
    logger.instructions([
        "This process will take 10-30 minutes",
        "Please keep this window open",
        "You can minimize Blender while this runs"
    ])
    
    logger.env_info()
    
    logger.divider()

    start_time = time.time()

    # Create virtual environment first
    logger.section("Setting Up Virtual Environment", icon="📦")

    # Import dependency installer to use venv functions
    python_exe = sys.executable
    script_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(script_dir))

    try:
        from . import dependency_installer
        logger.success("Loaded dependency_installer module")
    except ImportError as e:
        try:
            import dependency_installer
            logger.success("Loaded dependency_installer module (direct import)")
        except ImportError as e2:
            logger.error("Failed to import dependency_installer!")
            logger.plain(f"Error: {e2}", indent=1)
            logger.plain(f"Script dir: {script_dir}", indent=1)
            logger.info(f"Log saved to: {logger.log_file}")
            input("\nPress Enter to exit...")
            return 1

    torch_version = getattr(dependency_installer, "TORCH_VERSION", "2.6.0")
    torchvision_version = getattr(dependency_installer, "TORCHVISION_VERSION", "0.21.0")
    torchaudio_version = getattr(dependency_installer, "TORCHAUDIO_VERSION", "2.6.0")
    torch_index_url = getattr(dependency_installer, "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu124")
    cuda_variant = getattr(dependency_installer, "CUDA_VARIANT", "cu124")
    cuda_label = format_cuda_variant_label(cuda_variant)
    trellis_dependencies = tuple(getattr(dependency_installer, "TRELLIS_DEPENDENCIES", ()))
    xformers_version = getattr(dependency_installer, "XFORMERS_VERSION", "0.0.29.post3")
    triton_version = getattr(dependency_installer, "TRITON_VERSION", "3.1.0")
    triton_windows_version = getattr(dependency_installer, "TRITON_WINDOWS_VERSION", "3.1.0.post17")

    venv_path = dependency_installer.get_venv_path()
    logger.info(f"Location: {venv_path}", indent=1)

    if dependency_installer.check_venv_exists():
        logger.success("Virtual environment already exists", indent=1)
        venv_python = dependency_installer.get_venv_python()
        logger.info(f"Using venv Python: {venv_python}", indent=1)
    else:
        logger.info("Creating new virtual environment...", indent=1)
        success, msg = dependency_installer.create_venv()
        if success:
            logger.success(msg, indent=1)
        else:
            logger.error(msg, indent=1)
            input("\nPress Enter to exit...")
            return 1
        venv_python = dependency_installer.get_venv_python()
        logger.info(f"Venv Python: {venv_python}", indent=1)

    # Use venv Python from now on
    python_exe = venv_python

    logger.success("TRELLIS package bundled with addon")

    # Check current installations IN THE VENV (not Blender's Python!)
    logger.section("Checking Current Status", icon="🔍")
    
    try:
        # Check PyTorch in the VENV, not in Blender's Python
        result = subprocess.run(
            [python_exe, '-c', 'import torch; print(torch.__version__)'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            if '+cpu' in version:
                logger.warning(f"PyTorch {version} (CPU-only - will be replaced)", indent=1)
            else:
                logger.success(f"PyTorch {version} installed", indent=1)
            torch_installed = True
        else:
            logger.info("PyTorch not installed", indent=1)
            torch_installed = False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        logger.info("PyTorch not installed", indent=1)
        torch_installed = False

    # Check for DLL issues on Windows
    import platform
    if platform.system() == 'Windows':
        logger.subsection("Checking PyTorch DLL Status")
        has_pytorch_dll_issue = False
        try:
            # Test PyTorch in the VENV, not in Blender's Python
            result = subprocess.run(
                [python_exe, '-c', 'import torch; torch.zeros(1); print(torch.__version__)'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                if '+cpu' in version:
                    logger.warning(f"PyTorch {version} working (CPU-only - will be replaced)", indent=1)
                else:
                    logger.success(f"PyTorch {version} working correctly", indent=1)
            else:
                # Check stderr for DLL errors
                stderr = result.stderr
                if 'DLL' in stderr or 'c10.dll' in stderr:
                    logger.error("PyTorch DLL issues detected (c10.dll error)!", indent=1)
                    logger.plain(f"Error: {stderr[:200]}", indent=2)
                    has_pytorch_dll_issue = True
                else:
                    logger.info("PyTorch not installed yet (will install fresh)", indent=1)
        except subprocess.TimeoutExpired:
            logger.warning("PyTorch test timed out", indent=1)
        except Exception as e:
            logger.warning(f"Could not check PyTorch: {e}", indent=1)

        # Fix PyTorch DLL issues
        if has_pytorch_dll_issue:
            logger.section("Fixing PyTorch DLL Issues", icon="🔧")

            # CRITICAL: Clear environment variables
            env = os.environ.copy()
            env.pop('PYTHONHOME', None)
            env.pop('PYTHONPATH', None)

            logger.info("Uninstalling existing PyTorch completely...", indent=1)
            for package in ['torch', 'torchvision', 'torchaudio']:
                subprocess.run(
                    [python_exe, '-m', 'pip', 'uninstall', package, '-y'],
                    capture_output=True,
                    env=env
                )

            logger.info(f"Reinstalling PyTorch {torch_version} ({cuda_label} build)...", indent=1)
            logger.info("Using unified pipeline-compatible build", indent=2)
            
            cmd = [
                python_exe, '-m', 'pip', 'install',
                f'torch=={torch_version}',
                f'torchvision=={torchvision_version}',
                f'torchaudio=={torchaudio_version}',
                '--index-url', torch_index_url,
                '--isolated',  # Ignore environment variables and user configuration
                '--no-user',   # Don't do user install, force into venv
                '--force-reinstall',
                '--no-cache-dir'
            ]

            result = subprocess.run(cmd, capture_output=False, text=True, env=env)

            if result.returncode == 0:
                logger.success(f"PyTorch {torch_version} fixed successfully!", indent=1)
                torch_installed = True
            else:
                logger.warning("PyTorch fix may have issues", indent=1)

        # Check NumPy DLL issues IN THE VENV
        logger.subsection("Checking NumPy DLL Status")
        has_numpy_dll_issue = False
        try:
            # Test NumPy in the VENV, not in Blender's Python
            result = subprocess.run(
                [python_exe, '-c', 'import numpy; numpy.array([1,2,3]); print(numpy.__version__)'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.success(f"NumPy {version} working correctly")
            else:
                stderr = result.stderr
                if 'DLL load failed' in stderr or '_multiarray_umath' in stderr:
                    logger.error("NumPy DLL issues detected!")
                    logger.plain(f"Error output: {stderr[:200]}", indent=1)
                    has_numpy_dll_issue = True
                else:
                    logger.info("NumPy not installed yet (will install fresh)")
        except subprocess.TimeoutExpired:
            logger.warning("NumPy test timed out")
        except Exception as e:
            logger.warning(f"Could not check NumPy: {e}")

        # Fix NumPy DLL issues
        if has_numpy_dll_issue:
            logger.section("🔧 Fixing NumPy DLL Issues")

            # CRITICAL: Clear environment variables
            env = os.environ.copy()
            env.pop('PYTHONHOME', None)
            env.pop('PYTHONPATH', None)

            logger.info("Uninstalling existing NumPy...", indent=1)
            subprocess.run(
                [python_exe, '-m', 'pip', 'uninstall', 'numpy', '-y'],
                capture_output=True,
                env=env
            )

            logger.info("Reinstalling NumPy with proper version constraints...", indent=1)
            cmd = [
                python_exe, '-m', 'pip', 'install',
                'numpy==1.26.4',
                '--isolated',  # Ignore environment variables and user configuration
                '--no-user',   # Don't do user install, force into venv
                '--force-reinstall',
                '--no-cache-dir'
            ]

            result = subprocess.run(cmd, capture_output=False, text=True, env=env)

            if result.returncode == 0:
                logger.success("NumPy fixed successfully!", indent=1)
            else:
                logger.warning("NumPy fix may have issues", indent=1)
    
    # CRITICAL: Force NumPy 1.26.4 ALWAYS (current PyTorch stack still depends on NumPy 1.x)
    logger.subsection("Ensuring NumPy 1.26.4 compatibility")
    logger.info(f"PyTorch {torch_version} requires NumPy 1.x (not 2.x)", indent=1)
    try:
        # Check current NumPy version
        result = subprocess.run(
            [python_exe, '-c', 'import numpy; print(numpy.__version__)'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            current_numpy = result.stdout.strip()
            if current_numpy.startswith('2.'):
                logger.warning(f"Found NumPy {current_numpy} (incompatible!)", indent=1)
                logger.info("Downgrading to NumPy 1.26.4...", indent=2)
                
                env = os.environ.copy()
                env.pop('PYTHONHOME', None)
                env.pop('PYTHONPATH', None)
                
                subprocess.run(
                    [python_exe, '-m', 'pip', 'install', 'numpy==1.26.4', '--force-reinstall', '--no-cache-dir'],
                    capture_output=True,
                    env=env
                )
                logger.success("NumPy 1.26.4 installed!", indent=2)
            else:
                logger.success(f"NumPy {current_numpy} (compatible)", indent=1)
    except Exception as e:
        logger.warning(f"Could not check NumPy: {e}", indent=1)

    # Install PyTorch with CUDA support
    logger.section(f"Installing PyTorch with {cuda_label}", icon="🔥")

    # CRITICAL: Clear environment variables for all pip operations
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)

    # Check if we need to reinstall (wrong version or CPU-only)
    needs_reinstall = False
    if torch_installed:
        try:
            # Check in VENV, not Blender's Python
            result = subprocess.run(
                [python_exe, '-c', 'import torch; print(torch.__version__)'],
                capture_output=True,
                text=True,
                timeout=5,
                env=env
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                is_cpu = '+cpu' in version

                if is_cpu:
                    logger.warning(f"Found CPU-only PyTorch {version}", indent=1)
                    logger.info("Reinstalling with CUDA support...", indent=2)
                    needs_reinstall = True

                    # Uninstall CPU version
                    logger.info("Removing CPU version...", indent=1)
                    subprocess.run(
                        [python_exe, '-m', 'pip', 'uninstall', 'torch', 'torchvision', '-y'],
                        capture_output=True,
                        env=env
                    )
                else:
                    logger.success(f"PyTorch {version} with CUDA already installed", indent=1)
            else:
                needs_reinstall = True
        except Exception as e:
            logger.warning(f"Could not check PyTorch version: {e}")
            needs_reinstall = True
        pip_show_result = subprocess.run(
            [python_exe, '-m', 'pip', '--version'],
            capture_output=True,
            text=True,
            timeout=5,
            env=env
        )
        if pip_show_result.returncode == 0:
            logger.plain(f"pip info: {pip_show_result.stdout.strip()}", indent=2)
            if 'TRELLIS_venv' in pip_show_result.stdout:
                logger.success("pip is using venv!", indent=2)
            elif 'blender' in pip_show_result.stdout.lower():
                logger.warning("pip might be using Blender's Python!", indent=2)

        # Show site-packages location
        site_result = subprocess.run(
            [python_exe, '-c', 'import site; print(site.getsitepackages())'],
            capture_output=True,
            text=True,
            timeout=5,
            env=env
        )
        if site_result.returncode == 0:
            logger.plain(f"site-packages: {site_result.stdout.strip()}", indent=2)
    except Exception as e:
        logger.warning(f"Could not verify pip: {e}", indent=2)

    # Install PyTorch stack with aggressive isolation to ensure it goes into venv
    cmd = [
        python_exe, '-m', 'pip', 'install',
        f'torch=={torch_version}',
        f'torchvision=={torchvision_version}',
        f'torchaudio=={torchaudio_version}',
        '--index-url', torch_index_url,
        '--isolated',  # Ignore environment variables and user configuration
        '--no-user',   # Don't do user install, force into venv
        '--force-reinstall',  # Force install even if Blender has it
        '--no-cache-dir'  # Don't use cache
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env  # Use cleaned environment
    )

    for line in process.stdout:
        print(f"    {line}", end='')

    process.wait()
    success = process.returncode == 0

    if success:
        logger.success(f"PyTorch {torch_version} with {cuda_label} installed!")

        # CRITICAL: Verify CUDA actually works IN THE VENV!
        logger.info("Verifying CUDA support...", indent=1)
        try:
            # Check CUDA in the VENV, not Blender's Python
            result = subprocess.run(
                [python_exe, '-c', '''
import torch
print(f"VERSION:{torch.__version__}")
print(f"CUDA:{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"DEVICES:{torch.cuda.device_count()}")
    print(f"GPU:{torch.cuda.get_device_name(0)}")
'''],
                capture_output=True,
                text=True,
                timeout=15,
                env=env
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                # Safely parse output
                version = "unknown"
                cuda_available = False
                device_count = "0"
                device_name = "Unknown"

                for line in output.splitlines():
                    if line.startswith('VERSION:'):
                        version = line.split(':', 1)[1].strip()
                    elif line.startswith('CUDA:'):
                        cuda_available = line.split(':', 1)[1].strip() == 'True'
                    elif line.startswith('DEVICES:'):
                        device_count = line.split(':', 1)[1].strip()
                    elif line.startswith('GPU:'):
                        device_name = line.split(':', 1)[1].strip()

                if cuda_available:
                    logger.success("CUDA is working!")
                    logger.plain(f"PyTorch: {version}", indent=1)
                    logger.plain(f"GPU: {device_name}", indent=1)
                    logger.plain(f"Devices: {device_count}", indent=1)
                else:
                    logger.warning("PyTorch installed but CUDA NOT AVAILABLE!")
                    logger.plain(f"PyTorch version: {version}", indent=1)
                    logger.plain("This means you'll be using CPU-only (10-50x slower!)", indent=1)
                    logger.plain("")
                    logger.plain("Possible causes:", indent=1)
                    logger.plain("- No NVIDIA GPU detected", indent=2)
                    logger.plain("- CUDA drivers not installed", indent=2)
                    logger.plain(f"- GPU driver too old (need 550+ for {cuda_label})", indent=2)
                    logger.plain("")
                    logger.warning("Continuing anyway, but generation will be VERY slow...")
            else:
                stderr = result.stderr.strip()
                logger.warning("CUDA verification failed!")
                if stderr:
                    logger.plain(f"Error: {stderr[:200]}", indent=1)

                # Check if this is a DLL error
                if 'DLL' in stderr or 'torch_cuda.dll' in stderr or 'c10.dll' in stderr or 'WinError 127' in stderr or 'WinError 126' in stderr:
                    logger.error(f"═══ DLL ERROR DETECTED in PyTorch {torch_version} ═══")
                    logger.warning(f"PyTorch {torch_version} has DLL dependency issues on your system.")
                    logger.plain("")
                    logger.plain("This is a critical error. Installation cannot continue.", indent=1)
                    logger.plain("")
                    logger.plain("Possible solutions:", indent=1)
                    logger.plain("1. Update your NVIDIA GPU drivers", indent=2)
                    logger.plain("2. Install/Reinstall Visual C++ Redistributable (2015-2022)", indent=2)
                    logger.plain("3. Check Windows Update for system updates", indent=2)
                    logger.plain(f"4. Install {cuda_label} Toolkit manually", indent=2)
                    logger.plain(f"Error details: {stderr[:300]}", indent=1)
        except Exception as e:
            logger.warning(f"Could not verify CUDA: {e}")
            logger.plain("Continuing anyway...", indent=1)
    else:
        logger.error("PyTorch installation failed!")
        logger.plain("\nPress Enter to exit...", indent=0)
        input()
        return 1

    # Install xformers (needed for sparse attention)
    logger.section("📦 Installing xformers (attention backend)")

    logger.info(f"Installing xformers=={xformers_version} from PyTorch index...")
    logger.plain("(This provides sparse attention operations)", indent=1)

    cmd = [
        python_exe, '-m', 'pip', 'install',
        f'xformers=={xformers_version}',
        '--no-deps',
        '--index-url', torch_index_url,
        '--isolated',
        '--no-user',
        '--no-cache-dir'
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    for line in process.stdout:
        print(f"    {line}", end='')

    process.wait()
    xformers_success = process.returncode == 0

    if not xformers_success:
        logger.warning("xformers install via PyTorch index failed")
        logger.plain("Retrying from PyPI (may take longer)...", indent=1)
        fallback_cmd = [
            python_exe, '-m', 'pip', 'install',
            f'xformers=={xformers_version}',
            '--no-deps',
            '--isolated',
            '--no-user',
            '--no-cache-dir',
            '--prefer-binary'
        ]
        fallback_proc = subprocess.Popen(
            fallback_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        for line in fallback_proc.stdout:
            print(f"    {line}", end='')
        fallback_proc.wait()
        xformers_success = fallback_proc.returncode == 0

    if xformers_success:
        logger.success("xformers installed!")
    else:
        logger.warning("xformers installation failed")
        logger.plain("This may cause issues with sparse attention operations.", indent=1)
        logger.plain("Continuing anyway...", indent=1)

    # Install triton compiler directly from PyTorch index
    logger.section("📦 Installing Triton compiler")
    triton_pkg = "triton-windows" if platform.system() == 'Windows' else "triton"
    triton_display_version = triton_windows_version if triton_pkg == "triton-windows" else triton_version
    logger.info(f"Installing {triton_pkg}=={triton_display_version}...", indent=1)
    triton_ok, triton_msg = dependency_installer.install_triton()
    if triton_ok:
        logger.success(triton_msg, indent=1)
    else:
        logger.warning("triton installation failed", indent=1)
        logger.plain(triton_msg, indent=2)
        logger.plain("Continuing without Triton may break attention kernels.", indent=2)

    # Install custom binary wheels from HuggingFace
    logger.section("Installing Custom Binary Wheels", icon="🔩")

    logger.info(
        "Installing flash-attn, gsplat, torchsparse, spconv, nvdiffrast, diffoctreerast, mip-splatting from HuggingFace...",
        indent=1
    )
    custom_success, custom_messages = dependency_installer.install_custom_wheels()
    for msg in custom_messages:
        if "✓" in msg:
            logger.success(msg.replace("✓ ", ""), indent=1)
        else:
            logger.error(msg.replace("✗ ", ""), indent=1)

    if custom_success:
        logger.success("All custom wheels installed!")
    else:
        logger.warning("Some custom wheels failed")
        logger.plain("Continuing anyway...", indent=1)

    # Install other dependencies
    logger.section("Installing Other Dependencies", icon="📦")

    # CRITICAL: Install NumPy FIRST to prevent scipy from upgrading it to 2.x
    logger.info("Pre-installing NumPy 1.26.4 (prevent scipy from upgrading to 2.x)...")
    
    cmd_numpy = [
        python_exe, '-m', 'pip', 'install',
        'numpy==1.26.4',
        '--isolated',
        '--no-user',
        '--prefer-binary',
        '--force-reinstall',
        '--no-cache-dir'
    ]
    
    process = subprocess.Popen(
        cmd_numpy,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    for line in process.stdout:
        print(f"    {line}", end='')
    
    process.wait()
    numpy_preinstall_success = process.returncode == 0
    
    if numpy_preinstall_success:
        logger.success("NumPy 1.26.4 locked!")
    else:
        logger.warning("NumPy pre-installation had issues, continuing...")

    # Filter out xformers and numpy since we already installed them
    dependencies_to_install = [
        dep for dep in trellis_dependencies
        if not dep.startswith('xformers') and not dep.startswith('numpy')
    ]

    # Check which packages are already installed to skip them
    logger.info("Checking already installed packages...", indent=1)
    already_installed = []
    for dep in dependencies_to_install[:]:  # Copy list to modify during iteration
        pkg_name = dep.split('>=')[0].split('==')[0].split('[')[0]
        try:
            result = subprocess.run(
                [python_exe, '-c', f'import {pkg_name}'],
                capture_output=True,
                timeout=2,
                env=env
            )
            if result.returncode == 0:
                already_installed.append(pkg_name)
                dependencies_to_install.remove(dep)
        except:
            pass  # Package not installed or import failed
    
    if already_installed:
        logger.success(f"Skipping {len(already_installed)} already installed packages", indent=1)
        logger.plain(f"({', '.join(already_installed[:5])}{'...' if len(already_installed) > 5 else ''})", indent=2)

    if not dependencies_to_install:
        logger.success("All dependencies already installed!")
        logger.plain("Nothing to install", indent=1)
    else:
        # Group packages: install small packages in batches for speed
        logger.info(f"Installing {len(dependencies_to_install)} packages...")
        logger.plain("(Using batch installation for speed)", indent=1)

    failed = []
    failed_details = {}
    
    if not dependencies_to_install:
        # Nothing to install, skip to verification
        pass
    else:
        # Install all packages in one pip command for speed
        logger.step(1, 1, f"Installing all packages in batch")
        
        # Create pip command with all packages as separate arguments
        # CRITICAL: Use --upgrade-strategy only-if-needed to prevent NumPy from being upgraded to 2.x
        cmd = [python_exe, '-m', 'pip', 'install'] + dependencies_to_install + [
            '--isolated',
            '--no-user',
            '--prefer-binary',
            '--upgrade-strategy', 'only-if-needed',  # Prevent NumPy 2.x upgrade!
            '--timeout', '600'  # 10 minute timeout for large packages
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        for line in process.stdout:
            print(f"    {line}", end='')
        
        process.wait()
        success = process.returncode == 0
        error_msg = None if success else "Batch installation failed"
    
        if not success:
            # If batch install fails, fall back to individual installation
            logger.warning("Batch installation failed, trying individual packages...")
            
            for i, dep in enumerate(dependencies_to_install, 1):
                # Extract package name for display
                if dep.startswith('git+'):
                    pkg_name = dep.split('/')[-1].split('.git')[0]
                    display_name = f"{pkg_name} (from git)"
                else:
                    pkg_name = dep.split('>=')[0].split('==')[0].split('[')[0]
                    display_name = pkg_name

                logger.step(i, len(dependencies_to_install), f"Installing {display_name}")
                success, error_msg = install_package(python_exe, dep, show_output=True)

                if success:
                    logger.success(display_name, indent=1)
                else:
                    logger.error(f"{display_name} (failed)", indent=1)
                    if error_msg:
                        # Show first line of error immediately
                        error_first_line = error_msg.split('\n')[0]
                        logger.plain(f"Error: {error_first_line}", indent=2)
                    failed.append(display_name)
                    failed_details[display_name] = error_msg
        else:
            logger.success("All packages installed successfully!")

    logger.plain(f"\nInstalled {len(trellis_dependencies) - len(failed)}/{len(trellis_dependencies)} packages")

    if failed:
        logger.divider()
        logger.warning("⚠ Some packages failed to install:")
        for pkg in failed:
            logger.plain(f"• {pkg}", indent=1)
            if pkg in failed_details and failed_details[pkg]:
                # Show detailed error for each failed package
                error_lines = failed_details[pkg].split('\n')
                for line in error_lines[:3]:  # Show first 3 lines
                    if line.strip():
                        logger.plain(f"  {line}", indent=2)
        logger.divider()
        logger.plain(f"Full error details saved to: {logger.log_file}", indent=1)
        logger.plain("Common causes: network timeout, missing build tools, incompatible versions", indent=1)

    # Check bundled TRELLIS
    logger.section("Verifying Bundled TRELLIS Package", icon="🎯")

    # Find bundled TRELLIS in addon directory
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_trellis_path = os.path.join(addon_dir, 'trellis')

    if os.path.exists(bundled_trellis_path):
        logger.info(f"Location: {bundled_trellis_path}", indent=1)

        # Check for key TRELLIS files
        init_file = os.path.join(bundled_trellis_path, '__init__.py')
        pipelines_dir = os.path.join(bundled_trellis_path, 'pipelines')

        if os.path.exists(init_file) and os.path.exists(pipelines_dir):
            logger.success("TRELLIS package is bundled and ready!", indent=1)
            logger.plain("(TRELLIS will be imported via addon's sys.path setup)", indent=2)
        else:
            logger.warning("TRELLIS package structure incomplete", indent=1)
    else:
        logger.error(f"Bundled TRELLIS not found at: {bundled_trellis_path}", indent=1)
        logger.plain("Addon may not work correctly!", indent=2)

    # Verify IN THE VENV (not Blender's Python!)
    logger.section("Verification", icon="🔍")

    # Create clean environment for verification
    verify_env = os.environ.copy()
    verify_env.pop('PYTHONHOME', None)
    verify_env.pop('PYTHONPATH', None)

    # CRITICAL: Verify NumPy version (must be 1.26.4, NOT 2.x!)
    try:
        result = subprocess.run(
            [python_exe, '-c', 'import numpy; print(numpy.__version__)'],
            capture_output=True,
            text=True,
            timeout=10,
            env=verify_env
        )
        if result.returncode == 0:
            numpy_version = result.stdout.strip()
            if numpy_version.startswith('2.'):
                logger.error(f"NumPy {numpy_version} detected (INCOMPATIBLE!)")
                logger.warning("Fixing NumPy version...")
                # Force downgrade to 1.26.4
                subprocess.run(
                    [python_exe, '-m', 'pip', 'install', 'numpy==1.26.4', '--force-reinstall', '--no-cache-dir'],
                    capture_output=True,
                    timeout=120,
                    env=verify_env
                )
                logger.success("NumPy downgraded to 1.26.4")
            else:
                logger.success(f"NumPy {numpy_version} (compatible)")
        else:
            logger.warning("Could not verify NumPy version")
    except Exception as e:
        logger.warning(f"NumPy verification failed: {e}")

    # Check PyTorch in the VENV
    try:
        result = subprocess.run(
            [python_exe, '-c', '''
import torch
print(f"VERSION:{torch.__version__}")
print(f"CUDA:{torch.cuda.is_available()}")
'''],
            capture_output=True,
            text=True,
            timeout=10,
            env=verify_env
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            # Safely parse output
            version = "unknown"
            cuda_available = "unknown"
            for line in output.splitlines():
                if line.startswith('VERSION:'):
                    version = line.split(':', 1)[1].strip()
                elif line.startswith('CUDA:'):
                    cuda_available = line.split(':', 1)[1].strip()

            logger.success(f"PyTorch {version}")
            logger.success(f"CUDA available: {cuda_available}")
        else:
            stderr = result.stderr.strip()
            logger.error("PyTorch verification failed")
            if stderr:
                logger.plain(f"Error: {stderr[:200]}", indent=1)
    except Exception as e:
        logger.error(f"PyTorch verification failed: {e}")

    # Check TRELLIS in the VENV
    # Add addon directory to PYTHONPATH so venv can find bundled TRELLIS
    trellis_verify_env = verify_env.copy()
    trellis_verify_env['PYTHONPATH'] = addon_dir

    try:
        result = subprocess.run(
            [python_exe, '-c', '''
import sys
sys.path.insert(0, r"''' + addon_dir + '''")
from trellis.pipelines import TrellisImageTo3DPipeline
print("OK")
'''],
            capture_output=True,
            text=True,
            timeout=60,  # 60s timeout - first import loads models
            env=trellis_verify_env
        )
        if result.returncode == 0 and 'OK' in result.stdout:
            logger.success("TRELLIS imports successfully", indent=1)
        else:
            stderr = result.stderr.strip()
            logger.error("TRELLIS import failed", indent=1)
            if stderr:
                # Show first error line only
                first_error = stderr.split('\n')[-1] if stderr else "Unknown error"
                logger.plain(f"Error: {first_error}", indent=2)
    except Exception as e:
        logger.error(f"TRELLIS import failed: {e}", indent=1)

    # Summary
    logger.summary(
        success=len(failed) == 0,
        message=f"{len(trellis_dependencies) - len(failed)}/{len(trellis_dependencies)} packages installed"
    )

    if failed:
        logger.warning("Some packages failed:")
        for pkg in failed:
            logger.plain(f"- {pkg}", indent=1)

    logger.box([
        "Return to Blender",
        "Click 'Initialize TRELLIS' in the addon panel",
        "Start generating 3D models!"
    ], title="Next Steps")

    # Write final status
    if failed:
        write_install_status('failed', f"{len(failed)} packages failed to install")
    else:
        # Delete status file on success - no status means "installed successfully"
        try:
            status_file = script_dir / '.install_status'
            if status_file.exists():
                status_file.unlink()
        except Exception as e:
            logger.warning(f"Could not remove status file: {e}")

    input("\nPress Enter to close this window...")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        write_install_status('failed', 'Installation cancelled by user')
        logger.warning("\nInstallation cancelled by user")
        input("\nPress Enter to exit...")
        sys.exit(1)
    except Exception as e:
        import traceback

        error_msg = f"Fatal error: {str(e)}"
        write_install_status('failed', error_msg)
        
        logger.error(error_msg)
        logger.plain("\nFull traceback:", indent=1)
        tb_str = traceback.format_exc()
        for line in tb_str.splitlines():
            logger.plain(line, indent=2)
        
        logger.divider("=")
        logger.info(f"Error details saved to: {logger.log_file}")
        logger.plain("Please share this log file with the developer!")
        logger.divider("=")
        
        input("\nPress Enter to exit...")
        sys.exit(1)
