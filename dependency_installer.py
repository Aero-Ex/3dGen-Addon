"""Centralised dependency management for the 3D-Gen addon.

This module owns the logic for preparing a dedicated virtual environment,
installing the CUDA 12.4 / PyTorch 2.6.0 stack, downloading custom wheels,
and providing health checks that the UI and console installers can reuse.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
TORCHAUDIO_VERSION = "2.6.0"
CUDA_VARIANT = "cu124"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
TRITON_VERSION = "3.1.0"
TRITON_WINDOWS_VERSION = "3.1.0.post17"
XFORMERS_VERSION = "0.0.29.post3"

DEFAULT_WHEELS_BASE_URL = (
    "https://huggingface.co/SumitMathur8956/myaddon_cuda14/resolve/main"
)
WHEELS_BASE_URL = os.environ.get("TRELLIS_WHEELS_BASE_URL", DEFAULT_WHEELS_BASE_URL)

TRELLIS_DEPENDENCIES: Sequence[str] = (
    "Pillow>=10.4.0",
    "imageio>=2.36.0",
    "imageio-ffmpeg>=0.5.1",
    "opencv-python-headless==4.10.0.84",
    "rembg==2.0.56",
    "trimesh>=4.4.3",
    "open3d==0.18.0",
    "xatlas==0.0.9",
    "pyvista==0.43.10",
    "pymeshfix==0.17.0",
    "igraph==0.11.5",
    "transformers==4.40.2",
    "diffusers==0.30.3",
    "accelerate==0.34.2",
    "huggingface-hub[torch]==0.25.2",
    "onnxruntime-directml==1.19.0",
    "tokenizers==0.19.1",
    "safetensors==0.4.5",
    "scipy==1.13.1",
    "numpy==1.26.4",
    "tqdm==4.66.5",
    "easydict==1.11",
    "ninja>=1.11.1",
    "omegaconf==2.3.0",
    "einops==0.8.0",
    "packaging==24.1",
    "filelock==3.16.1",
    "fsspec==2024.10.0",
    "regex==2024.9.11",
    "requests==2.32.3",
    "pyyaml==6.0.2",
    "typing-extensions==4.12.2",
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
    "pyrender==0.1.45",
    "rtree==1.3.0",
    "shapely==2.0.6",
    "networkx==3.2.1",
    "pyglet==2.0.18",
    "freetype-py==2.5.1",
    "meshio==5.3.5",
    "plyfile==1.1",
    "mapbox-earcut==1.0.2",
    "colorlog==6.9.0",
    "jsonschema==4.23.0",
    "lxml==5.3.0",
    "glfw==2.8.0",
    "pyopengl==3.1.0",
    "scikit-image==0.24.0",
    "scikit-learn==1.5.2",
    "matplotlib==3.9.2",
    "hydra-core==1.3.2",
    "lightning==2.5.0",
    "tensorboardX==2.6",
    "torchmetrics==1.5.1",
    "roma==1.5.0",
    "jaxtyping==0.2.30",
    "lpips==0.1.4",
    "colorspacious==1.1.2",
    "moviepy==1.0.3",
    "pillow-heif==1.1.1",
    "viser==0.1.29",
    "gradio==4.44.0",
    "tyro==0.8.6",
    "pycolmap==3.10.0",
    "h5py==3.11.0",
    "rootutils",
    "rootpath",
    "backports.cached-property",
    "rich>=13.3.5",
    "rootutils==1.0.7",
    "tensorboard==2.17.0",
    "pre-commit==3.8.0",
    "pytest==8.3.2",
    "typeguard<3",
    "wheel",
    "spconv-cu124",
    "setuptools",
)

CUSTOM_WHEELS: Sequence[Dict[str, str]] = (
    {
        "name": "flash-attn",
        "filename": "flash_attn-2.7.4+cu124torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl",
    },
    {
        "name": "nvdiffrast",
        "filename": "nvdiffrast-0.3.4-cp311-cp311-win_amd64.whl",
        "pip_fallback": "nvdiffrast==0.3.4",
    },
    {
        "name": "diffoctreerast",
        "filename": "diffoctreerast-0.1.0-cp311-cp311-win_amd64.whl",
    },
    {
        "name": "mip-splatting",
        "filename": "mip_splatting-0.1.0-cp311-cp311-win_amd64.whl",
    },
    {
        "name": "gsplat",
        "filename": "gsplat-1.5.3-cp311-cp311-win_amd64.whl",
    },
    {
        "name": "torchsparse",
        "filename": "torchsparse-2.1.0-cp311-cp311-win_amd64.whl",
    },
    {
        "name": "udf-ext",
        "filename": "udf_ext-0.0.0-cp311-cp311-win_amd64.whl",
    },
)

_VERIFICATION_MODULES: Sequence[str] = (
    "torch",
    "torchvision",
    "numpy",
    "triton",
    "flash_attn",
    "nvdiffrast",
    "diffoctreerast",
    "mip_splatting",
    "gsplat",
    "torchsparse",
    "udf_ext",
)


def _documents_dir() -> Path:
    return (Path.home() / "Documents").resolve()


def get_venv_path() -> Path:
    return _documents_dir() / "TRELLIS_venv"


def _venv_python_path() -> Path:
    if os.name == "nt":
        return get_venv_path() / "Scripts" / "python.exe"
    return get_venv_path() / "bin" / "python"


def get_venv_python() -> str:
    return str(_venv_python_path())


def check_venv_exists() -> bool:
    return _venv_python_path().exists()


def create_venv() -> Tuple[bool, str]:
    if check_venv_exists():
        return True, f"✓ Virtual environment already exists at {get_venv_path()}"

    venv_path = get_venv_path()
    venv_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "venv", str(venv_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode == 0:
        return True, f"✓ Virtual environment created at {venv_path}"
    return False, f"✗ Failed to create venv: {result.stderr.strip() or result.stdout.strip()}"


def _clean_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)

    # Add venv scripts to PATH to ensure build tools like ninja are found
    if check_venv_exists():
        venv_scripts = _venv_python_path().parent
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")

    return env


def get_python_executable() -> str:
    return get_venv_python() if check_venv_exists() else sys.executable


def _pip(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    python = get_python_executable()
    full_cmd = [python, "-m", "pip", *cmd]
    return subprocess.run(
        full_cmd,
        text=True,
        capture_output=True,
        env=_clean_env(),
        timeout=600,
    )


def install_package(package: str, upgrade: bool = False) -> Tuple[bool, str]:
    args = ["install", package, "--upgrade-strategy", "only-if-needed"]
    if upgrade:
        args.append("--upgrade")
    result = _pip(args)
    if result.returncode == 0:
        return True, f"✓ Installed {package}"
    return False, result.stderr.strip() or result.stdout.strip() or "pip failed"


def check_package_installed(package_name: str) -> Tuple[bool, str]:
    result = _pip(["show", package_name])
    if result.returncode != 0:
        return False, ""
    version = "unknown"
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
            break
    return True, version


def install_pytorch(cuda_variant: str = CUDA_VARIANT) -> Tuple[bool, str]:
    args = [
        "install",
        *[
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            f"torchaudio=={TORCHAUDIO_VERSION}",
        ],
        "--index-url",
        TORCH_INDEX_URL,
        "--upgrade-strategy",
        "only-if-needed",
    ]
    result = _pip(args)
    if result.returncode == 0:
        return True, "✓ PyTorch 2.6.0 + CUDA 12.4 installed"
    return False, result.stderr.strip() or result.stdout.strip()


def install_triton() -> Tuple[bool, str]:
    system = platform.system()
    if system == "Windows":
        args = [
            "install",
            f"triton-windows=={TRITON_WINDOWS_VERSION}",
            "--no-deps",
            "--upgrade-strategy",
            "only-if-needed",
        ]
        result = _pip(args)
        if result.returncode == 0:
            return True, f"✓ triton-windows {TRITON_WINDOWS_VERSION} installed"
        return False, result.stderr.strip() or result.stdout.strip() or "pip failed"

    args = [
        "install",
        f"triton=={TRITON_VERSION}",
        "--index-url",
        TORCH_INDEX_URL,
        "--no-deps",
        "--upgrade-strategy",
        "only-if-needed",
    ]
    result = _pip(args)
    if result.returncode == 0:
        return True, f"✓ triton {TRITON_VERSION} installed from PyTorch index"
    return False, result.stderr.strip() or result.stdout.strip() or "pip failed"


def _run_python(code: str, timeout: int = 30) -> subprocess.CompletedProcess:
    python = get_python_executable()
    return subprocess.run(
        [python, "-c", code],
        text=True,
        capture_output=True,
        env=_clean_env(),
        timeout=timeout,
    )


def check_pytorch_dll_issue() -> bool:
    if platform.system() != "Windows":
        return False
    if not check_package_installed("torch")[0]:
        return False
    result = _run_python("import torch; torch.zeros(1); print('OK')")
    if result.returncode != 0:
        stderr = result.stderr.lower()
        return any(token in stderr for token in ("dll", "c10.dll", "torch_cuda.dll"))
    return False


def fix_pytorch_dll_windows() -> Tuple[bool, str]:
    if platform.system() != "Windows":
        return False, "Not running on Windows"
    _pip(["uninstall", "torch", "torchvision", "torchaudio", "-y"])
    return install_pytorch()


def check_numpy_dll_issue() -> bool:
    if platform.system() != "Windows":
        return False
    if not check_package_installed("numpy")[0]:
        return False
    result = _run_python("import numpy; numpy.array([1,2,3]); print('OK')")
    if result.returncode != 0:
        stderr = result.stderr.lower()
        return "dll" in stderr or "_multiarray_umath" in stderr
    return False


def fix_numpy_dll_windows() -> Tuple[bool, str]:
    if platform.system() != "Windows":
        return False, "Not running on Windows"
    _pip(["uninstall", "numpy", "-y"])
    result = _pip(["install", "numpy==1.26.4", "--force-reinstall"])
    if result.returncode == 0:
        return True, "✓ NumPy 1.26.4 reinstalled"
    return False, result.stderr.strip() or result.stdout.strip()


def get_installation_status(detailed: bool = True) -> Dict[str, Dict[str, str]]:
    status: Dict[str, Dict[str, str]] = {
        "venv": {"path": str(get_venv_path()), "exists": check_venv_exists()},
    }

    torch_installed, torch_version = check_package_installed("torch")
    status["torch"] = {
        "installed": torch_installed,
        "version": torch_version,
    }

    numpy_installed, numpy_version = check_package_installed("numpy")
    status["numpy"] = {
        "installed": numpy_installed,
        "version": numpy_version,
    }

    trellis_dir = Path(__file__).parent / "trellis"
    status["trellis"] = {
        "installed": trellis_dir.exists(),
        "path": str(trellis_dir),
    }

    if detailed:
        installed = 0
        for package in TRELLIS_DEPENDENCIES:
            pkg_name = package.split("[")[0].split("==")[0]
            ok, _ = check_package_installed(pkg_name)
            installed += int(ok)
        status["dependencies"] = {
            "total": str(len(TRELLIS_DEPENDENCIES)),
            "installed": str(installed),
        }
    return status


def _wheel_dir() -> Path:
    path = _documents_dir() / "wheels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_wheel(filename: str) -> Path:
    target = _wheel_dir() / filename
    if target.exists():
        return target
    url = f"{WHEELS_BASE_URL}/{filename}"
    try:
        with urlopen(url) as response, open(target, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except (URLError, HTTPError) as exc:
        raise RuntimeError(f"Failed to download {filename}: {exc}") from exc
    return target


def install_custom_wheels() -> Tuple[bool, List[str]]:
    messages: List[str] = []
    success = True
    for wheel in CUSTOM_WHEELS:
        filename = wheel["filename"]
        name = wheel["name"]
        try:
            wheel_path = _download_wheel(filename)
            result = _pip([
                "install",
                str(wheel_path),
                "--no-deps",
                "--force-reinstall",
            ])
            if result.returncode == 0:
                messages.append(f"✓ {name} installed")
            else:
                success = False
                messages.append(f"✗ {name} failed: {result.stderr.strip() or result.stdout.strip()}")
        except Exception as exc:  # noqa: BLE001
            fallback_spec = wheel.get("pip_fallback")
            if fallback_spec:
                result = _pip(["install", fallback_spec, "--no-deps", "--upgrade-strategy", "only-if-needed"])
                if result.returncode == 0:
                    messages.append(
                        f"✓ {name} installed from PyPI fallback ({fallback_spec})"
                    )
                    continue
                messages.append(
                    f"✗ {name} fallback failed: {result.stderr.strip() or result.stdout.strip()}"
                )
                success = False
                continue
            success = False
            messages.append(
                f"✗ {name} failed: {exc}. Set TRELLIS_WHEELS_BASE_URL to override the download source"
            )
    return success, messages


def verify_installation() -> Tuple[bool, List[str]]:
    messages: List[str] = []
    healthy = True

    result = _run_python(
        textwrap.dedent(
            """
            import torch
            info = {
                "version": torch.__version__,
                "cuda": torch.cuda.is_available(),
            }
            print(json.dumps(info))
            """
        )
    )
    if result.returncode == 0:
        payload = json.loads(result.stdout.strip())
        messages.append(
            f"PyTorch {payload['version']} (CUDA available: {payload['cuda']})"
        )
    else:
        healthy = False
        messages.append(f"PyTorch test failed: {result.stderr.strip()}")

    result = _run_python("import numpy as np; print(np.__version__)")
    if result.returncode == 0:
        messages.append(f"NumPy {result.stdout.strip()} OK")
    else:
        healthy = False
        messages.append(f"NumPy import failed: {result.stderr.strip()}")

    failed = []
    for module in _VERIFICATION_MODULES:
        result = _run_python(f"import {module}; print('OK')")
        if result.returncode == 0:
            messages.append(f"{module}: OK")
        else:
            failed.append(module)
            messages.append(f"{module}: {result.stderr.strip() or 'import failed'}")
    if failed:
        healthy = False

    return healthy, messages
