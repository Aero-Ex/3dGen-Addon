"""
Initialization script that runs in a separate console window
Shows all initialization output with professional formatting
"""

import sys
import os
import time
from pathlib import Path

# CRITICAL: Set attention backend BEFORE any imports
# This prevents xformers/triton issues on Windows
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

# Import professional logging system
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from console_logger import ConsoleLogger

# Initialize professional logger
logger = ConsoleLogger("TRELLIS_init")


def main():
    logger.header("TRELLIS Pipeline Initialization")

    logger.instructions([
        "This process will download ~10GB of models on first run",
        "Please keep this window open",
        "You can use Blender normally while this runs"
    ])

    logger.env_info()
    logger.divider()

    start_time = time.time()

    # Add addon directory to sys.path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import dependency_installer
        logger.success("Loaded dependency_installer module")
    except ImportError as e:
        logger.error("Failed to import dependency_installer!")
        logger.plain(f"Error: {e}", indent=1)
        log_and_print(f"\nLog saved to: {LOG_FILE}")
        print("\nPress Enter to exit...")
        input()
        return 1

    # Verify virtual environment exists
    print_section("📦 Verifying Virtual Environment")

    venv_path = dependency_installer.get_venv_path()
    logger.plain(f"Location: {venv_path}", indent=1)

    if not dependency_installer.check_venv_exists():
        logger.error("Virtual environment not found!")
        logger.plain("Please run 'Install Dependencies' first.", indent=1)
        logger.plain("\nPress Enter to exit...", indent=1)
        input()
        return 1

    logger.success("Virtual environment ready")
    logger.divider()

    # Verify PyTorch is installed
    logger.section("Verifying PyTorch Installation")

    status = dependency_installer.get_installation_status(detailed=False)
    torch_installed = status.get('torch', {}).get('installed', False)
    torch_version = status.get('torch', {}).get('version', 'unknown')
    cuda_available = status.get('torch', {}).get('cuda', False)

    if not torch_installed:
        logger.error("PyTorch not installed!")
        logger.plain("Please run 'Install Dependencies' first.", indent=1)
        logger.plain("\nPress Enter to exit...", indent=1)
        input()
        return 1

    logger.success(f"PyTorch {torch_version}")
    if cuda_available:
        logger.success("CUDA available")
    else:
        logger.warning("CUDA not available (will use CPU - slow!)")

    logger.divider()

    # Initialize TRELLIS pipelines
    logger.section("Initializing TRELLIS Pipelines")

    logger.info("This will download models from HuggingFace on first run...")
    logger.info("⏳ Download size: ~10GB")
    logger.info("⏳ Time: 5-10 minutes (depending on internet speed)")
    logger.plain("")

    success = False
    try:
        logger.subsection("Starting initialization...")
        logger.plain("")

        # CRITICAL: Set up venv in sys.path FIRST
        logger.step("Setting up virtual environment", 1, 4)

        venv_path = dependency_installer.get_venv_path()

        # Get venv site-packages
        if os.name == 'nt':  # Windows
            site_packages = venv_path / 'Lib' / 'site-packages'
        else:  # Linux/Mac
            lib_path = venv_path / 'lib'
            python_dirs = [d for d in lib_path.iterdir() if d.name.startswith('python3.')]
            if python_dirs:
                site_packages = python_dirs[0] / 'site-packages'
            else:
                raise RuntimeError(f"Could not find site-packages in venv")

        site_packages_str = str(site_packages)

        # Ensure venv is at the BEGINNING of sys.path
        if site_packages_str in sys.path:
            sys.path.remove(site_packages_str)
        sys.path.insert(0, site_packages_str)
        logger.plain(f"Venv added to sys.path[0]: {site_packages_str}", indent=1)

        # CRITICAL: Clear any cached torch modules
        torch_modules = [key for key in sys.modules.keys() if key.startswith('torch')]
        if torch_modules:
            for module in torch_modules:
                del sys.modules[module]
            logger.plain(f"Cleared {len(torch_modules)} torch modules from cache", indent=1)

        # Set environment variables
        os.environ['ATTN_BACKEND'] = 'sdpa'
        os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'  # Use PyTorch SDPA instead of xformers
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10 minute timeout for large downloads
        os.environ['TRANSFORMERS_VERBOSITY'] = 'info'  # Show transformers library progress
        logger.plain("Environment variables set (using SDPA attention backend)", indent=1)
        logger.plain("HuggingFace timeout: 10 minutes", indent=1)

        # Import PyTorch from venv
        logger.step("Importing PyTorch from venv", 2, 4)
        import torch
        logger.plain(f"PyTorch {torch.__version__} loaded", indent=1)
        logger.plain(f"CUDA available: {torch.cuda.is_available()}", indent=1)
        if torch.cuda.is_available():
            logger.plain(f"GPU: {torch.cuda.get_device_name(0)}", indent=1)

        # Import TRELLIS (bundled with addon)
        logger.step("Importing TRELLIS pipelines", 3, 4)
        sys.path.insert(0, script_dir)  # Ensure addon dir is in path for trellis import
        from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
        logger.plain("TRELLIS modules imported", indent=1)

        # Load image pipeline
        logger.step("Loading Image-to-3D pipeline from HuggingFace", 4, 4)
        logger.info("This may take 5-10 minutes (downloading ~10GB)")
        logger.plain("")

        image_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        logger.success("Image pipeline loaded!")

        # Load text pipeline
        logger.subsection("Loading Text-to-3D pipeline from HuggingFace...")
        logger.info("This may take 5-10 minutes (downloading ~10GB + CLIP model)")
        logger.plain("")

        logger.info("Note: CLIP text model (~1.7GB) will download silently.")
        logger.info("Please be patient if no output for several minutes.")
        logger.plain("")

        text_pipeline = TrellisTextTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-text-xlarge"
        )
        logger.success("Text pipeline loaded!")

        # Move to CUDA if available
        if cuda_available:
            logger.subsection("Moving pipelines to CUDA...")
            image_pipeline.cuda()
            text_pipeline.cuda()
            logger.success("Pipelines moved to GPU")

        success = True

        logger.divider()
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        logger.success(f"Initialization Complete in {minutes}m {seconds}s!")
        logger.plain("")
        logger.plain("✅ TRELLIS pipelines initialized successfully", indent=1)
        logger.plain("✅ Models downloaded and cached", indent=1)
        if cuda_available:
            logger.plain("✅ GPU acceleration enabled", indent=1)
        else:
            logger.plain("⚠ Running on CPU", indent=1)
        
        logger.divider()

        logger.box(
            "Next Steps",
            [
                "1. Return to Blender",
                "2. Use 'Image to 3D' or 'Text to 3D' in the addon panel",
                "3. Start generating 3D models!"
            ]
        )

    except OSError as e:
        # Handle Windows paging file / memory errors
        if "paging file" in str(e).lower() or "1455" in str(e):
            logger.error("MEMORY ERROR: Windows paging file is too small!")
            logger.plain("")
            logger.box(
                "TRELLIS Memory Requirements",
                [
                    "TRELLIS requires large amounts of virtual memory (~20-40 GB)",
                    "",
                    "SOLUTION: Increase Windows Virtual Memory",
                    "1. Press Win+Pause (or right-click 'This PC' → Properties)",
                    "2. Click 'Advanced system settings'",
                    "3. Under Performance → Click 'Settings'",
                    "4. Go to 'Advanced' tab → Click 'Change' under Virtual Memory",
                    "5. Uncheck 'Automatically manage paging file size'",
                    "6. Select your C: drive and choose 'Custom size'",
                    "7. Set:",
                    "   Initial size (MB):  24576  (24 GB)",
                    "   Maximum size (MB):  49152  (48 GB)",
                    "8. Click 'Set' → 'OK'",
                    "9. RESTART YOUR COMPUTER"
                ]
            )

            logger.warning(f"Error details saved to log file")
        else:
            # Other OS errors
            import traceback
            logger.error("Fatal error during initialization:")
            logger.plain(str(e), indent=1)
            logger.plain("")

            tb_str = traceback.format_exc()
            logger.plain("Full traceback:", indent=1)
            logger.plain(tb_str, indent=2)

            logger.warning("Error details saved to log file")

    except Exception as e:
        import traceback

        logger.error("Fatal error during initialization:")
        logger.plain(str(e), indent=1)
        logger.plain("")

        tb_str = traceback.format_exc()
        logger.plain("Full traceback:", indent=1)
        logger.plain(tb_str, indent=2)

        logger.warning("Error details saved to log file")

    logger.plain("\nPress Enter to close this window...")
    input()

    return 0


if __name__ == "__main__":
    try:
        logger.plain("Script starting...", indent=0)
        sys.exit(main())
    except KeyboardInterrupt:
        logger.plain("")
        logger.warning("Initialization cancelled by user.")
        logger.plain("\nPress Enter to exit...")
        input()
        sys.exit(1)
    except Exception as e:
        import traceback

        logger.error(f"Fatal error: {e}")
        tb_str = traceback.format_exc()
        logger.plain("\nFull traceback:", indent=1)
        logger.plain(tb_str, indent=2)
        logger.warning("Error details saved to log file")
        logger.plain("\nPress Enter to exit...")
        input()
        sys.exit(1)
