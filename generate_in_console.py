"""
Run TRELLIS generation in a separate console window with full logging.
This script is launched by Blender to run generation externally.
"""

import sys
import os
from pathlib import Path
import time
import argparse

ADDON_DIR = Path(__file__).parent
if str(ADDON_DIR) not in sys.path:
    sys.path.insert(0, str(ADDON_DIR))

from console_logger import ConsoleLogger
from pipeline_metadata import (
    get_pipeline_descriptor,
    is_mode_supported,
    is_pipeline_available,
    pipeline_cli_choices,
    pipeline_key_from_cli,
)

# Initialize logger
logger = ConsoleLogger("TRELLIS_generation")

def main():
    """Main generation function with logging"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='3D Generation Console')
    parser.add_argument('input', nargs='*', help='Input image path(s) (for image-to-3D or multi-image)')
    parser.add_argument('--text', type=str, help='Text prompt (for text-to-3D)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # TRELLIS-specific parameters
    parser.add_argument('--sparse-steps', type=int, default=8, help='TRELLIS: Sparse structure sampling steps')
    parser.add_argument('--sparse-cfg', type=float, default=7.5, help='TRELLIS: Sparse structure CFG strength')
    parser.add_argument('--slat-steps', type=int, default=8, help='TRELLIS: SLAT sampling steps')
    parser.add_argument('--slat-cfg', type=float, default=3.0, help='TRELLIS: SLAT CFG strength')
    
    # Direct3D-S2 specific parameters
    parser.add_argument('--direct3d-steps', type=int, default=15, help='Direct3D-S2: Sampling steps')
    parser.add_argument('--direct3d-guidance', type=float, default=7.0, help='Direct3D-S2: Guidance scale')
    parser.add_argument('--direct3d-resolution', type=str, default='AUTO', choices=['AUTO', '512', '1024'],
                        help='Direct3D-S2: SDF resolution (AUTO/512/1024)')
    

    
    # Common parameters
    parser.add_argument('--texture-size', type=str, default='1024', choices=['512', '1024', '2048'],
                        help='Texture resolution')
    parser.add_argument('--export-mode', type=str, default='MESH_TEXTURE', choices=['MESH_ONLY', 'MESH_TEXTURE'],
                        help='Export mode: MESH_ONLY (no texture) or MESH_TEXTURE (default)')
    parser.add_argument('--mesh-simplify', type=float, default=0.95, help='Mesh simplification ratio')
    parser.add_argument('--multi-image', action='store_true', help='Multi-image generation mode')
    parser.add_argument('--preprocess', action='store_true', help='Remove background from images')
    parser.add_argument('--pipeline', type=str, choices=pipeline_cli_choices(), default='trellis',
                        help='Select backend pipeline to run')

    args = parser.parse_args()
    
    # Print received arguments for debugging
    logger.subsection("Command Line Arguments")
    logger.plain(f"sys.argv: {sys.argv}", indent=1)
    logger.plain(f"Parsed args: {args}", indent=1)
    logger.divider()

    # Extract parameters
    texture_size = int(args.texture_size)
    mesh_simplify = args.mesh_simplify
    seed = args.seed
    
    # Extract pipeline-specific parameters
    sparse_steps = args.sparse_steps
    sparse_cfg = args.sparse_cfg
    slat_steps = args.slat_steps
    slat_cfg = args.slat_cfg
    direct3d_steps = args.direct3d_steps
    direct3d_guidance = args.direct3d_guidance
    direct3d_resolution = args.direct3d_resolution


    pipeline_key = pipeline_key_from_cli(args.pipeline.lower())
    descriptor = get_pipeline_descriptor(pipeline_key)
    pipeline_name = descriptor.label

    # *** CRITICAL: Set backend environment variables BEFORE any imports ***
    # These must be set before importing PipelineManager or any TRELLIS/Direct3D modules
    if pipeline_key == 'DIRECT3D':
        # Direct3D-S2 uses torchsparse and flash_attn
        os.environ['SPARSE_BACKEND'] = 'torchsparse'
        os.environ['SPARSE_ATTN_BACKEND'] = 'flash_attn'
        os.environ['ATTN_BACKEND'] = 'flash_attn'  # Direct3D attention module also checks this
        
        # Configure MSVC path for Triton JIT compilation
        # MSVC found at: C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207
        msvc_base = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
        msvc_version = "14.44.35207"
        msvc_bin = rf"{msvc_base}\VC\Tools\MSVC\{msvc_version}\bin\Hostx64\x64"
        windows_sdk_base = rf"{msvc_base}\VC\Tools\MSVC\{msvc_version}\include"
        
        # Add MSVC to PATH so Triton can find cl.exe
        current_path = os.environ.get('PATH', '')
        if msvc_bin not in current_path:
            os.environ['PATH'] = f"{msvc_bin};{current_path}"
        
        # Set Visual Studio environment for compilation
        # CRITICAL: Must include both MSVC and Windows SDK paths
        msvc_include = rf"{msvc_base}\VC\Tools\MSVC\{msvc_version}\include"
        
        # Windows SDK paths (required for stdlib.h, windows.h, etc.)
        windows_sdk_base = r"C:\Program Files (x86)\Windows Kits\10"
        # Use latest SDK version available (typically 10.0.xxxxx.0)
        windows_sdk_version = "10.0.26100.0"  # User's installed SDK version
        windows_sdk_include = rf"{windows_sdk_base}\Include\{windows_sdk_version}\ucrt"
        windows_sdk_shared = rf"{windows_sdk_base}\Include\{windows_sdk_version}\shared"
        windows_sdk_um = rf"{windows_sdk_base}\Include\{windows_sdk_version}\um"
        
        # Python include path (for Python.h) - CRITICAL for Triton
        # Get the venv's Python include directory
        python_include = os.path.join(os.path.dirname(sys.executable), '..', 'Include')
        python_include = os.path.abspath(python_include)
        
        # Combine all include paths (semicolon separated)
        # Python include MUST come first to override any Blender Python paths
        os.environ['INCLUDE'] = f"{python_include};{msvc_include};{windows_sdk_include};{windows_sdk_shared};{windows_sdk_um}"
        
        # Library paths
        msvc_lib = rf"{msvc_base}\VC\Tools\MSVC\{msvc_version}\lib\x64"
        windows_sdk_lib_ucrt = rf"{windows_sdk_base}\Lib\{windows_sdk_version}\ucrt\x64"
        windows_sdk_lib_um = rf"{windows_sdk_base}\Lib\{windows_sdk_version}\um\x64"
        
        # Python library path (for python311.lib) - CRITICAL for Triton
        python_libs = os.path.join(os.path.dirname(sys.executable), '..', 'libs')
        python_libs = os.path.abspath(python_libs)
        
        # Python libs MUST come first so Triton finds python311.lib
        os.environ['LIB'] = f"{python_libs};{msvc_lib};{windows_sdk_lib_ucrt};{windows_sdk_lib_um}"
        
        # CRITICAL: Force Triton to use MSVC cl.exe instead of bundled TCC
        # Without this, Triton defaults to TCC which has Python.h issues
        os.environ['CC'] = rf"{msvc_bin}\cl.exe"
        os.environ['CXX'] = rf"{msvc_bin}\cl.exe"
        
        # JIT compilation enabled - interpreter mode disabled for full performance
        print("✓ MSVC + Windows SDK configured for Triton JIT compilation (FAST mode)")
        print(f"  Using cl.exe from: {msvc_bin}")



        # Set resolution override if not AUTO
        if direct3d_resolution != 'AUTO':
            os.environ['DIRECT3D_S2_RESOLUTION'] = direct3d_resolution
    else:
        # TRELLIS uses spconv and xformers
        os.environ['SPARSE_BACKEND'] = 'spconv'
        os.environ['SPARSE_ATTN_BACKEND'] = 'xformers'
        os.environ['ATTN_BACKEND'] = 'xformers'

    inferred_multi = args.multi_image or (not args.text and len(args.input) > 1)
    requested_mode = 'TEXT' if args.text else ('MULTI_IMAGE' if inferred_multi else 'IMAGE')
    is_multi_image = requested_mode == 'MULTI_IMAGE'

    if not is_pipeline_available(pipeline_key):
        logger.error(f"{pipeline_name} pipeline is not available on this build")
        logger.plain(descriptor.status_hint, indent=1)
        logger.plain("\nPress Enter to close...", indent=1)
        input()
        return 1

    if not is_mode_supported(pipeline_key, requested_mode):
        logger.error(f"{pipeline_name} pipeline does not support {requested_mode.replace('_', ' ').title()} mode")
        logger.plain("Switch to a supported mode or change pipelines.", indent=1)
        logger.plain("\nPress Enter to close...", indent=1)
        input()
        return 1

    logger.header("TRELLIS 3D Generation")
    logger.info(f"Selected pipeline: {pipeline_name}")
    logger.env_info()
    logger.divider()

    start_time = time.time()

    try:
        # Add addon directory to path (already inserted globally, but ensure precedence)
        addon_dir = ADDON_DIR
        if str(addon_dir) in sys.path:
            sys.path.remove(str(addon_dir))
        sys.path.insert(0, str(addon_dir))

        # Setup venv
        from dependency_installer import get_venv_path, get_python_executable
        venv_path = get_venv_path()

        if not venv_path.exists():
            print(f"❌ Virtual environment not found at: {venv_path}")
            print("   Please run 'Install Dependencies' first.")
            input("\nPress Enter to close...")
            return 1

        print(f"✓ Found venv at: {venv_path}")

        # Add venv to sys.path
        if os.name == 'nt':
            site_packages = venv_path / 'Lib' / 'site-packages'
        else:
            lib_path = venv_path / 'lib'
            python_dirs = [d for d in lib_path.iterdir() if d.name.startswith('python3.')]
            if python_dirs:
                site_packages = python_dirs[0] / 'site-packages'
            else:
                logger.error("Could not find site-packages in venv")
                logger.plain("\nPress Enter to close...", indent=1)
                input()
                return 1

        site_packages_str = str(site_packages)
        if site_packages_str in sys.path:
            sys.path.remove(site_packages_str)
        sys.path.insert(0, site_packages_str)
        logger.success("Added venv to sys.path")

        # Import TRELLIS
        logger.section("Importing TRELLIS modules")
        from pipeline_manager import PipelineManager
        from PIL import Image
        logger.success("TRELLIS modules imported")

        # Determine input type
        image_paths = []
        text_prompt = None
        is_multi_image = args.multi_image

        if args.text:
            # Text-to-3D mode
            text_prompt = args.text
            logger.info(f"Mode: Text-to-3D")
            logger.plain(f"Text prompt: {text_prompt}", indent=1)
        elif args.input:
            # Image-to-3D or Multi-Image mode
            if is_multi_image or len(args.input) > 1:
                # Multi-image mode
                image_paths = args.input
                # Validate all paths
                invalid_paths = [p for p in image_paths if not os.path.exists(p)]
                if invalid_paths:
                    logger.error("Image file(s) not found:")
                    for p in invalid_paths:
                        logger.plain(p, indent=2)
                    logger.plain("\nPress Enter to close...", indent=1)
                    input()
                    return 1
                logger.info(f"Mode: Multi-Image to 3D")
                logger.plain(f"Images ({len(image_paths)}):", indent=1)
                for i, path in enumerate(image_paths, 1):
                    logger.plain(f"{i}. {os.path.basename(path)}", indent=2)
            else:
                # Single image mode
                image_path = args.input[0]
                if not os.path.exists(image_path):
                    logger.error(f"Image file not found: {image_path}")
                    logger.plain("\nPress Enter to close...", indent=1)
                    input()
                    return 1
                image_paths = [image_path]
                logger.info(f"Mode: Image-to-3D")
                logger.plain(f"Image: {image_path}", indent=1)
        else:
            logger.error("No input provided!")
            logger.box(
                "Usage",
                [
                    "Image-to-3D:      python generate_in_console.py <image_path> [options]",
                    "Multi-Image:      python generate_in_console.py <image1> <image2> ... --multi-image [options]",
                    "Text-to-3D:       python generate_in_console.py --text <prompt> [options]"
                ]
            )
            logger.plain("\nPress Enter to close...", indent=1)
            input()
            return 1

        # Initialize pipeline
        logger.divider()
        logger.section("Initializing TRELLIS Pipeline")

        manager = PipelineManager()
        
        # Check if already initialized (e.g., by Blender's auto-init)
        if manager.initialized:
            logger.success("TRELLIS pipeline already initialized")
        else:
            # skip_cleanup=True because we're running from console, not Blender
            if not manager.initialize(use_cuda=True, skip_cleanup=True):
                logger.error("Failed to initialize TRELLIS")
                logger.plain("\nPress Enter to close...", indent=1)
                input()
                return 1
            logger.success("TRELLIS pipeline initialized")

        logger.divider()

        # Generate
        logger.section("Starting 3D Generation")

        logger.subsection("Generation Parameters")
        logger.plain(f"Seed: {seed}", indent=1)
        
        # Display pipeline-specific parameters
        if pipeline_key == 'TRELLIS':
            logger.plain(f"Sparse Steps: {sparse_steps}, CFG: {sparse_cfg}", indent=1)
            logger.plain(f"SLAT Steps: {slat_steps}, CFG: {slat_cfg}", indent=1)
        elif pipeline_key == 'DIRECT3D':
            logger.plain(f"Sampling Steps: {direct3d_steps}", indent=1)
            logger.plain(f"Guidance Scale: {direct3d_guidance}", indent=1)

        
        logger.plain(f"Texture Size: {texture_size}x{texture_size}", indent=1)
        logger.plain(f"Mesh Simplify: {mesh_simplify}", indent=1)
        logger.plain(f"Preprocess (Remove BG): {args.preprocess}", indent=1)
        logger.plain("")
        
        # Build generation kwargs based on pipeline
        gen_kwargs = {
            'seed': seed,
            'preprocess': args.preprocess,
            'formats': ['mesh', 'gaussian'],
            'pipeline_key': pipeline_key,
        }
        
        if pipeline_key == 'TRELLIS':
            gen_kwargs.update({
                'sparse_steps': sparse_steps,
                'sparse_cfg': sparse_cfg,
                'slat_steps': slat_steps,
                'slat_cfg': slat_cfg,
            })
        elif pipeline_key == 'DIRECT3D':
            # Pass resolution via environment variable so PipelineManager picks it up
            if direct3d_resolution != 'AUTO':
                os.environ['DIRECT3D_S2_RESOLUTION'] = direct3d_resolution
                logger.info(f"Setting Direct3D-S2 Resolution: {direct3d_resolution}")

            gen_kwargs.update({
                'num_steps': direct3d_steps,
                'guidance_scale': direct3d_guidance,
                'simplify_mesh': mesh_simplify, # Pass mesh simplify ratio
            })
            
            # Redirect stdout/stderr to logger to capture pipeline output
            # Use sys.__stdout__ to avoid recursion since logger.plain() calls print()
            class StreamToLogger(object):
                def __init__(self, logger):
                    self.logger = logger
                def write(self, buf):
                    for line in buf.rstrip().splitlines():
                        # Write to log file directly
                        self.logger._write_file(line + "\n")
                        # Write to original stdout
                        sys.__stdout__.write(line + "\n")
                def flush(self):
                    sys.__stdout__.flush()
            
            sys.stdout = StreamToLogger(logger)
            sys.stderr = StreamToLogger(logger)


        if image_paths and len(image_paths) > 1 and is_multi_image:
            # Multi-image generation
            logger.info(f"Input Images: {len(image_paths)} files")
            outputs = manager.generate_from_multi_image(
                image_paths=image_paths,
                **gen_kwargs
            )
        elif image_paths:
            # Single image generation
            logger.info(f"Input Image: {image_paths[0]}")
            outputs = manager.generate_from_image(
                image_path=image_paths[0],
                **gen_kwargs
            )
        else:
            # Text generation
            logger.info(f"Text Prompt: {text_prompt}")
            outputs = manager.generate_from_text(
                prompt=text_prompt,
                **gen_kwargs
            )

        if outputs:
            # Export to GLB
            output_dir = Path.home() / "Documents" / "TRELLIS_Output"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"trellis_{timestamp}.glb"

            logger.divider()
            logger.section("Exporting Model")
            logger.plain(f"Output: {output_file}", indent=1)
            logger.plain(f"Texture size: {texture_size}x{texture_size}", indent=1)
            logger.plain(f"Mesh simplification: {mesh_simplify}", indent=1)

            if manager.export_to_glb(outputs, str(output_file),
                                    simplify=mesh_simplify,
                                    texture_size=texture_size,
                                    export_mode=args.export_mode):
                logger.success("3D model exported successfully!")
                logger.box(
                    "Output File",
                    [str(output_file)]
                )
            else:
                logger.error("Export failed")
        else:
            logger.error("Generation failed - no outputs")

        # Cleanup
        manager.cleanup()

        logger.divider()

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        if outputs:
            logger.success(f"Generation Complete in {minutes}m {seconds}s!")
            logger.plain("")
            logger.plain("✅ 3D model generated successfully", indent=1)
            logger.plain(f"✅ Output saved to Documents/TRELLIS_Output", indent=1)
            logger.plain(f"✅ Texture resolution: {texture_size}x{texture_size}", indent=1)
        else:
            logger.error(f"Generation Failed after {minutes}m {seconds}s")
            logger.plain("")
            logger.plain("❌ 3D model generation failed", indent=1)
            logger.plain("❌ Check error messages above for details", indent=1)

        logger.divider()
        logger.info(f"Log file: {logger.log_file}")
        logger.divider()

        logger.plain("\nPress Enter to close...", indent=0)
        input()
        return 0 if outputs else 1

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        logger.plain("\nFull traceback:", indent=1)
        logger.plain(traceback.format_exc(), indent=2)
        logger.warning("Full log saved to log file")
        logger.plain("\nPress Enter to close...", indent=0)
        input()
        return 1

if __name__ == "__main__":
    sys.exit(main())

