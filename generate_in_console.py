"""
Run TRELLIS generation in a separate console window with full logging.
This script is launched by Blender to run generation externally.
"""

import sys
import os
from pathlib import Path
import time
import argparse

# Import console logger
from console_logger import ConsoleLogger

# Initialize logger
logger = ConsoleLogger("TRELLIS_generation")

def main():
    """Main generation function with logging"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TRELLIS 3D Generation Console')
    parser.add_argument('input', nargs='*', help='Input image path(s) (for image-to-3D or multi-image)')
    parser.add_argument('--text', type=str, help='Text prompt (for text-to-3D)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sparse-steps', type=int, default=8, help='Sparse structure sampling steps')
    parser.add_argument('--sparse-cfg', type=float, default=7.5, help='Sparse structure CFG strength')
    parser.add_argument('--slat-steps', type=int, default=8, help='SLAT sampling steps')
    parser.add_argument('--slat-cfg', type=float, default=3.0, help='SLAT CFG strength')
    parser.add_argument('--texture-size', type=str, default='1024', choices=['512', '1024', '2048'],
                        help='Texture resolution')
    parser.add_argument('--mesh-simplify', type=float, default=0.95, help='Mesh simplification ratio')
    parser.add_argument('--multi-image', action='store_true', help='Multi-image generation mode')
    parser.add_argument('--preprocess', action='store_true', help='Remove background from images')

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
    sparse_steps = args.sparse_steps
    sparse_cfg = args.sparse_cfg
    slat_steps = args.slat_steps
    slat_cfg = args.slat_cfg

    logger.header("TRELLIS 3D Generation")
    logger.env_info()
    logger.divider()

    start_time = time.time()

    try:
        # Add addon directory to path
        addon_dir = Path(__file__).parent
        if str(addon_dir) not in sys.path:
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
            if not manager.initialize(use_cuda=True):
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
        logger.plain(f"Sparse Steps: {sparse_steps}, CFG: {sparse_cfg}", indent=1)
        logger.plain(f"SLAT Steps: {slat_steps}, CFG: {slat_cfg}", indent=1)
        logger.plain(f"Texture Size: {texture_size}x{texture_size}", indent=1)
        logger.plain(f"Mesh Simplify: {mesh_simplify}", indent=1)
        logger.plain(f"Preprocess (Remove BG): {args.preprocess}", indent=1)
        logger.plain("")

        if image_paths and len(image_paths) > 1 and is_multi_image:
            # Multi-image generation
            logger.info(f"Input Images: {len(image_paths)} files")
            outputs = manager.generate_from_multi_image(
                image_paths=image_paths,
                seed=seed,
                sparse_steps=sparse_steps,
                sparse_cfg=sparse_cfg,
                slat_steps=slat_steps,
                slat_cfg=slat_cfg,
                preprocess=args.preprocess,
                formats=['mesh', 'gaussian']
            )
        elif image_paths:
            # Single image generation
            logger.info(f"Input Image: {image_paths[0]}")
            outputs = manager.generate_from_image(
                image_path=image_paths[0],
                seed=seed,
                sparse_steps=sparse_steps,
                sparse_cfg=sparse_cfg,
                slat_steps=slat_steps,
                slat_cfg=slat_cfg,
                preprocess=args.preprocess,
                formats=['mesh', 'gaussian']
            )
        else:
            # Text generation
            logger.info(f"Text Prompt: {text_prompt}")
            outputs = manager.generate_from_text(
                prompt=text_prompt,
                seed=seed,
                sparse_steps=sparse_steps,
                sparse_cfg=sparse_cfg,
                slat_steps=slat_steps,
                slat_cfg=slat_cfg,
                formats=['mesh', 'gaussian']
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
                                    texture_size=texture_size):
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
        
        logger.success(f"Generation Complete in {minutes}m {seconds}s!")
        logger.plain("")
        logger.plain("✅ 3D model generated successfully", indent=1)
        logger.plain(f"✅ Output saved to Documents/TRELLIS_Output", indent=1)
        logger.plain(f"✅ Texture resolution: {texture_size}x{texture_size}", indent=1)
        
        logger.divider()
        logger.info(f"Log file: {logger.log_file}")
        logger.divider()

        logger.plain("\nPress Enter to close...", indent=0)
        input()
        return 0

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

