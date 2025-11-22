"""
Console script for upscaling existing meshes using Direct3D-S2
Usage: python upscale_in_console.py input_mesh.obj reference_image.png [options]
"""

import os

# Set memory management before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Upscale existing 3D mesh using Direct3D-S2 refiner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Upscale with 512³ resolution (works on 6GB VRAM)
  python upscale_in_console.py mesh.obj image.png --resolution 512

  # Preserve input mesh structure (low steps, low guidance)
  python upscale_in_console.py mesh.obj image.png --steps 8 --guidance 4.0

  # Strong refinement (high steps, high guidance)
  python upscale_in_console.py mesh.obj image.png --steps 25 --guidance 10.0

  # Upscale without simplification
  python upscale_in_console.py mesh.obj image.png --no-simplify

  # Custom output path
  python upscale_in_console.py mesh.obj image.png --output refined_mesh.obj
        '''
    )

    parser.add_argument('input_mesh', type=str,
                       help='Path to input mesh file (OBJ, PLY, STL, FBX, etc.)')
    parser.add_argument('reference_image', type=str,
                       help='Path to reference image')
    parser.add_argument('--output', type=str, default=None,
                       help='Output mesh path (default: input_mesh_upscaled.obj)')
    parser.add_argument('--resolution', type=int, choices=[512, 1024], default=512,
                       help='Refinement resolution: 512 (6GB VRAM) or 1024 (12GB+ VRAM). Default: 512')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--simplify', type=float, default=0.95,
                       help='Mesh simplification ratio 0.1-1.0 (default: 0.95)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Disable mesh simplification (very high poly count)')
    parser.add_argument('--steps', type=int, default=15,
                       help='Diffusion steps: lower (5-10) preserves input, higher (15-30) refines more (default: 15)')
    parser.add_argument('--guidance', type=float, default=7.0,
                       help='Guidance scale: lower (3-5) preserves mesh, higher (7-12) follows image more (default: 7.0)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_mesh):
        print(f"Error: Input mesh not found: {args.input_mesh}")
        return 1

    if not os.path.exists(args.reference_image):
        print(f"Error: Reference image not found: {args.reference_image}")
        return 1

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Use Documents/TRELLIS_Output as default output directory
        output_dir = Path.home() / "Documents" / "TRELLIS_Output"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = Path(args.input_mesh)
        output_filename = f"{input_path.stem}_upscaled{input_path.suffix}"
        output_path = str(output_dir / output_filename)

    simplify_ratio = 1.0 if args.no_simplify else args.simplify

    print("=" * 80)
    print("                     MESH UPSCALE - Direct3D-S2")
    print("=" * 80)
    print(f"Input Mesh:      {args.input_mesh}")
    print(f"Reference Image: {args.reference_image}")
    print(f"Output Path:     {output_path}")
    print(f"Resolution:      {args.resolution}³")
    print(f"Steps:           {args.steps} (lower=preserve input, higher=more refinement)")
    print(f"Guidance:        {args.guidance} (lower=preserve mesh, higher=follow image)")
    print(f"Seed:            {args.seed}")
    print(f"Simplification:  {'Disabled' if args.no_simplify else f'{simplify_ratio:.2f}'}")
    print("-" * 80)

    # Setup venv first (CRITICAL - must happen before any torch imports)
    print("Setting up virtual environment...")
    addon_dir = Path(__file__).parent
    if str(addon_dir) not in sys.path:
        sys.path.insert(0, str(addon_dir))

    try:
        from dependency_installer import get_venv_path
        venv_path = get_venv_path()

        if not venv_path.exists():
            print(f"❌ Virtual environment not found at: {venv_path}")
            print("   Please run 'Install Dependencies' from Blender first.")
            input("\nPress Enter to close...")
            return 1

        print(f"✓ Found venv at: {venv_path}")

        # Add venv site-packages to sys.path
        if os.name == 'nt':
            site_packages = venv_path / 'Lib' / 'site-packages'
        else:
            lib_path = venv_path / 'lib'
            python_dirs = [d for d in lib_path.iterdir() if d.name.startswith('python3.')]
            if python_dirs:
                site_packages = python_dirs[0] / 'site-packages'
            else:
                print(f"❌ Could not find site-packages in venv")
                input("\nPress Enter to close...")
                return 1

        site_packages_str = str(site_packages)
        if site_packages_str in sys.path:
            sys.path.remove(site_packages_str)
        sys.path.insert(0, site_packages_str)
        print(f"✓ Added venv to sys.path")

    except Exception as e:
        print(f"❌ Error setting up venv: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to close...")
        return 1

    # Set backend environment variables for Direct3D-S2
    os.environ['SPARSE_BACKEND'] = 'torchsparse'
    os.environ['SPARSE_ATTN_BACKEND'] = 'flash_attn'
    os.environ['ATTN_BACKEND'] = 'flash_attn'

    print("-" * 80)

    # Import dependencies
    try:
        import torch
        import trimesh
        from PIL import Image
        
        # Enable cuDNN benchmark for speed
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  VRAM: {gpu_mem:.2f} GB")
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Please run: pip install torch trimesh pillow")
        return 1

    # Initialize pipeline
    print("-" * 80)
    print("Initializing Direct3D-S2 pipeline...")
    print("-" * 80)

    try:
        from pipeline_manager import PipelineManager

        manager = PipelineManager()
        # skip_cleanup=True because we're running from console, not Blender
        # (torch is already loaded cleanly from venv, no need to clear/reload)
        if not manager.initialize(use_cuda=torch.cuda.is_available(), skip_cleanup=True):
            print("Error: Failed to initialize pipeline")
            return 1

        print("✓ Pipeline initialized")

    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load input mesh
    print("-" * 80)
    print("Loading input mesh...")
    try:
        input_mesh = trimesh.load(args.input_mesh, force='mesh')
        print(f"✓ Loaded mesh: {len(input_mesh.vertices)} vertices, {len(input_mesh.faces)} faces")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return 1

    # Load reference image
    print("Loading reference image...")
    try:
        image = Image.open(args.reference_image).convert("RGB")
        print(f"✓ Loaded image: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return 1

    # Run upscale
    print("-" * 80)
    print(f"Starting upscale at {args.resolution}³ resolution...")
    print("⏳ This may take several minutes...")
    print("-" * 80)

    try:
        # Ensure Direct3D pipeline is loaded
        if not manager._ensure_direct3d_pipeline_loaded():
            print("Error: Failed to load Direct3D-S2 pipeline")
            return 1

        # Set seed
        if args.seed >= 0:
            generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
            generator.manual_seed(args.seed)
        else:
            generator = None

        # Run upscale
        with torch.no_grad():
            outputs = manager.direct3d_pipeline.upscale(
                image=image,
                mesh=input_mesh,
                resolution=args.resolution,
                sparse_sampler_params={
                    'num_inference_steps': args.steps,
                    'guidance_scale': args.guidance
                },
                generator=generator,
                remesh=True,
                simplify_ratio=simplify_ratio,
                remove_interior=True
            )

        result_mesh = outputs['mesh']
        print(f"✓ Upscale complete!")
        print(f"  Output mesh: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} faces")

    except Exception as e:
        print(f"Error during upscale: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save output
    print("-" * 80)
    print(f"Saving to: {output_path}")
    try:
        result_mesh.export(output_path)
        print(f"✓ Mesh saved successfully!")
    except Exception as e:
        print(f"Error saving mesh: {e}")
        return 1

    print("=" * 80)
    print("✓ UPSCALE COMPLETE!")
    print("=" * 80)

    return 0

if __name__ == '__main__':
    sys.exit(main())
