"""
Pipeline manager for TRELLIS integration
Handles model loading and 3D generation
"""

import os
import sys
from typing import Optional, Dict, Any, List, Callable
from PIL import Image

# Handle both standalone and package imports
# When running in console, __package__ is None; when in Blender addon, it's set
try:
    # Try relative import (when running as part of addon)
    from . import dependency_installer
except (ImportError, ValueError):
    # Fall back to absolute import (when running standalone)
    import dependency_installer

# Global progress tracking state
_generation_progress = {
    'active': False,
    'stage': '',
    'step': 0,
    'total_steps': 0,
    'message': ''
}


def get_generation_progress() -> Dict[str, Any]:
    """Get current generation progress"""
    return _generation_progress.copy()


def update_progress(stage: str = None, step: int = None, total_steps: int = None, message: str = None):
    """Update generation progress"""
    global _generation_progress
    if stage is not None:
        _generation_progress['stage'] = stage
    if step is not None:
        _generation_progress['step'] = step
    if total_steps is not None:
        _generation_progress['total_steps'] = total_steps
    if message is not None:
        _generation_progress['message'] = message

    # Print progress to console
    if _generation_progress['total_steps'] > 0:
        percentage = (_generation_progress['step'] / _generation_progress['total_steps']) * 100
        print(f"[{_generation_progress['stage']}] {_generation_progress['step']}/{_generation_progress['total_steps']} ({percentage:.1f}%) - {_generation_progress['message']}")
    else:
        print(f"[{_generation_progress['stage']}] {_generation_progress['message']}")


class PipelineManager:
    """Manages TRELLIS pipelines and generation"""

    def __init__(self):
        self.image_pipeline = None
        self.text_pipeline = None
        self.initialized = False
        self.lazy_loading = True  # Enable lazy loading by default
        self._image_pipeline_loaded = False
        self._text_pipeline_loaded = False
        self.use_cuda = True  # Default to using CUDA if available

    def initialize(self, use_cuda: bool = True) -> bool:
        """
        Initialize TRELLIS pipelines

        Args:
            use_cuda: Whether to use CUDA acceleration

        Returns:
            bool: True if initialization successful
        """
        # Check if already initialized
        if self.initialized:
            return True

        try:
            # Re-run venv setup to ensure it's in sys.path
            # dependency_installer already imported at module level
            venv_path = dependency_installer.get_venv_path()

            if not venv_path.exists():
                raise RuntimeError(
                    "Virtual environment not found!\n"
                    f"Expected location: {venv_path}\n"
                    "Please click 'Install Dependencies' first."
                )

            # Get venv site-packages
            if os.name == 'nt':  # Windows
                site_packages = venv_path / 'Lib' / 'site-packages'
            else:  # Linux/Mac
                lib_path = venv_path / 'lib'
                python_dirs = [d for d in lib_path.iterdir() if d.name.startswith('python3.')]
                if python_dirs:
                    site_packages = python_dirs[0] / 'site-packages'
                else:
                    raise RuntimeError(f"Could not find site-packages in venv at {venv_path}")

            site_packages_str = str(site_packages)

            # Ensure venv is at the BEGINNING of sys.path
            if site_packages_str in sys.path:
                sys.path.remove(site_packages_str)
            sys.path.insert(0, site_packages_str)

            # CRITICAL: Clear any torch modules from Blender's Python
            torch_modules_before = [key for key in sys.modules.keys() if key.startswith('torch')]
            if torch_modules_before:
                for module in torch_modules_before:
                    del sys.modules[module]

            # Move Blender's site-packages to END of sys.path
            blender_site_packages = []
            for path in list(sys.path):
                if 'blender' in path.lower() and 'site-packages' in path.lower():
                    blender_site_packages.append(path)

            for path in blender_site_packages:
                if path in sys.path:
                    sys.path.remove(path)
                    sys.path.append(path)

            # Verify torch can be imported from venv
            try:
                import torch
                torch_file = torch.__file__
                torch_version = torch.__version__
                print(f"✓ PyTorch {torch_version} loaded from:")
                print(f"  {torch_file}")

                # Check if it's from venv
                if 'TRELLIS_venv' in torch_file or str(venv_path) in torch_file:
                    print("✓ PyTorch is from VENV (correct!)")
                elif 'blender' in torch_file.lower():
                    raise RuntimeError(
                        f"ERROR: PyTorch is loading from BLENDER'S Python, not venv!\n"
                        f"  Loaded from: {torch_file}\n"
                        f"  Expected from: {site_packages_str}\n"
                        f"\nThis means the venv isolation is broken."
                    )

                # Check CUDA availability
                cuda_available = torch.cuda.is_available()
                print(f"✓ CUDA available: {cuda_available}")
                if cuda_available:
                    print(f"  GPU: {torch.cuda.get_device_name(0)}")

            except ImportError as e:
                raise RuntimeError(
                    f"PyTorch not found in venv!\n"
                    f"  Searched in: {site_packages_str}\n"
                    f"  Error: {e}\n"
                    f"\nPlease click 'Install Dependencies' to install PyTorch."
                )

            # Set ALL attention backends to use xformers for best performance
            os.environ['ATTN_BACKEND'] = 'xformers'
            os.environ['SPARSE_ATTN_BACKEND'] = 'xformers'
            os.environ['SPCONV_ALGO'] = 'native'
            
            # Enable PyTorch performance optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Check GPU memory and enable smart offloading for low VRAM (if user enables it)
            import torch

            # Try to get user preference from Blender (only available when running inside Blender)
            enable_offload_pref = True  # Default to True for console execution (better for 6GB GPUs)
            try:
                import bpy
                # Get user preference for offloading
                preferences = bpy.context.preferences.addons[__package__].preferences
                enable_offload_pref = preferences.enable_smart_offload
            except (ImportError, KeyError, AttributeError):
                # bpy not available (console execution) or preferences not accessible
                # Default to True for safety on low VRAM systems
                pass

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

                # Only auto-enable offloading if user preference is enabled AND VRAM is low
                if enable_offload_pref and gpu_memory <= 8:
                    # Enable offloading mode silently
                    os.environ['TRELLIS_ENABLE_OFFLOAD'] = '1'
                elif not enable_offload_pref and gpu_memory <= 8:
                    os.environ['TRELLIS_ENABLE_OFFLOAD'] = '0'
                else:
                    os.environ['TRELLIS_ENABLE_OFFLOAD'] = '0'

            # Import TRELLIS modules (bundled with addon)
            try:
                # Try relative import (when running as part of addon)
                from .trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
            except (ImportError, ValueError):
                # Fall back to absolute import (when running standalone)
                from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline

            # Set HuggingFace timeout (default is no timeout!)
            # This prevents indefinite hangs on slow/unreliable connections
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minute timeout
            
            # Enable fast transfer if available
            try:
                import hf_transfer
                os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
                print("✓ Enabled HF_TRANSFER for faster downloads")
            except ImportError:
                pass

            # LIGHTWEIGHT INITIALIZATION: Don't load pipelines yet!
            # Let them load lazily when actually needed to avoid crashing Blender
            
            # Just mark as initialized - pipelines will load when needed
            self.initialized = True
            self.use_cuda = use_cuda  # Store for lazy loading

            return True

        except Exception as e:
            print(f"Error initializing TRELLIS: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _ensure_image_pipeline_loaded(self) -> bool:
        """Lazily load image pipeline if not already loaded"""
        if self._image_pipeline_loaded and self.image_pipeline is not None:
            return True

        if not self.initialized:
            print("Error: Cannot load image pipeline - manager not initialized")
            return False

        try:
            import os
            try:
                from .trellis.pipelines import TrellisImageTo3DPipeline
            except (ImportError, ValueError):
                from trellis.pipelines import TrellisImageTo3DPipeline
            import torch

            # Check if offloading is enabled
            offload_enabled = os.environ.get('TRELLIS_ENABLE_OFFLOAD', '0') == '1'

            # Load pipeline
            self.image_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-image-large"
            )

            # If offloading enabled, keep models on CPU and move to GPU only when needed
            if offload_enabled:
                # Ensure all models are on CPU
                for name, model in self.image_pipeline.models.items():
                    model.cpu()
                self._image_pipeline_loaded = True
                return True

            # Standard mode: move to GPU if available
            if self.use_cuda:
                import torch
                if torch.cuda.is_available():
                    self.image_pipeline.cuda()
                    
                    # Set models to eval mode and disable gradient computation (inference optimization)
                    for name, model in self.image_pipeline.models.items():
                        model.eval()
                        for param in model.parameters():
                            param.requires_grad = False
                    
                    # Enable channels_last memory format for faster convolutions on modern GPUs
                    try:
                        for name, model in self.image_pipeline.models.items():
                            model = model.to(memory_format=torch.channels_last)
                    except:
                        pass
                    
                    # Try to compile models with torch.compile for maximum performance (PyTorch 2.0+)
                    try:
                        if hasattr(torch, 'compile'):
                            print("✓ torch.compile available - compiling models for faster inference...")
                            # Only compile flow models as they are used repeatedly
                            compiled_count = 0
                            for name, model in self.image_pipeline.models.items():
                                if 'flow' in name.lower():
                                    try:
                                        self.image_pipeline.models[name] = torch.compile(
                                            model,
                                            mode='reduce-overhead',  # Best for repeated inference
                                            fullgraph=False,  # Allow graph breaks for compatibility
                                        )
                                        compiled_count += 1
                                    except:
                                        pass
                            if compiled_count > 0:
                                print(f"✓ Compiled {compiled_count} models with torch.compile (first run will be slower)")
                    except Exception as e:
                        pass  # Silently skip if compilation fails
                else:
                    print("✓ Pipeline on CPU (CUDA not available)")
            else:
                print("✓ Pipeline on CPU (CPU mode forced)")

            self._image_pipeline_loaded = True
            print("✓ Image pipeline ready!\n")
            return True

        except Exception as e:
            print(f"Error loading image pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _ensure_text_pipeline_loaded(self) -> bool:
        """Lazily load text pipeline if not already loaded"""
        if self._text_pipeline_loaded and self.text_pipeline is not None:
            return True

        if not self.initialized:
            print("Error: Cannot load text pipeline - manager not initialized")
            return False

        try:
            import os
            try:
                from .trellis.pipelines import TrellisTextTo3DPipeline
            except (ImportError, ValueError):
                from trellis.pipelines import TrellisTextTo3DPipeline
            import torch

            # Check if offloading is enabled
            offload_enabled = os.environ.get('TRELLIS_ENABLE_OFFLOAD', '0') == '1'

            # Load pipeline
            self.text_pipeline = TrellisTextTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-text-xlarge"
            )

            # If offloading enabled, keep models on CPU and move to GPU only when needed
            if offload_enabled:
                # Ensure all models are on CPU
                for name, model in self.text_pipeline.models.items():
                    model.cpu()
                # Also move text cond model to CPU
                if hasattr(self.text_pipeline, 'text_cond_model'):
                    self.text_pipeline.text_cond_model['model'].cpu()
                self._text_pipeline_loaded = True
                return True

            # Standard mode: move to GPU if available
            if self.use_cuda:
                import torch
                if torch.cuda.is_available():
                    print("Moving pipeline to GPU...")
                    self.text_pipeline.cuda()
                    print("✓ Pipeline on GPU")
                    
                    # Set models to eval mode and disable gradient computation (inference optimization)
                    for name, model in self.text_pipeline.models.items():
                        model.eval()
                        for param in model.parameters():
                            param.requires_grad = False
                    if hasattr(self.text_pipeline, 'text_cond_model'):
                        self.text_pipeline.text_cond_model['model'].eval()
                        for param in self.text_pipeline.text_cond_model['model'].parameters():
                            param.requires_grad = False
                    print("✓ Models set to inference mode (no gradient tracking)")
                    
                    # Enable channels_last memory format for faster convolutions
                    try:
                        for name, model in self.text_pipeline.models.items():
                            model = model.to(memory_format=torch.channels_last)
                        print("✓ Enabled channels_last memory format (faster convolutions)")
                    except:
                        pass
                    
                    # Try to compile models with torch.compile for maximum performance (PyTorch 2.0+)
                    try:
                        if hasattr(torch, 'compile'):
                            print("✓ torch.compile available - compiling models for faster inference...")
                            compiled_count = 0
                            for name, model in self.text_pipeline.models.items():
                                if 'flow' in name.lower():
                                    try:
                                        self.text_pipeline.models[name] = torch.compile(
                                            model,
                                            mode='reduce-overhead',
                                            fullgraph=False,
                                        )
                                        compiled_count += 1
                                    except:
                                        pass
                            if compiled_count > 0:
                                print(f"✓ Compiled {compiled_count} models with torch.compile (first run will be slower)")
                    except:
                        pass
                else:
                    print("✓ Pipeline on CPU (CUDA not available)")
            else:
                print("✓ Pipeline on CPU (CPU mode forced)")

            self._text_pipeline_loaded = True
            print("✓ Text pipeline ready!\n")
            return True

        except Exception as e:
            print(f"Error loading text pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_from_image(
        self,
        image_path: str,
        seed: int = 42,
        sparse_steps: int = 12,
        sparse_cfg: float = 7.5,
        slat_steps: int = 12,
        slat_cfg: float = 3.0,
        preprocess: bool = True,
        formats: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate 3D from image

        Args:
            image_path: Path to input image
            seed: Random seed
            sparse_steps: Steps for sparse structure sampling
            sparse_cfg: CFG strength for sparse structure
            slat_steps: Steps for SLAT sampling
            slat_cfg: CFG strength for SLAT
            preprocess: Whether to preprocess image (background removal)
            formats: Output formats (mesh, gaussian, radiance_field)

        Returns:
            Dictionary with generated outputs or None on error
        """
        # Lazy load image pipeline if needed
        if not self._ensure_image_pipeline_loaded():
            print("Error: Failed to load image pipeline")
            return None

        try:
            import torch
            # Mark generation as active
            global _generation_progress
            _generation_progress['active'] = True

            # CRITICAL: Clear cache before starting generation (prevents OOM from previous runs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ Cleared CUDA cache before generation")

            # Load image
            update_progress(stage="Preprocessing", step=0, total_steps=100, message="Loading image...")
            image = Image.open(image_path).convert("RGBA")

            if formats is None:
                formats = ['mesh', 'gaussian']

            # Generate
            update_progress(stage="Generation", step=0, total_steps=sparse_steps + slat_steps, message="Starting generation...")
            print(f"Generating 3D from image: {image_path}")

            # Note: Progress updates during generation would require modifying TRELLIS internals
            # For now, we show start/end markers
            update_progress(stage="Sparse Structure", step=0, total_steps=sparse_steps, message="Generating structure...")

            # Use automatic mixed precision (AMP) for faster inference on modern GPUs
            # This uses FP16 where possible, with FP32 fallback for numerical stability
            import torch
            use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7  # Volta+ GPUs
            
            # Use inference_mode instead of no_grad for even faster inference (no view tracking)
            with torch.inference_mode():
                # Note: AMP disabled due to dtype mismatch in mesh extraction (scatter operations)
                # The mesh decoder uses scatter_reduce which requires exact dtype matching
                outputs = self.image_pipeline.run(
                    image,
                    seed=seed,
                    sparse_structure_sampler_params={
                        "steps": sparse_steps,
                        "cfg_strength": sparse_cfg,
                    },
                    slat_sampler_params={
                        "steps": slat_steps,
                        "cfg_strength": slat_cfg,
                    },
                    formats=formats,
                    preprocess_image=preprocess,
                )

            # CRITICAL: Clear cache after generation (prevents OOM accumulation)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ Cleared CUDA cache after generation")

            update_progress(stage="Complete", step=100, total_steps=100, message="Generation finished!")
            _generation_progress['active'] = False

            return outputs

        except Exception as e:
            print(f"Error generating from image: {e}")
            import traceback
            traceback.print_exc()
            _generation_progress['active'] = False
            # Clear cache even on error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            return None

    def generate_from_multi_image(
        self,
        image_paths: List[str],
        seed: int = 42,
        sparse_steps: int = 12,
        sparse_cfg: float = 7.5,
        slat_steps: int = 12,
        slat_cfg: float = 3.0,
        preprocess: bool = True,
        formats: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate 3D from multiple images (multi-view)

        Args:
            image_paths: List of paths to input images
            seed: Random seed
            sparse_steps: Steps for sparse structure sampling
            sparse_cfg: CFG strength for sparse structure
            slat_steps: Steps for SLAT sampling
            slat_cfg: CFG strength for SLAT
            preprocess: Whether to preprocess images (background removal)
            formats: Output formats (mesh, gaussian, radiance_field)

        Returns:
            Dictionary with generated outputs or None on error
        """
        # Lazy load image pipeline if needed
        if not self._ensure_image_pipeline_loaded():
            print("Error: Failed to load image pipeline")
            return None

        try:
            import torch
            # Mark generation as active
            global _generation_progress
            _generation_progress['active'] = True

            # CRITICAL: Clear cache before starting generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ Cleared CUDA cache before multi-image generation")

            # Load images
            update_progress(stage="Preprocessing", step=0, total_steps=100, message=f"Loading {len(image_paths)} images...")
            from PIL import Image
            images = [Image.open(path).convert("RGBA") for path in image_paths]

            if formats is None:
                formats = ['mesh', 'gaussian']

            # Generate
            update_progress(stage="Generation", step=0, total_steps=sparse_steps + slat_steps, message="Starting multi-image generation...")
            print(f"Generating 3D from {len(images)} images")

            update_progress(stage="Sparse Structure", step=0, total_steps=sparse_steps, message="Generating structure from multiple views...")

            # Use inference_mode for faster inference
            with torch.inference_mode():
                # Call run_multi_image method from the pipeline
                outputs = self.image_pipeline.run_multi_image(
                    images,
                    seed=seed,
                    sparse_structure_sampler_params={
                        "steps": sparse_steps,
                        "cfg_strength": sparse_cfg,
                    },
                    slat_sampler_params={
                        "steps": slat_steps,
                        "cfg_strength": slat_cfg,
                    },
                    formats=formats,
                    preprocess_image=preprocess,
                )

            # CRITICAL: Clear cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ Cleared CUDA cache after multi-image generation")

            update_progress(stage="Complete", step=100, total_steps=100, message="Multi-image generation finished!")
            _generation_progress['active'] = False

            return outputs

        except Exception as e:
            print(f"Error generating from multiple images: {e}")
            import traceback
            traceback.print_exc()
            _generation_progress['active'] = False
            # Clear cache even on error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            return None

    def generate_from_text(
        self,
        prompt: str,
        seed: int = 42,
        sparse_steps: int = 12,
        sparse_cfg: float = 7.5,
        slat_steps: int = 12,
        slat_cfg: float = 3.0,
        formats: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate 3D from text

        Args:
            prompt: Text prompt
            seed: Random seed
            sparse_steps: Steps for sparse structure sampling
            sparse_cfg: CFG strength for sparse structure
            slat_steps: Steps for SLAT sampling
            slat_cfg: CFG strength for SLAT
            formats: Output formats (mesh, gaussian, radiance_field)

        Returns:
            Dictionary with generated outputs or None on error
        """
        # Lazy load text pipeline if needed
        if not self._ensure_text_pipeline_loaded():
            print("Error: Failed to load text pipeline")
            return None

        try:
            import torch
            # Mark generation as active
            global _generation_progress
            _generation_progress['active'] = True

            # CRITICAL: Clear cache before starting generation (prevents OOM from previous runs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ Cleared CUDA cache before generation")

            update_progress(stage="Text Encoding", step=0, total_steps=100, message="Encoding text prompt...")

            if formats is None:
                formats = ['mesh', 'gaussian']

            # Generate
            update_progress(stage="Generation", step=0, total_steps=sparse_steps + slat_steps, message="Starting generation...")
            print(f"Generating 3D from text: {prompt}")

            update_progress(stage="Sparse Structure", step=0, total_steps=sparse_steps, message="Generating structure...")

            # Use automatic mixed precision (AMP) for faster inference on modern GPUs
            import torch
            use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7  # Volta+ GPUs
            
            # Use inference_mode instead of no_grad for even faster inference (no view tracking)
            with torch.inference_mode():
                # Note: AMP disabled for text-to-3D due to dtype mismatch in mesh extraction (scatter operations)
                # The mesh decoder uses scatter_reduce which requires exact dtype matching
                outputs = self.text_pipeline.run(
                    prompt,
                    seed=seed,
                    sparse_structure_sampler_params={
                        "steps": sparse_steps,
                        "cfg_strength": sparse_cfg,
                    },
                    slat_sampler_params={
                        "steps": slat_steps,
                        "cfg_strength": slat_cfg,
                    },
                    formats=formats,
                )

            # CRITICAL: Clear cache after generation (prevents OOM accumulation)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ Cleared CUDA cache after generation")

            update_progress(stage="Complete", step=100, total_steps=100, message="Generation finished!")
            _generation_progress['active'] = False

            return outputs

        except Exception as e:
            print(f"Error generating from text: {e}")
            import traceback
            traceback.print_exc()
            _generation_progress['active'] = False
            # Clear cache even on error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            return None

    def export_to_glb(
        self,
        outputs: Dict[str, Any],
        output_path: str,
        simplify: float = 0.95,
        texture_size: int = 1024,
    ) -> bool:
        """
        Export generated assets to GLB format

        Args:
            outputs: Dictionary from generate_from_image/text
            output_path: Path to save GLB file
            simplify: Mesh simplification ratio (0-1)
            texture_size: Texture resolution

        Returns:
            bool: True if export successful
        """
        try:
            try:
                from .trellis.utils import postprocessing_utils
            except (ImportError, ValueError):
                from trellis.utils import postprocessing_utils

            # Get first outputs
            gaussian = outputs.get('gaussian', [None])[0]
            radiance = outputs.get('radiance_field', [None])[0]
            mesh = outputs.get('mesh', [None])[0]

            if mesh is None:
                print("Error: No mesh in outputs")
                return False

            # Use gaussian or radiance for appearance
            appearance = gaussian if gaussian is not None else radiance

            if appearance is None:
                print("Warning: No appearance data, exporting mesh only")

            # Move outputs to GPU for fast texture baking
            # (texture baking is GPU-accelerated optimization)
            import torch
            if torch.cuda.is_available():
                print("Moving outputs to GPU for texture baking...")

                # Move Gaussian object's internal tensors to GPU
                if appearance is not None and hasattr(appearance, '__class__') and appearance.__class__.__name__ == 'Gaussian':
                    print("  → Moving Gaussian tensors to GPU...")
                    appearance.device = 'cuda'
                    # Move all internal tensor attributes
                    for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', 'aabb']:
                        tensor = getattr(appearance, attr, None)
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            setattr(appearance, attr, tensor.cuda())
                    # Move bias tensors
                    for attr in ['scale_bias', 'rots_bias', 'opacity_bias']:
                        tensor = getattr(appearance, attr, None)
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            setattr(appearance, attr, tensor.cuda())
                # Move simple tensors or objects with .cuda() method
                elif appearance is not None and hasattr(appearance, 'cuda') and callable(getattr(appearance, 'cuda')):
                    appearance = appearance.cuda()

                # Move mesh if it has .cuda() method
                if mesh is not None and hasattr(mesh, 'cuda') and callable(getattr(mesh, 'cuda')):
                    mesh = mesh.cuda()

            # Export to GLB
            print(f"Exporting to GLB: {output_path}")
            
            # If no appearance data (mesh-only mode), export without texture
            if appearance is None:
                print("Mesh-only mode: Exporting without texture baking")
                # Create a simple mesh export without appearance
                from trellis.representations import MeshExtractResult
                from trimesh.exchange.gltf import export_glb as trimesh_export
                import trimesh
                
                # Convert mesh to trimesh format
                if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                    vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
                    faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces
                    
                    # Apply simplification if requested
                    if simplify < 1.0:
                        print(f"Simplifying mesh to {simplify*100:.1f}% faces...")
                        try:
                            import pyfqmr
                            mesh_simplifier = pyfqmr.Simplify()
                            mesh_simplifier.setMesh(vertices, faces)
                            target_count = int(len(faces) * simplify)
                            mesh_simplifier.simplify_mesh(target_count=target_count, aggressiveness=7, preserve_border=True, verbose=False)
                            vertices, faces, _ = mesh_simplifier.getMesh()
                            print(f"Simplified to {len(faces)} faces")
                        except Exception as e:
                            print(f"Simplification failed, using original mesh: {e}")
                    
                    # Create trimesh object
                    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    
                    # Export to GLB
                    tri_mesh.export(output_path, file_type='glb')
                    print("GLB export successful (mesh-only)!")
                    return True
                else:
                    print("Error: Mesh format not recognized")
                    return False
            else:
                # Normal texture baking export
                glb_mesh = postprocessing_utils.to_glb(
                    appearance,
                    mesh,
                    simplify=simplify,
                    texture_size=texture_size,
                    fill_holes=True,
                )
                glb_mesh.export(output_path)
                print("GLB export successful!")
                return True

        except Exception as e:
            print(f"Error exporting to GLB: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Clean up GPU memory and free pipelines"""
        import gc
        try:
            import torch

            # Free image pipeline if loaded
            if self._image_pipeline_loaded and self.image_pipeline is not None:
                print("Freeing image pipeline from memory...")
                # Move to CPU first to free GPU memory
                if torch.cuda.is_available():
                    try:
                        self.image_pipeline.cpu()
                    except:
                        pass
                del self.image_pipeline
                self.image_pipeline = None
                self._image_pipeline_loaded = False
                print("✓ Image pipeline freed")

            # Free text pipeline if loaded
            if self._text_pipeline_loaded and self.text_pipeline is not None:
                print("Freeing text pipeline from memory...")
                # Move to CPU first to free GPU memory
                if torch.cuda.is_available():
                    try:
                        self.text_pipeline.cpu()
                    except:
                        pass
                del self.text_pipeline
                self.text_pipeline = None
                self._text_pipeline_loaded = False
                print("✓ Text pipeline freed")

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ GPU memory cache cleared")

            print("✓ Full cleanup completed - all pipelines freed")
        except Exception as e:
            # Log error but don't crash
            print(f"⚠ Warning: Failed to cleanup GPU memory: {e}")
            import traceback
            traceback.print_exc()


# Global pipeline manager instance
_pipeline_manager = None


def get_pipeline_manager() -> PipelineManager:
    """Get or create global pipeline manager"""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager
