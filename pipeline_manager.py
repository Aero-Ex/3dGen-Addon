"""
Pipeline manager for TRELLIS integration
Handles model loading and 3D generation
"""

import os
import sys
from pathlib import Path
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
        self.direct3d_pipeline = None
        self._direct3d_loaded = False
        self._direct3d_device = 'cpu'


    @staticmethod
    def _addon_root() -> Path:
        return Path(__file__).resolve().parent

    def _ensure_repo_on_path(self, folder_name: str) -> Path:
        repo_path = self._addon_root() / folder_name
        repo_str = str(repo_path)
        if repo_path.exists() and repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        return repo_path

    def initialize(self, use_cuda: bool = True, skip_cleanup: bool = False) -> bool:
        """
        Initialize TRELLIS pipelines

        Args:
            use_cuda: Whether to use CUDA acceleration
            skip_cleanup: Skip Blender-specific cleanup (set True when running from console)

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

            # Only do cleanup when running from Blender (not from console scripts)
            if not skip_cleanup:
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

            # CRITICAL: Set PyTorch memory config BEFORE importing torch
            # This MUST be set before import for it to take effect
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

            # Note: Backend settings (ATTN_BACKEND, SPARSE_ATTN_BACKEND) are now set
            # in generate_in_console.py BEFORE importing this manager, based on selected pipeline.
            # Direct3D uses flash_attn, TRELLIS uses xformers.
            # DO NOT override those settings here!
            
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
                print(f"✓ Detected GPU memory: {gpu_memory:.2f} GB")

                # Force offloading for < 12GB VRAM to be safe
                if enable_offload_pref and gpu_memory < 12:
                    print("✓ Low VRAM detected (< 12GB) - Enabling Ultimate Smart Offloading")
                    print("   → PyTorch expandable segments enabled (set at startup)")
                    os.environ['TRELLIS_ENABLE_OFFLOAD'] = '1'
                    os.environ['DIRECT3D_ENABLE_OFFLOAD'] = '1'

                    # Enable ULTRA offloading for GPUs < 10GB (for 1024 resolution refiner)
                    if gpu_memory < 10:
                        print("🚀 ULTRA-AGGRESSIVE OFFLOADING ENABLED for < 10GB VRAM")
                        print("   → Will load/unload refiner models per-patch (slower but fits in VRAM)")
                        os.environ['DIRECT3D_ULTRA_OFFLOAD'] = '1'
                    else:
                        os.environ['DIRECT3D_ULTRA_OFFLOAD'] = '0'
                elif not enable_offload_pref and gpu_memory <= 8:
                    # Even if user disabled it, force it for very low VRAM to prevent crashes
                    print("⚠ VRAM is critical (< 8GB) - Forcing Smart Offloading despite preference")
                    print("   → PyTorch expandable segments enabled (set at startup)")
                    os.environ['TRELLIS_ENABLE_OFFLOAD'] = '1'
                    os.environ['DIRECT3D_ENABLE_OFFLOAD'] = '1'

                    os.environ['DIRECT3D_ULTRA_OFFLOAD'] = '1'  # Force ultra offload for critical VRAM
                else:
                    os.environ['TRELLIS_ENABLE_OFFLOAD'] = '0'
                    os.environ['DIRECT3D_ENABLE_OFFLOAD'] = '0'

                    os.environ['DIRECT3D_ULTRA_OFFLOAD'] = '0'

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

        # Unload other pipelines to free memory
        self._unload_all_pipelines(except_pipeline='image')

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

        # Unload other pipelines to free memory
        self._unload_all_pipelines(except_pipeline='text')

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

    @staticmethod
    def _wrap_mesh_output(mesh_obj):
        if mesh_obj is None:
            return None
        return {
            'mesh': [mesh_obj],
            'gaussian': [None],
            'radiance_field': [None],
        }

    @staticmethod
    def _detect_gpu_memory_gb() -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass
        return 0.0

    def _select_direct3d_resolution(self) -> int:
        override = os.environ.get('DIRECT3D_S2_RESOLUTION')
        if override in {'512', '1024'}:
            return int(override)
        gpu_mem = self._detect_gpu_memory_gb()
        return 1024 if gpu_mem >= 24 else 512

    def _ensure_direct3d_pipeline_loaded(self) -> bool:
        if self._direct3d_loaded and self.direct3d_pipeline is not None:
            return True

        # Unload other pipelines to free memory
        self._unload_all_pipelines(except_pipeline='direct3d')

        try:
            repo_path = self._ensure_repo_on_path("Direct3D-S2")
            if not repo_path.exists():
                print(f"Direct3D-S2 repository not found at {repo_path}")
                return False

            import torch
            
            # CRITICAL: Set environment variables BEFORE importing Direct3D modules
            # The modules read these on import via __from_env()
            os.environ['SPARSE_BACKEND'] = 'torchsparse'
            os.environ['SPARSE_ATTN_BACKEND'] = 'flash_attn'
            os.environ['ATTN_BACKEND'] = 'flash_attn'
            print("[BACKEND] Set Direct3D environment: SPARSE_BACKEND=torchsparse, ATTN_BACKEND=flash_attn")
            
            from direct3d_s2.pipeline import Direct3DS2Pipeline

            model_id = os.environ.get('DIRECT3D_S2_MODEL', 'wushuang98/Direct3D-S2')
            subfolder = os.environ.get('DIRECT3D_S2_SUBFOLDER', 'direct3d-s2-v-1-1')

            print(f"Loading Direct3D-S2 pipeline ({model_id}/{subfolder})... This may take several minutes on first run.")
            self.direct3d_pipeline = Direct3DS2Pipeline.from_pretrained(model_id, subfolder=subfolder)
            device = 'cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu'
            self.direct3d_pipeline.to(device)
            self._direct3d_device = device
            self._direct3d_loaded = True
            print(f"✓ Direct3D-S2 pipeline ready on {device}")
            return True
        except Exception as exc:
            print(f"Error loading Direct3D-S2 pipeline: {exc}")
            if "torchsparse" in str(exc) or "No module named 'torchsparse'" in str(exc):
                print("\n❌ CRITICAL: torchsparse is missing!")
                print("Direct3D-S2 requires torchsparse which is difficult to install on Windows.")
                print("Please try installing it manually or check the addon documentation.")
            import traceback
            traceback.print_exc()
            return False

    def _generate_direct3d_from_image(
        self,
        image_path: str,
        seed: int,
        mesh_simplify: float,
        steps: int = 15,
        guidance_scale: float = 7.0,
    ) -> Optional[Dict[str, Any]]:
        if not self._ensure_direct3d_pipeline_loaded():
            return None

        try:
            import torch

            generator = torch.Generator(device=self._direct3d_device)
            generator.manual_seed(seed)

            sdf_resolution = self._select_direct3d_resolution()
            print(f"Direct3D-S2: running inference at {sdf_resolution}^3 resolution")
            print(f"  Steps: {steps}, Guidance: {guidance_scale}")

            # Note: Direct3D pipeline signature uses num_inference_steps and guidance_scale
            # These go into the sampler_params dictionaries
            outputs = self.direct3d_pipeline(
                image=image_path,
                sdf_resolution=sdf_resolution,
                dense_sampler_params={'num_inference_steps': steps, 'guidance_scale': guidance_scale},
                sparse_512_sampler_params={'num_inference_steps': steps, 'guidance_scale': guidance_scale},
                sparse_1024_sampler_params={'num_inference_steps': steps, 'guidance_scale': guidance_scale},
                generator=generator,
                remesh=True,
                simplify_ratio=mesh_simplify,
            )

            mesh = outputs.get('mesh') if isinstance(outputs, dict) else outputs
            return self._wrap_mesh_output(mesh)

        except Exception as exc:
            print(f"Direct3D-S2 generation failed: {exc}")
            import traceback
            traceback.print_exc()
            return None

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
        pipeline_key: str = 'TRELLIS',
        **kwargs,
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
            pipeline_key: Key of the pipeline to use
            **kwargs: Additional arguments for specific pipelines

        Returns:
            Dictionary with generated outputs or None on error
        """
        pipeline_key = (pipeline_key or 'TRELLIS').upper()

        if pipeline_key == 'DIRECT3D':
            return self._generate_direct3d_from_image(
                image_path=image_path,
                seed=seed,
                mesh_simplify=kwargs.get('simplify_mesh', 0.95),
                steps=kwargs.get('direct3d_steps', 15),
                guidance_scale=kwargs.get('direct3d_guidance', 7.0),
            )


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
            image = Image.open(image_path).convert("RGB")

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
        pipeline_key: str = 'TRELLIS',
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate 3D from multiple images (multi-view)
        """
        pipeline_key = (pipeline_key or 'TRELLIS').upper()


            
        if pipeline_key != 'TRELLIS':
            print(f"Pipeline {pipeline_key} does not support multi-image mode yet")
            return None

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
            images = [Image.open(path).convert("RGB") for path in image_paths]

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
        pipeline_key: str = 'TRELLIS',
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
        pipeline_key = (pipeline_key or 'TRELLIS').upper()
        if pipeline_key != 'TRELLIS':
            print(f"Pipeline {pipeline_key} does not support text-to-3D generation")
            return None

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

    def _ensure_direct3d_loaded(self) -> bool:
        """Lazily load Direct3D-S2 pipeline if not already loaded"""
        if hasattr(self, 'direct3d_pipeline') and self.direct3d_pipeline is not None:
            return True

        try:
            print("Loading Direct3D-S2 pipeline...")
            import sys
            import os
            
            # Add Direct3D-S2 to path
            d3d_path = os.path.join(os.path.dirname(__file__), 'Direct3D-S2')
            if d3d_path not in sys.path:
                sys.path.append(d3d_path)
                
            from direct3d_s2.pipeline import Direct3DS2Pipeline
            
            # Load pipeline
            self.direct3d_pipeline = Direct3DS2Pipeline.from_pretrained(
                "Direct3D-S2/Direct3D-S2-1.0",
                device=self.device
            )
            
            return True
        except Exception as e:
            print(f"Error loading Direct3D-S2 pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _upscale_direct3d_mesh(self, image_path: str, mesh_obj, seed: int, simplify_ratio: float, resolution: int = 512) -> dict:
        """
        Upscale an existing mesh using Direct3D-S2 stage 3.
        """
        if not self._ensure_direct3d_loaded():
            return {'error': "Failed to load Direct3D-S2 pipeline"}

        try:
            import torch
            import numpy as np
            from PIL import Image
            import trimesh
            import bpy
            import bmesh

            # Set seed
            if seed >= 0:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None

            # Load image
            print(f"Loading reference image: {image_path}")
            image = Image.open(image_path).convert("RGB")

            # Extract mesh data from Blender object
            print(f"Extracting mesh data from {mesh_obj.name}...")
            
            # Ensure we have a mesh with modifiers applied
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = mesh_obj.evaluated_get(depsgraph)
            mesh_data = eval_obj.to_mesh()
            
            # Get vertices and faces
            verts = np.array([v.co for v in mesh_data.vertices])
            
            # Handle faces (triangulate if needed)
            bm = bmesh.new()
            bm.from_mesh(mesh_data)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            
            bm.faces.ensure_lookup_table()
            faces = np.array([[v.index for v in f.verts] for f in bm.faces])
            
            bm.free()
            eval_obj.to_mesh_clear()
            
            # Create trimesh object
            input_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            print(f"Input mesh has {len(verts)} vertices and {len(faces)} faces")

            # Run upscale
            print(f"Starting Direct3D-S2 upscale at {resolution}³ resolution...")
            with torch.no_grad():
                outputs = self.direct3d_pipeline.upscale(
                    image=image,
                    mesh=input_mesh,
                    resolution=resolution,  # Pass resolution (512 or 1024)
                    generator=generator,
                    remesh=True,
                    simplify_ratio=simplify_ratio,
                    remove_interior=True
                )

            # Process result
            result_mesh = outputs['mesh']
            
            # Convert back to Blender format
            new_verts = result_mesh.vertices
            new_faces = result_mesh.faces
            
            return {
                'verts': new_verts,
                'faces': new_faces,
                'message': "Upscaling complete"
            }

        except Exception as e:
            print(f"Error during Direct3D upscaling: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def export_to_glb(
        self,
        outputs: Dict[str, Any],
        output_path: str,
        simplify: float = 0.95,
        texture_size: int = 1024,
        export_mode: str = 'MESH_TEXTURE',
    ) -> bool:
        """
        Export generated assets to GLB format

        Args:
            outputs: Dictionary from generate_from_image/text
            output_path: Path to save GLB file
            simplify: Mesh simplification ratio (0-1)
            texture_size: Texture resolution
            export_mode: Export mode ('MESH_ONLY' or 'MESH_TEXTURE')

        Returns:
            bool: True if export successful
        """
        try:
            try:
                from .trellis.utils import postprocessing_utils
            except (ImportError, ValueError):
                from trellis.utils import postprocessing_utils

            # Get first outputs
            gaussian_field = outputs.get('gaussian') if outputs else None
            radiance_field = outputs.get('radiance_field') if outputs else None
            mesh_field = outputs.get('mesh') if outputs else None

            gaussian = gaussian_field[0] if isinstance(gaussian_field, (list, tuple)) else gaussian_field
            radiance = radiance_field[0] if isinstance(radiance_field, (list, tuple)) else radiance_field
            mesh = mesh_field[0] if isinstance(mesh_field, (list, tuple)) else mesh_field

            if mesh is None:
                print("Error: No mesh in outputs")
                return False

            mesh_module = mesh.__class__.__module__.lower()
            if 'trimesh' in mesh_module or hasattr(mesh, 'export') and not hasattr(mesh, 'to_glb'):
                return self._export_trimesh_mesh(mesh, output_path)

            # MESH_ONLY mode: Skip texture baking entirely
            if export_mode == 'MESH_ONLY':
                print("Export mode: MESH_ONLY (skipping texture generation)")
                return self._export_mesh_only(mesh, output_path, simplify)

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
            glb_mesh = postprocessing_utils.to_glb(
                appearance,
                mesh,
                simplify=simplify,
                texture_size=texture_size,  # Use default (1024) for full quality
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

    @staticmethod
    def _export_trimesh_mesh(mesh_obj, output_path: str) -> bool:
        try:
            mesh_obj.export(output_path)
            print(f"✓ Exported GLB via trimesh to {output_path}")
            return True
        except Exception as exc:
            print(f"Error exporting trimesh object: {exc}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def _export_mesh_only(mesh_obj, output_path: str, simplify: float = 0.95) -> bool:
        """Export mesh without texture baking (MESH_ONLY mode)"""
        try:
            import trimesh
            import numpy as np
            try:
                from .trellis.utils import postprocessing_utils
            except (ImportError, ValueError):
                from trellis.utils import postprocessing_utils

            # Get mesh data
            vertices = mesh_obj.vertices.cpu().numpy()
            faces = mesh_obj.faces.cpu().numpy()

            print(f"Original mesh: {len(vertices)} vertices, {len(faces)} faces")

            # Apply mesh postprocessing (simplification, hole filling, etc.)
            vertices, faces = postprocessing_utils.postprocess_mesh(
                vertices, faces,
                simplify=simplify > 0,
                simplify_ratio=simplify,
                fill_holes=True,
                fill_holes_max_hole_size=0.04,
                fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),
                fill_holes_resolution=1024,
                fill_holes_num_views=1000,
                debug=False,
                verbose=True,
            )

            print(f"Post-processed mesh: {len(vertices)} vertices, {len(faces)} faces")

            # Rotate mesh (from z-up to y-up for standard GLB format)
            vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

            # Create trimesh object without texture
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Export to GLB
            mesh.export(output_path)
            print(f"✓ Exported mesh-only GLB to {output_path}")
            return True

        except Exception as exc:
            print(f"Error exporting mesh-only: {exc}")
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

            if self._direct3d_loaded and self.direct3d_pipeline is not None:
                print("Freeing Direct3D-S2 pipeline from memory...")
                try:
                    if torch.cuda.is_available():
                        self.direct3d_pipeline.to('cpu')
                except Exception:
                    pass
                self.direct3d_pipeline = None
                self._direct3d_loaded = False
                print("✓ Direct3D-S2 pipeline freed")



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

    def _unload_all_pipelines(self, except_pipeline: str = None):
        """
        Aggressively unload all pipelines except the specified one.
        
        Args:
            except_pipeline: Name of pipeline to keep ('image', 'text', 'direct3d')
        """
        import gc
        import torch
        
        unloaded_something = False
        
        # Unload Image Pipeline
        if except_pipeline != 'image' and self._image_pipeline_loaded and self.image_pipeline is not None:
            print("Offloading Image Pipeline...")
            try:
                # Move to CPU first
                if hasattr(self.image_pipeline, 'models'):
                    for name, model in self.image_pipeline.models.items():
                        model.cpu()
                del self.image_pipeline
                self.image_pipeline = None
                self._image_pipeline_loaded = False
                unloaded_something = True
            except Exception as e:
                print(f"Error unloading image pipeline: {e}")

        # Unload Text Pipeline
        if except_pipeline != 'text' and self._text_pipeline_loaded and self.text_pipeline is not None:
            print("Offloading Text Pipeline...")
            try:
                if hasattr(self.text_pipeline, 'models'):
                    for name, model in self.text_pipeline.models.items():
                        model.cpu()
                del self.text_pipeline
                self.text_pipeline = None
                self._text_pipeline_loaded = False
                unloaded_something = True
            except Exception as e:
                print(f"Error unloading text pipeline: {e}")

        # Unload Direct3D Pipeline
        if except_pipeline != 'direct3d' and self._direct3d_loaded and self.direct3d_pipeline is not None:
            print("Offloading Direct3D-S2 Pipeline...")
            try:
                # Move to CPU
                if hasattr(self.direct3d_pipeline, 'to'):
                    self.direct3d_pipeline.to('cpu')
                del self.direct3d_pipeline
                self.direct3d_pipeline = None
                self._direct3d_loaded = False
                unloaded_something = True
            except Exception as e:
                print(f"Error unloading Direct3D pipeline: {e}")



        if unloaded_something:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ Aggressive cleanup completed")

# Singleton instance
_manager = None


def get_pipeline_manager() -> PipelineManager:
    """Get or create the singleton pipeline manager instance"""
    global _manager
    if _manager is None:
        _manager = PipelineManager()
    return _manager
