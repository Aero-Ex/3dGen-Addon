import os
import torch
import numpy as np
from typing import Any
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from typing import Union, List, Optional
from direct3d_s2.modules import sparse as sp
from direct3d_s2.utils import (
    instantiate_from_config, 
    preprocess_image, 
    sort_block, 
    extract_tokens_and_coords,
    normalize_mesh,
    mesh2index,
)


class Direct3DS2Pipeline(object):

    def __init__(self, 
                 dense_vae, 
                 dense_dit,
                 sparse_vae_512,
                 sparse_dit_512,
                 sparse_vae_1024,
                 sparse_dit_1024,
                 refiner,
                 refiner_1024,
                 dense_image_encoder,
                 sparse_image_encoder,
                 dense_scheduler,
                 sparse_scheduler_512,
                 sparse_scheduler_1024,
                 dtype=torch.float16,
        ):
        self.dense_vae = dense_vae
        self.dense_dit = dense_dit
        self.sparse_vae_512 = sparse_vae_512
        self.sparse_dit_512 = sparse_dit_512
        self.sparse_vae_1024 = sparse_vae_1024
        self.sparse_dit_1024 = sparse_dit_1024
        self.refiner = refiner
        self.refiner_1024 = refiner_1024
        self.dense_image_encoder = dense_image_encoder
        self.sparse_image_encoder = sparse_image_encoder
        self.dense_scheduler = dense_scheduler
        self.sparse_scheduler_512 = sparse_scheduler_512
        self.sparse_scheduler_1024 = sparse_scheduler_1024
        self.dtype = dtype
    
    def to(self, device):
        self.device = torch.device(device)
        
        # If offloading is enabled, keep models on CPU
        if self._is_offload_enabled():
            return
            
        self.dense_vae.to(device)
        self.dense_dit.to(device)
        self.sparse_vae_512.to(device)
        self.sparse_dit_512.to(device)
        self.sparse_vae_1024.to(device)
        self.sparse_dit_1024.to(device)
        self.refiner.to(device)
        self.refiner_1024.to(device)
        self.dense_image_encoder.to(device)
        self.sparse_image_encoder.to(device)

    @classmethod
    def from_pretrained(cls, pipeline_path, subfolder="direct3d-s2-v-1-1"):
        
        if os.path.isdir(pipeline_path):
            config_path = os.path.join(pipeline_path, 'config.yaml')
            model_dense_path = os.path.join(pipeline_path, 'model_dense.ckpt')
            model_sparse_512_path = os.path.join(pipeline_path, 'model_sparse_512.ckpt')
            model_sparse_1024_path = os.path.join(pipeline_path, 'model_sparse_1024.ckpt')
            model_refiner_path = os.path.join(pipeline_path, 'model_refiner.ckpt')
            model_refiner_1024_path = os.path.join(pipeline_path, 'model_refiner_1024.ckpt')
        else:
            config_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder, 
                filename="config.yaml", 
                repo_type="model"
            )
            model_dense_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_dense.ckpt", 
                repo_type="model"
            )
            model_sparse_512_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_sparse_512.ckpt", 
                repo_type="model"
            )
            model_sparse_1024_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_sparse_1024.ckpt", 
                repo_type="model"
            )
            model_refiner_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_refiner.ckpt", 
                repo_type="model"
            )
            model_refiner_1024_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_refiner_1024.ckpt", 
                repo_type="model"
            )

        cfg = OmegaConf.load(config_path)

        def convert_weight_to_kernel(state_dict):
            """Convert weight parameters to kernel for torchsparse compatibility"""
            new_dict = {}
            for k, v in state_dict.items():
                if k.endswith('.weight') and '.conv.' in k:
                    new_dict[k[:-6] + 'kernel'] = v
                else:
                    new_dict[k] = v
            return new_dict

        state_dict_dense = torch.load(model_dense_path, map_location='cpu', weights_only=True)
        dense_vae = instantiate_from_config(cfg.dense_vae)
        dense_vae.load_state_dict(state_dict_dense["vae"], strict=True)
        dense_vae.eval()
        dense_dit = instantiate_from_config(cfg.dense_dit)
        dense_dit.load_state_dict(state_dict_dense["dit"], strict=True)
        dense_dit.eval()

        state_dict_sparse_512 = torch.load(model_sparse_512_path, map_location='cpu', weights_only=True)
        sparse_vae_512 = instantiate_from_config(cfg.sparse_vae_512)
        sparse_vae_512.load_state_dict(convert_weight_to_kernel(state_dict_sparse_512["vae"]), strict=True)
        sparse_vae_512.eval()
        sparse_dit_512 = instantiate_from_config(cfg.sparse_dit_512)
        sparse_dit_512.load_state_dict(state_dict_sparse_512["dit"], strict=True)
        sparse_dit_512.eval()

        state_dict_sparse_1024 = torch.load(model_sparse_1024_path, map_location='cpu', weights_only=True)
        sparse_vae_1024 = instantiate_from_config(cfg.sparse_vae_1024)
        sparse_vae_1024.load_state_dict(convert_weight_to_kernel(state_dict_sparse_1024["vae"]), strict=True)
        sparse_vae_1024.eval()
        sparse_dit_1024 = instantiate_from_config(cfg.sparse_dit_1024)
        sparse_dit_1024.load_state_dict(state_dict_sparse_1024["dit"], strict=True)
        sparse_dit_1024.eval()

        state_dict_refiner = torch.load(model_refiner_path, map_location='cpu', weights_only=True)
        refiner = instantiate_from_config(cfg.refiner)
        refiner.load_state_dict(state_dict_refiner["refiner"], strict=True)
        refiner.eval()

        state_dict_refiner_1024 = torch.load(model_refiner_1024_path, map_location='cpu', weights_only=True)
        refiner_1024 = instantiate_from_config(cfg.refiner_1024)
        refiner_1024.load_state_dict(state_dict_refiner_1024["refiner"], strict=True)
        refiner_1024.eval()

        dense_image_encoder = instantiate_from_config(cfg.dense_image_encoder)
        sparse_image_encoder = instantiate_from_config(cfg.sparse_image_encoder)

        dense_scheduler = instantiate_from_config(cfg.dense_scheduler)
        sparse_scheduler_512 = instantiate_from_config(cfg.sparse_scheduler_512)
        sparse_scheduler_1024 = instantiate_from_config(cfg.sparse_scheduler_1024)

        return cls(
            dense_vae=dense_vae,
            dense_dit=dense_dit,
            sparse_vae_512=sparse_vae_512,
            sparse_dit_512=sparse_dit_512,
            sparse_vae_1024=sparse_vae_1024,
            sparse_dit_1024=sparse_dit_1024,
            dense_image_encoder=dense_image_encoder,
            sparse_image_encoder=sparse_image_encoder,
            dense_scheduler=dense_scheduler,
            sparse_scheduler_512=sparse_scheduler_512,
            sparse_scheduler_1024=sparse_scheduler_1024,
            refiner=refiner,
            refiner_1024=refiner_1024,
        )

    def preprocess(self, image):
        if image.mode == 'RGBA':
            image = np.array(image)
        else:
            if getattr(self, 'birefnet_model', None) is None:
                from direct3d_s2.utils import BiRefNet
                # Initialize on CPU first if offloading enabled
                device = 'cpu' if self._is_offload_enabled() else self.device
                self.birefnet_model = BiRefNet(device)
            
            # Use context manager for offloading - BiRefNet wrapper now has .to() method
            with self._temporary_gpu_model(self.birefnet_model):
                image = self.birefnet_model.run(image)
                
        image = preprocess_image(image)
        return image

    def prepare_image(self, image: Union[str, List[str], Image.Image, List[Image.Image]]):
        if not isinstance(image, list):
            image = [image]
        if isinstance(image[0], str):
            image = [Image.open(img) for img in image]
        image = [self.preprocess(img) for img in image]
        image = torch.stack([img for img in image]).to(self.device)
        return image
    
    def encode_image(self, image: torch.Tensor, conditioner: Any, 
                     do_classifier_free_guidance: bool = True, use_mask: bool = False):
        if use_mask:
            cond = conditioner(image[:, :3], image[:, 3:])
        else:
            cond = conditioner(image[:, :3])

        if isinstance(cond, tuple):
            cond, cond_mask = cond
            cond, cond_coords = extract_tokens_and_coords(cond, cond_mask)
        else:
            cond_mask, cond_coords = None, None

        if do_classifier_free_guidance:
            uncond = torch.zeros_like(cond)
        else:
            uncond = None
        
        if cond_coords is not None:
            cond = sp.SparseTensor(cond, cond_coords.int())
            if uncond is not None:
                uncond = sp.SparseTensor(uncond, cond_coords.int())

        return cond, uncond

    def inference(
            self,
            image,
            vae,
            dit,
            conditioner,
            scheduler,
            num_inference_steps: int = 30, 
            guidance_scale: int = 7.0, 
            generator: Optional[torch.Generator] = None,
            latent_index: torch.Tensor = None,
            mode: str = 'dense', # 'dense', 'sparse512' or 'sparse1024
            remove_interior: bool = False,
            mc_threshold: float = 0.02):
        
        # Use smart offloading context
        with self._temporary_gpu_model(vae, dit, conditioner):
            do_classifier_free_guidance = guidance_scale > 0
            if mode == 'dense':
                sparse_conditions = False
            else:
                sparse_conditions = dit.sparse_conditions
            cond, uncond = self.encode_image(image, conditioner, 
                                             do_classifier_free_guidance, sparse_conditions)
            batch_size = cond.shape[0]

            if mode == 'dense':
                latent_shape = (batch_size, *dit.latent_shape)
            else:
                latent_shape = (len(latent_index), dit.out_channels)
            
            # Ensure generator is on correct device
            device = next(dit.parameters()).device
            latents = torch.randn(latent_shape, dtype=self.dtype, device=device, generator=generator)

            scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = scheduler.timesteps

            extra_step_kwargs = {
                "generator": generator
            }

            for i, t in enumerate(tqdm(timesteps, desc=f"{mode} Sampling:")):
                latent_model_input = latents
                timestep_tensor = torch.tensor([t], dtype=latent_model_input.dtype, device=device)

                if mode == 'dense':
                    x_input = latent_model_input
                elif mode in ['sparse512', 'sparse1024']:
                    x_input = sp.SparseTensor(latent_model_input, latent_index.int())

                diffusion_inputs = {
                    "x": x_input,
                    "t": timestep_tensor,
                    "cond": cond,
                }

                noise_pred_cond = dit(**diffusion_inputs)
                if mode != 'dense':
                    noise_pred_cond = noise_pred_cond.feats

                if do_classifier_free_guidance:
                    diffusion_inputs["cond"] = uncond
                    noise_pred_uncond = dit(**diffusion_inputs)
                    if mode != 'dense':
                        noise_pred_uncond = noise_pred_uncond.feats
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond
                
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            
            latents = 1. / vae.latents_scale * latents + vae.latents_shift
            
            if mode != 'dense':
                latents = sp.SparseTensor(latents, latent_index.int())

            decoder_inputs = {
                "latents": latents,
                "mc_threshold": mc_threshold,
            }
            if mode == 'dense':
                decoder_inputs['return_index'] = True
            elif remove_interior:
                decoder_inputs['return_feat'] = True
            if mode == 'sparse1024':
                decoder_inputs['voxel_resolution'] = 1024

            outputs = vae.decode_mesh(**decoder_inputs)

            if remove_interior:
                del latents, noise_pred, noise_pred_cond, noise_pred_uncond, x_input, cond, uncond
                # Also delete decoder_inputs to free references
                if 'decoder_inputs' in locals():
                    del decoder_inputs
                torch.cuda.empty_cache()

                # Handle refiner offloading
                refiner_model = self.refiner if mode == 'sparse512' else (self.refiner_1024 if mode == 'sparse1024' else None)
                if refiner_model is not None:
                    # EXTREME cleanup before refiner - move ALL other models to CPU
                    import os
                    if os.environ.get('DIRECT3D_ULTRA_OFFLOAD', '0') == '1' and mode == 'sparse1024':
                        print("🧹 EXTREME CLEANUP: Moving all models to CPU before refiner...")
                        # Move VAE and DIT to CPU to free maximum VRAM
                        vae.cpu()
                        dit.cpu()
                        conditioner.cpu()
                        # CRITICAL: Force synchronize and aggressive cleanup
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        # Also move ALL pipeline models to CPU
                        if hasattr(self, 'sparse_vae_512'):
                            self.sparse_vae_512.cpu()
                        if hasattr(self, 'sparse_dit_512'):
                            self.sparse_dit_512.cpu()
                        if hasattr(self, 'dense_vae'):
                            self.dense_vae.cpu()
                        if hasattr(self, 'dense_dit'):
                            self.dense_dit.cpu()
                        if hasattr(self, 'dense_image_encoder'):
                            self.dense_image_encoder.cpu()
                        if hasattr(self, 'sparse_image_encoder'):
                            self.sparse_image_encoder.cpu()

                        # SUPER AGGRESSIVE CLEANUP - repeat multiple times to ensure everything is freed
                        import gc
                        for _ in range(3):
                            torch.cuda.empty_cache()
                            gc.collect()
                            torch.cuda.synchronize()

                        # Report actual free memory after cleanup
                        if torch.cuda.is_available():
                            free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                            print(f"   ✓ All models moved to CPU, GPU memory freed")
                            print(f"   ✓ ACTUAL free VRAM after cleanup: {free_mem:.2f} GB")

                    with self._temporary_gpu_model(refiner_model):
                        if mode == 'sparse512':
                            outputs = self.refiner.run(*outputs, mc_threshold=mc_threshold*2.0)
                        elif mode == 'sparse1024':
                            outputs = self.refiner_1024.run(*outputs, mc_threshold=mc_threshold)

            return outputs
    
    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, List[Image.Image]] = None,
        sdf_resolution: int = 1024,
        dense_sampler_params: dict = {'num_inference_steps': 50, 'guidance_scale': 7.0},
        sparse_512_sampler_params: dict = {'num_inference_steps': 30, 'guidance_scale': 7.0},
        sparse_1024_sampler_params: dict = {'num_inference_steps': 15, 'guidance_scale': 7.0},
        generator: Optional[torch.Generator] = None,
        remesh: bool = False,
        simplify_ratio: float = 0.95,
        mc_threshold: float = 0.2,
        remove_interior: bool = True):

        image = self.prepare_image(image)
        
        latent_index = self.inference(image, self.dense_vae, self.dense_dit, self.dense_image_encoder,
                                    self.dense_scheduler, generator=generator, mode='dense', mc_threshold=0.1, **dense_sampler_params)[0]
        
        latent_index = sort_block(latent_index, self.sparse_dit_512.selection_block_size)

        torch.cuda.empty_cache()

        mesh = self.inference(image, self.sparse_vae_512, self.sparse_dit_512, 
                                self.sparse_image_encoder, self.sparse_scheduler_512, 
                                generator=generator, mode='sparse512', 
                                mc_threshold=mc_threshold, latent_index=latent_index, 
                                remove_interior=True, **sparse_512_sampler_params)[0]

        if sdf_resolution == 1024:
            del latent_index
            torch.cuda.empty_cache()

            # COMPREHENSIVE CLEANUP: Unload ALL previous stage models before 1024³ refinement
            import os
            if os.environ.get('DIRECT3D_ULTRA_OFFLOAD', '0') == '1':
                print("🧹 COMPREHENSIVE PRE-1024 CLEANUP: Freeing ALL previous models...")
                # Move ALL dense stage models to CPU
                if hasattr(self, 'dense_vae'):
                    self.dense_vae.cpu()
                if hasattr(self, 'dense_dit'):
                    self.dense_dit.cpu()
                if hasattr(self, 'dense_image_encoder'):
                    self.dense_image_encoder.cpu()
                if hasattr(self, 'dense_scheduler'):
                    # Schedulers are lightweight, but move anyway for completeness
                    pass

                # Move ALL sparse512 stage models to CPU
                if hasattr(self, 'sparse_vae_512'):
                    self.sparse_vae_512.cpu()
                if hasattr(self, 'sparse_dit_512'):
                    self.sparse_dit_512.cpu()
                if hasattr(self, 'refiner'):
                    self.refiner.cpu()
                if hasattr(self, 'sparse_scheduler_512'):
                    pass

                # Also move sparse_image_encoder to CPU (shared encoder)
                if hasattr(self, 'sparse_image_encoder'):
                    self.sparse_image_encoder.cpu()

                # Aggressive memory cleanup
                torch.cuda.empty_cache()
                import gc
                gc.collect()

                # Check memory status
                if torch.cuda.is_available():
                    free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                    print(f"   ✓ All previous models cleared - Free VRAM: {free_memory_gb:.2f} GB")

            mesh = normalize_mesh(mesh)
            latent_index = mesh2index(mesh, size=1024, factor=8)
            # Free the sparse512 mesh to save memory
            del mesh
            torch.cuda.empty_cache()

            latent_index = sort_block(latent_index, self.sparse_dit_1024.selection_block_size)
            print(f"number of latent tokens: {len(latent_index)}")

            mesh = self.inference(image, self.sparse_vae_1024, self.sparse_dit_1024, 
                                self.sparse_image_encoder, self.sparse_scheduler_1024, 
                                generator=generator, mode='sparse1024', 
                                mc_threshold=mc_threshold, latent_index=latent_index, 
                                remove_interior=remove_interior, **sparse_1024_sampler_params)[0]
            
        if remesh:
            import trimesh
            from direct3d_s2.utils import postprocess_mesh
            filled_mesh = postprocess_mesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                simplify=True,
                simplify_ratio=simplify_ratio,
                verbose=True,
            )
            mesh = trimesh.Trimesh(filled_mesh[0], filled_mesh[1])

        outputs = {"mesh": mesh}

        return outputs

    @torch.no_grad()
    def upscale(
        self,
        image: Union[str, List[str], Image.Image, List[Image.Image]],
        mesh,
        resolution: int = 1024,  # New parameter: 512 or 1024
        sparse_sampler_params: dict = {'num_inference_steps': 15, 'guidance_scale': 7.0},
        generator: Optional[torch.Generator] = None,
        remesh: bool = False,
        simplify_ratio: float = 0.95,
        mc_threshold: float = 0.2,
        remove_interior: bool = True):
        """
        Upscale an existing mesh using the 512 or 1024 resolution stage.

        Args:
            resolution: 512 (for 6GB VRAM) or 1024 (for 12GB+ VRAM)
        """
        from direct3d_s2.utils import mesh2index, normalize_mesh

        image = self.prepare_image(image)

        # Normalize and convert mesh to latent index
        mesh = normalize_mesh(mesh)

        if resolution == 1024:
            print("Using 1024³ refiner (requires 12GB+ VRAM)")
            latent_index = mesh2index(mesh, size=1024, factor=8)
            latent_index = sort_block(latent_index, self.sparse_dit_1024.selection_block_size)
            print(f"number of latent tokens: {len(latent_index)}")

            # Run 1024 inference
            mesh = self.inference(image, self.sparse_vae_1024, self.sparse_dit_1024,
                                self.sparse_image_encoder, self.sparse_scheduler_1024,
                                generator=generator, mode='sparse1024',
                                mc_threshold=mc_threshold, latent_index=latent_index,
                                remove_interior=remove_interior, **sparse_sampler_params)[0]
        else:  # resolution == 512
            print("Using 512³ refiner (works on 6GB VRAM)")
            latent_index = mesh2index(mesh, size=512, factor=8)  # Fixed: should be factor=8, not 4
            latent_index = sort_block(latent_index, self.sparse_dit_512.selection_block_size)
            print(f"number of latent tokens: {len(latent_index)}")

            # Run 512 inference
            mesh = self.inference(image, self.sparse_vae_512, self.sparse_dit_512,
                                self.sparse_image_encoder, self.sparse_scheduler_512,
                                generator=generator, mode='sparse512',
                                mc_threshold=mc_threshold, latent_index=latent_index,
                                remove_interior=remove_interior, **sparse_sampler_params)[0]
        
        if remesh:
            import trimesh
            from direct3d_s2.utils import postprocess_mesh
            filled_mesh = postprocess_mesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                simplify=True,
                simplify_ratio=simplify_ratio,
                verbose=True,
            )
            mesh = trimesh.Trimesh(filled_mesh[0], filled_mesh[1])

        outputs = {"mesh": mesh}
        return outputs

    def _is_offload_enabled(self) -> bool:
        """Check if model offloading is enabled"""
        return os.environ.get('DIRECT3D_ENABLE_OFFLOAD', '0') == '1'

    @torch.no_grad()
    def _move_model(self, model, device):
        """Helper to move model to device"""
        if model is not None:
            model.to(device)

    from contextlib import contextmanager
    @contextmanager
    def _temporary_gpu_model(self, *models):
        """
        Context manager to temporarily move models to GPU, then back to CPU.
        Only active when offloading is enabled.
        """
        if not self._is_offload_enabled():
            yield
            return

        # Move to GPU
        if torch.cuda.is_available():
            for model in models:
                if model is not None:
                    model.to('cuda')
            torch.cuda.empty_cache()

        try:
            yield
        finally:
            # Move back to CPU
            if torch.cuda.is_available():
                for model in models:
                    if model is not None:
                        model.to('cpu')
                torch.cuda.empty_cache()
