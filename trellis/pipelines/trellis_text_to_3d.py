from typing import *
import torch
import torch.nn as nn
import numpy as np
import os
from contextlib import contextmanager
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        # Use pipeline's device instead of hardcoded .cuda()
        model = model.to(self.device)
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    def _is_offload_enabled(self) -> bool:
        """Check if model offloading is enabled"""
        return os.environ.get('TRELLIS_ENABLE_OFFLOAD', '0') == '1'

    @contextmanager
    def _temporary_gpu_model(self, model_name: str):
        """
        Context manager to temporarily move a model to GPU, then back to CPU.
        Only active when offloading is enabled (low VRAM systems).
        """
        if not self._is_offload_enabled():
            # Offloading disabled, model stays on current device
            yield self.models[model_name]
            return

        model = self.models[model_name]
        original_device = next(model.parameters()).device

        try:
            # Move to GPU
            if torch.cuda.is_available() and original_device.type == 'cpu':
                print(f"  → Moving {model_name} to GPU...")
                model.to('cuda')  # Explicitly use CUDA, not self.device (which is CPU in offload mode)
                torch.cuda.empty_cache()

            yield model

        finally:
            # Move back to CPU
            if torch.cuda.is_available() and original_device.type == 'cpu':
                print(f"  → Moving {model_name} back to CPU...")
                model.to('cpu')
                torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        # Move tokens to same device as text model (not self.device which checks other models)
        text_model_device = next(self.text_cond_model['model'].parameters()).device
        tokens = encoding['input_ids'].to(text_model_device)
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state

        return embeddings
        
    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent with offloading
        with self._temporary_gpu_model('sparse_structure_flow_model') as flow_model:
            reso = flow_model.resolution
            # Get model's current device (GPU when offloading, otherwise self.device)
            model_device = next(flow_model.parameters()).device
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(model_device)
            
            # Move conditioning tensors to same device as model
            cond_on_device = {}
            for k, v in cond.items():
                if isinstance(v, torch.Tensor):
                    cond_on_device[k] = v.to(model_device)
                else:
                    cond_on_device[k] = v
            
            sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond_on_device,
                **sampler_params,
                verbose=True
            ).samples

        # Decode occupancy latent with offloading
        with self._temporary_gpu_model('sparse_structure_decoder') as decoder:
            coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        # Decode each format with offloading - one decoder on GPU at a time
        if 'mesh' in formats:
            # Clear cache before mesh decoding (mesh decoder needs more VRAM)
            import torch
            import gc
            if torch.cuda.is_available():
                gc.collect()  # Force Python garbage collection
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
            with self._temporary_gpu_model('slat_decoder_mesh') as decoder:
                ret['mesh'] = decoder(slat)
                # Clear immediately after mesh decode
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        if 'gaussian' in formats:
            with self._temporary_gpu_model('slat_decoder_gs') as decoder:
                ret['gaussian'] = decoder(slat)
        if 'radiance_field' in formats:
            with self._temporary_gpu_model('slat_decoder_rf') as decoder:
                ret['radiance_field'] = decoder(slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent with offloading
        with self._temporary_gpu_model('slat_flow_model') as flow_model:
            # Get model's current device (GPU when offloading, otherwise self.device)
            model_device = next(flow_model.parameters()).device
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], flow_model.in_channels).to(model_device),
                coords=coords,
            )
            
            # Move conditioning tensors to same device as model
            cond_on_device = {}
            for k, v in cond.items():
                if isinstance(v, torch.Tensor):
                    cond_on_device[k] = v.to(model_device)
                else:
                    cond_on_device[k] = v
            
            sampler_params = {**self.slat_sampler_params, **sampler_params}
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                **cond_on_device,
                **sampler_params,
                verbose=True
            ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])

        # CRITICAL: Clear cache before sparse structure (prevents OOM)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)

        # CRITICAL: Clear cache after sparse structure, before SLAT (prevents OOM)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        slat = self.sample_slat(cond, coords, slat_sampler_params)

        # CRITICAL: Clear cache after SLAT, before decoding (prevents OOM)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.decode_slat(slat, formats)
    
    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        # Use pipeline's device instead of hardcoded .cuda()
        return torch.tensor(vertices).int().to(self.device)

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().to(self.device),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
