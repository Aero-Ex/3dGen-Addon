# -*- coding: utf-8 -*-
import itertools
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .unet3d import UNet3DModel
import trimesh
from tqdm import tqdm
from skimage import measure
from direct3d_s2.modules.utils import convert_module_to_f16, convert_module_to_f32
import direct3d_s2.modules.sparse as sp
import gc


def adaptive_conv(inputs,weights):
    padding = (1, 1, 1, 1, 1, 1)
    padded_input = F.pad(inputs, padding, mode="constant", value=0)
    output = torch.zeros_like(inputs)
    size=inputs.shape[-1]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                output=output+padded_input[:,:,i:i+size,j:j+size,k:k+size]*weights[:,i*9+j*3+k:i*9+j*3+k+1]
    return output

def adaptive_block(inputs,conv,weights_=None):
    if weights_ != None:
        weights = conv(weights_)
    else:
        weights = conv(inputs)
    weights = F.normalize(weights, dim=1, p=1)
    for i in range(3):
        inputs = adaptive_conv(inputs, weights)
    return inputs

class GeoDecoder(nn.Module):

    def __init__(self, 
                 n_features: int,
                 hidden_dim: int = 32, 
                 num_layers: int = 4, 
                 use_sdf: bool = False,
                 activation: nn.Module = nn.ReLU):
        super().__init__()
        self.use_sdf=use_sdf
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 8),
        )

        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.net(x)
        return x


class Voxel_RefinerXL(nn.Module):
    def __init__(self,
                in_channels: int = 1,
                out_channels: int = 1,
                layers_per_block: int = 2,
                layers_mid_block: int = 2,
                patch_size: int = 192,
                res: int = 512,
                use_checkpoint: bool=False,
                use_fp16: bool = False):

        super().__init__()

        self.unet3d1 = UNet3DModel(in_channels=16, out_channels=8, use_conv_out=False,
                                   layers_per_block=layers_per_block, layers_mid_block=layers_mid_block, 
                                   block_out_channels=(8, 32, 128,512), norm_num_groups=4, use_checkpoint=use_checkpoint)
        self.conv_in = nn.Conv3d(in_channels, 8, kernel_size=3, padding=1)
        self.latent_mlp = GeoDecoder(32)
        self.adaptive_conv1 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False))
        self.adaptive_conv2 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False))
        self.adaptive_conv3 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False))
        self.mid_conv = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.conv_out = nn.Conv3d(8, out_channels, kernel_size=3, padding=1)
        self.patch_size = patch_size
        self.res = res

        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        # self.blocks.apply(convert_module_to_f16)
        self.apply(convert_module_to_f16)

    def run(self,
            reconst_x,
            feat, 
            mc_threshold=0,
        ):
        batch_size = int(reconst_x.coords[..., 0].max()) + 1
        sparse_sdf, sparse_index = reconst_x.feats, reconst_x.coords
        sparse_feat = feat.feats
        device = sparse_sdf.device
        dtype = sparse_sdf.dtype
        res = self.res
        
        # Smart Offloading: Store large dense tensors on CPU to save VRAM
        # feats tensor takes ~4GB at 512 resolution (float16), causing OOM on <8GB GPUs
        storage_device = 'cpu'

        sdfs = []
        # Move sparse data to CPU for scattering into CPU tensors
        sparse_sdf_cpu = sparse_sdf.to(storage_device)
        sparse_index_cpu = sparse_index.to(storage_device)
        sparse_feat_cpu = sparse_feat.to(storage_device)

        for i in range(batch_size):
            idx = sparse_index_cpu[..., 0] == i
            sparse_sdf_i, sparse_index_i = sparse_sdf_cpu[idx].squeeze(-1),  sparse_index_cpu[idx][..., 1:]

            # Debug: Check if indices are out of bounds
            max_idx = sparse_index_i.max().item() if sparse_index_i.numel() > 0 else 0
            if max_idx >= res:
                print(f"ERROR: sparse_index max value {max_idx} >= res {res}")
                print(f"  sparse_index shape: {sparse_index_i.shape}")
                print(f"  sparse_index range: [{sparse_index_i.min().item()}, {max_idx}]")
                print(f"  Clamping indices to valid range [0, {res-1}]")
                sparse_index_i = torch.clamp(sparse_index_i, 0, res-1)

            sdf = torch.ones((res, res, res), device=storage_device, dtype=dtype)
            sdf[sparse_index_i[..., 0], sparse_index_i[..., 1], sparse_index_i[..., 2]] = sparse_sdf_i
            sdfs.append(sdf.unsqueeze(0))

        sdfs = torch.stack(sdfs, dim=0)
        feats = torch.zeros((batch_size, sparse_feat.shape[-1], res, res, res),
                            device=storage_device, dtype=dtype)

        # Clamp sparse_index_cpu to valid range to prevent out-of-bounds errors
        sparse_index_clamped = sparse_index_cpu.clone()
        sparse_index_clamped[..., 1:] = torch.clamp(sparse_index_clamped[..., 1:], 0, res-1)
        feats[sparse_index_clamped[...,0],:,sparse_index_clamped[...,1],sparse_index_clamped[...,2],sparse_index_clamped[...,3]] = sparse_feat_cpu
        
        N = sdfs.shape[0]
        # outputs can stay on GPU or CPU, it's smaller (256MB for 512^3 float16)
        # But let's keep it on device for now as it's constructed from patches
        outputs = torch.ones([N,1,res,res,res], dtype=dtype, device=device)
        
        stride = 160
        patch_size = self.patch_size
        step = 3
        # sdfs and feats are on CPU now
        
        patchs=[]
        for i in range(step):
            for j in range(step):
                for k in tqdm(range(step), desc=f"Refining {self.res}³ mesh"):
                    # Crop from CPU tensors
                    sdf = sdfs[:, :, stride * i: stride * i + patch_size,
                               stride * j: stride * j + patch_size,
                               stride * k: stride * k + patch_size]
                    crop_feats = feats[:, :, stride * i: stride * i + patch_size, 
                                       stride * j: stride * j + patch_size, 
                                       stride * k: stride * k + patch_size]
                    
                    # Move crops to GPU for processing
                    sdf = sdf.to(device)
                    crop_feats = crop_feats.to(device)

                    inputs = self.conv_in(sdf)
                    crop_feats = self.latent_mlp(crop_feats.permute(0,2,3,4,1)).permute(0,4,1,2,3)
                    inputs = torch.cat([inputs, crop_feats],dim=1)
                    mid_feat = self.unet3d1(inputs)  
                    mid_feat = adaptive_block(mid_feat, self.adaptive_conv1)
                    mid_feat = self.mid_conv(mid_feat)
                    mid_feat = adaptive_block(mid_feat, self.adaptive_conv2)
                    final_feat = self.conv_out(mid_feat)
                    final_feat = adaptive_block(final_feat, self.adaptive_conv3, weights_=mid_feat)
                    output = F.tanh(final_feat)
                    patchs.append(output)
        weights = torch.linspace(0, 1, steps=32, device=device, dtype=dtype)
        lines=[]
        for i in range(9):
            out1 = patchs[i * 3]
            out2 = patchs[i * 3 + 1]
            out3 = patchs[i * 3 + 2]
            line = torch.ones([N, 1, 192, 192,res], dtype=dtype, device=device) * 2
            line[:, :, :, :, :160] = out1[:, :, :, :, :160]
            line[:, :, :, :, 192:320] = out2[:, :, :, :, 32:160]
            line[:, :, :, :, 352:] = out3[:, :, :, :, 32:]
            
            line[:,:,:,:,160:192] = out1[:,:,:,:,160:] * (1-weights.reshape(1,1,1,1,-1)) + out2[:,:,:,:,:32] * weights.reshape(1,1,1,1,-1)
            line[:,:,:,:,320:352] = out2[:,:,:,:,160:] * (1-weights.reshape(1,1,1,1,-1)) + out3[:,:,:,:,:32] * weights.reshape(1,1,1,1,-1)
            lines.append(line)
        layers=[]
        for i in range(3):
            line1 = lines[i*3]
            line2 = lines[i*3+1]
            line3 = lines[i*3+2]
            layer = torch.ones([N,1,192,res,res], device=device, dtype=dtype) * 2
            layer[:,:,:,:160] = line1[:,:,:,:160]
            layer[:,:,:,192:320] = line2[:,:,:,32:160]
            layer[:,:,:,352:] = line3[:,:,:,32:]
            layer[:,:,:,160:192] = line1[:,:,:,160:]*(1-weights.reshape(1,1,1,-1,1))+line2[:,:,:,:32]*weights.reshape(1,1,1,-1,1)
            layer[:,:,:,320:352] = line2[:,:,:,160:]*(1-weights.reshape(1,1,1,-1,1))+line3[:,:,:,:32]*weights.reshape(1,1,1,-1,1)
            layers.append(layer)
        outputs[:,:,:160] = layers[0][:,:,:160]
        outputs[:,:,192:320] = layers[1][:,:,32:160]
        outputs[:,:,352:] = layers[2][:,:,32:]
        outputs[:,:,160:192] = layers[0][:,:,160:]*(1-weights.reshape(1,1,-1,1,1))+layers[1][:,:,:32]*weights.reshape(1,1,-1,1,1)
        outputs[:,:,320:352] = layers[1][:,:,160:]*(1-weights.reshape(1,1,-1,1,1))+layers[2][:,:,:32]*weights.reshape(1,1,-1,1,1)
        # outputs = -outputs

        # Clean up large CPU tensors to free RAM for marching cubes
        del sdfs, feats, patchs, lines, layers
        gc.collect()
        torch.cuda.empty_cache()

        meshes = []
        for i in range(outputs.shape[0]):
            vertices, faces, _, _ = measure.marching_cubes(outputs[i, 0].cpu().numpy(), level=mc_threshold, method='lewiner')
            vertices = vertices / res * 2 - 1
            meshes.append(trimesh.Trimesh(vertices, faces))
        
        return meshes


class Voxel_RefinerXL_sign(nn.Module):
    def __init__(self,
                in_channels: int=1,
                out_channels: int=1,
                layers_per_block: int=2,
                layers_mid_block: int=2,
                patch_size: int=192,
                res: int=512,
                infer_patch_size: int=192,
                use_checkpoint: bool=False,
                use_fp16: bool = False):
        super().__init__()

        self.unet3d1 = UNet3DModel(in_channels=8, out_channels=8, use_conv_out=False, 
                                   layers_per_block=layers_per_block, layers_mid_block=layers_mid_block, 
                                   block_out_channels=(8,32,128,512), norm_num_groups=4, use_checkpoint=use_checkpoint)
        self.conv_in = nn.Conv3d(in_channels, 8, kernel_size=3, padding=1)
        self.conv_out = nn.Conv3d(8, out_channels, kernel_size=3, padding=1)
        self.downsample = sp.SparseDownsample(factor=2)
        self.patch_size = patch_size
        self.infer_patch_size = infer_patch_size
        self.res = res
       
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        self.apply(convert_module_to_f16)
    
    def run(self,
             reconst_x=None,
             feat=None,
             mc_threshold=0,
        ):
        import os
        ultra_offload = os.environ.get('DIRECT3D_ULTRA_OFFLOAD', '0') == '1'

        # CRITICAL: Enable gradient checkpointing for ultra offload mode to reduce memory
        if ultra_offload:
            self.unet3d1.use_checkpoint = True
            print("   → Gradient checkpointing ENABLED for UNet3D (reduces VRAM usage)")

        batch_size = int(reconst_x.coords[..., 0].max()) + 1
        sparse_sdf, sparse_index = reconst_x.feats, reconst_x.coords
        device = sparse_sdf.device
        voxel_resolution = 1024
        sdfs=[]
        for i in range(batch_size):
            idx = sparse_index[..., 0] == i
            sparse_sdf_i, sparse_index_i = sparse_sdf[idx].squeeze(-1),  sparse_index[idx][..., 1:]
            sdf = torch.ones((voxel_resolution, voxel_resolution, voxel_resolution)).to(device).to(sparse_sdf_i.dtype)
            sdf[sparse_index_i[..., 0], sparse_index_i[..., 1], sparse_index_i[..., 2]] = sparse_sdf_i
            sdfs.append(sdf.unsqueeze(0))

        sdfs1024 = torch.stack(sdfs,dim=0)
        reconst_x1024 = reconst_x
        reconst_x = self.downsample(reconst_x)
        batch_size = int(reconst_x.coords[..., 0].max()) + 1
        sparse_sdf, sparse_index = reconst_x.feats, reconst_x.coords
        device = sparse_sdf.device
        dtype = sparse_sdf.dtype
        voxel_resolution = 512
        sdfs = torch.ones((batch_size, voxel_resolution, voxel_resolution, voxel_resolution),device=device, dtype=sparse_sdf.dtype)
        sdfs[sparse_index[...,0],sparse_index[...,1],sparse_index[...,2],sparse_index[...,3]] = sparse_sdf.squeeze(-1)
        sdfs = sdfs.unsqueeze(1)

        N = sdfs.shape[0]
        stride = 128
        patch_size = self.patch_size
        step = 3

        # SMART AUTO-DETECTION: Check available GPU memory
        use_extreme_mode = False
        if ultra_offload:
            # Check how much VRAM is free after cleanup
            free_memory_gb = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
            free_memory_gb = free_memory_gb / (1024**3)
            total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)

            print(f"📊 GPU Memory Status:")
            print(f"   Total VRAM: {total_memory_gb:.2f} GB")
            print(f"   Free VRAM: {free_memory_gb:.2f} GB")

            # If we have > 3GB free, try regular offloading (faster)
            # If < 3GB free, use extreme offloading (safer)
            if free_memory_gb >= 3.0:
                print(f"✅ Sufficient VRAM available - using REGULAR offloading (FAST mode)")
                use_extreme_mode = False
            else:
                print(f"⚠️  Limited VRAM - using EXTREME offloading (SLOW but safe mode)")
                use_extreme_mode = True

        # ULTRA-AGGRESSIVE OFFLOADING MODE
        if ultra_offload and use_extreme_mode:
            print(f"🚀 EXTREME OFFLOAD MODE: Moving EVERYTHING to CPU, minimal GPU usage")

            # Move ALL data tensors to CPU to save VRAM
            sdfs_cpu = sdfs.cpu()
            outputs = torch.ones([N,1,512,512,512], device='cpu', dtype=dtype)

            # Move all models to CPU first
            self.conv_in.cpu()
            self.unet3d1.cpu()
            self.conv_out.cpu()
            torch.cuda.empty_cache()

            for i in range(step):
                for j in range(step):
                    for k in tqdm(range(step), desc=f"Refining {self.res}³ [EXTREME]"):
                        # Extract patch from CPU
                        sdf = sdfs_cpu[:,:,stride*i:stride*i+patch_size,stride*j:stride*j+patch_size,stride*k:stride*k+patch_size]

                        # Move ONLY this small patch to GPU
                        sdf_gpu = sdf.to(device)

                        # Load conv_in to GPU, process, unload
                        self.conv_in.to(device)
                        inputs = self.conv_in(sdf_gpu)
                        self.conv_in.cpu()
                        del sdf_gpu
                        torch.cuda.empty_cache()

                        # Load unet3d1 to GPU, process, unload
                        self.unet3d1.to(device)
                        mid_feat = self.unet3d1(inputs)
                        self.unet3d1.cpu()
                        del inputs
                        torch.cuda.empty_cache()

                        # Load conv_out to GPU, process, unload
                        self.conv_out.to(device)
                        final_feat = self.conv_out(mid_feat)
                        self.conv_out.cpu()
                        del mid_feat
                        torch.cuda.empty_cache()

                        # Process result on CPU
                        final_feat_cpu = final_feat.cpu()
                        del final_feat
                        torch.cuda.empty_cache()

                        output = F.sigmoid(final_feat_cpu)
                        output[output>=0.5] = 1
                        output[output<0.5] = -1
                        outputs[:, :, stride*i:stride*i+patch_size, stride*j:stride*j+patch_size, stride*k:stride*k+patch_size] = output
                        del final_feat_cpu, output, sdf
                        torch.cuda.empty_cache()

            # Move result back to GPU
            outputs = outputs.to(device)
            sdfs = sdfs_cpu.to(device)

            # Move models back to device for next use
            self.conv_in.to(device)
            self.unet3d1.to(device)
            self.conv_out.to(device)
        else:
            # Regular/Standard mode - keep data on GPU, process normally
            if ultra_offload and not use_extreme_mode:
                desc = f"Refining {self.res}³ [FAST]"
            else:
                desc = f"Refining {self.res}³ mesh"

            outputs = torch.ones([N,1,512,512,512],device=sdfs.device, dtype=dtype)
            for i in range(step):
                for j in range(step):
                    for k in tqdm(range(step), desc=desc):
                        sdf = sdfs[:,:,stride*i:stride*i+patch_size,stride*j:stride*j+patch_size,stride*k:stride*k+patch_size]
                        inputs = self.conv_in(sdf)
                        mid_feat = self.unet3d1(inputs)
                        final_feat = self.conv_out(mid_feat)
                        output = F.sigmoid(final_feat)
                        output[output>=0.5] = 1
                        output[output<0.5] = -1
                        outputs[:, :, stride*i:stride*i+patch_size, stride*j:stride*j+patch_size, stride*k:stride*k+patch_size] = output

                        # Add cache clearing for regular mode too when ultra_offload is enabled
                        if ultra_offload and not use_extreme_mode:
                            del inputs, mid_feat, final_feat, output, sdf
                            torch.cuda.empty_cache()
        outputs = outputs.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        
        # Memory Optimization: In-place update of sdfs1024 to save RAM
        # sdfs1024 is [N, 1, 1024, 1024, 1024]
        # outputs is [N, 1, 1024, 1024, 1024]
        
        # 1. Apply abs() in-place if possible, or re-assign
        sdfs1024.abs_() 
        
        # 2. Multiply in-place
        sdfs1024.mul_(outputs)
        
        # Free outputs immediately
        del outputs
        
        # 3. Update sparse regions
        sparse_index1024 = reconst_x1024.coords
        
        # We need to use the original values from reconst_x1024 for the sparse regions
        # reconst_x1024.feats is the sparse SDF values
        # We can't easily do this in-place without a temporary tensor for the sparse values if we want to be exact,
        # but let's try to minimize copies.
        
        # Extract sparse values
        # reconst_x1024.feats is [N, 1], which matches the target slice [N, 1]
        sparse_vals = reconst_x1024.feats
        
        # Assign to grid
        sdfs1024[sparse_index1024[...,0], :, sparse_index1024[...,1], sparse_index1024[...,2], sparse_index1024[...,3]] = sparse_vals
        
        # Free sparse source data if possible (though it's small)
        del reconst_x1024
        
        # Clean up GPU memory before moving to CPU
        torch.cuda.empty_cache()
        
        # Move to CPU one by one to avoid holding all 1024^3 grids in RAM if batch_size > 1
        # But sdfs1024 is already a full tensor.
        
        meshes = []
        batch_size = sdfs1024.shape[0]
        
        import gc
        
        for i in range(batch_size):
            # Extract single grid to CPU
            # This creates a copy, but we can delete the GPU version slice if we were careful, 
            # but sdfs1024 is one block.
            
            # To save RAM, we move the tensor to CPU, then delete the GPU one? 
            # Or better: process on GPU if marching cubes supported it (it doesn't).
            
            # Strategy: Move one item to numpy, process, then delete.
            
            print(f"    [INFO] Extracting mesh {i+1}/{batch_size}...")
            
            # Force GC to clear any fragmentation
            gc.collect()
            
            # Get the single grid as numpy array
            # We use .detach().cpu().numpy() which creates a copy in RAM.
            # sdfs1024 is on GPU (usually).
            
            grid_np = sdfs1024[i, 0].detach().cpu().numpy()
            
            # If sdfs1024 was on CPU, this is just a view or copy.
            
            # Run Marching Cubes
            # This requires ~4GB for 1024^3 float32
            try:
                vertices, faces, _, _ = measure.marching_cubes(grid_np, level=mc_threshold, method="lewiner")
                
                # Normalize vertices
                vertices = vertices / 1024 * 2 - 1
                meshes.append(trimesh.Trimesh(vertices, faces))
                
                # Free the grid immediately
                del grid_np
                gc.collect()
                
            except Exception as e:
                print(f"    [ERROR] Marching cubes failed: {e}")
                # Try to recover/return empty mesh or partial
                meshes.append(trimesh.Trimesh())

        # Clean up the big tensor
        del sdfs1024
        torch.cuda.empty_cache()
        gc.collect()
        
        return meshes

