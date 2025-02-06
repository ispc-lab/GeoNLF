import torch
import numpy as np
import tinycudann as tcnn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from .renderer import NeRFRenderer
from data.dataset import nus_dataset_barf
import pickle
import sys
import os

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))
trunc_exp = _trunc_exp.apply

class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        opt=None,
        device=torch.device("cuda"),
        desired_resolution=40000,
        log2_hashmap_size=19,
        encoding_dir="SphericalHarmonics",
        n_features_per_level=2,
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        out_color_dim=3,
        out_lidar_color_dim=2,
        bound=1,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        self.opt=opt
        self.device=device

        # optimize translation/rotation
        self.rot=opt.rot
        self.trans=opt.trans

        # add translation/rotation noise
        self.noise_rot=opt.noise_rot
        self.noise_trans=opt.noise_trans

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.desired_resolution = desired_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.out_color_dim = out_color_dim
        self.out_lidar_color_dim = out_lidar_color_dim
        self.n_features_per_level = n_features_per_level
        

        # We record the rre/rte for all frames and across all epochs during training.
        # NOTE: The optimized pose may not necessarily be aligned with the ground truth pose, 
        # and alignment cannot be performed before the optimization is completed. 
        # Therefore, this record is compared with the GT pose for reference, 
        # and is valid only when the GT pose is perturbed with noise.
        self.rre=[[] for i in range(self.opt.dataloader_size)]
        self.rte=[[] for i in range(self.opt.dataloader_size)]
        # We record the rre/rte for all frames and during Geo-optimizing process.
        self.rre_when_graph_optim=[[] for i in range(self.opt.dataloader_size)]
        self.rte_when_graph_optim=[[] for i in range(self.opt.dataloader_size)]
        # We record the rendering loss of all frames during training.
        self.loss_record=[[] for i in range(self.opt.dataloader_size)]
        # Use Parameter so it could be checkpointed
        self.progress = torch.nn.Parameter(torch.tensor(0.),requires_grad=False) 

        # Collect pose gradients
        self.se3_refine_trans = torch.nn.Embedding(self.opt.dataloader_size,3).to(self.device)
        torch.nn.init.zeros_(self.se3_refine_trans.weight)
        self.se3_refine_rot=torch.nn.Embedding(self.opt.dataloader_size,3).to(self.device)
        torch.nn.init.zeros_(self.se3_refine_rot.weight)

        # Perturb noise to GT poses
        # Initialization
        se3_noise_rot=torch.randn(self.opt.dataloader_size,3,device=self.device)*0
        se3_noise_trans=torch.randn(self.opt.dataloader_size,3,device=self.device)*0
        if self.noise_rot:
            se3_noise_rot = torch.randn(self.opt.dataloader_size,3,device=self.device)*opt.rot_value
        if self.noise_trans:
            se3_noise_trans = torch.randn(self.opt.dataloader_size,3,device=self.device)*opt.trans_value*self.opt.scale
        se3_noise=torch.cat([se3_noise_rot,se3_noise_trans],dim=-1)
        self.pose_noise =[]
        for s in se3_noise:
            self.pose_noise.append(self.lie.se3_to_SE3(s))
        
        # Network: hash-grids, sigma netwok.
        per_level_scale = np.exp2(np.log2(self.desired_resolution * bound / 15) / (15 - 1))
        self.encoder_hash = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 20,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": 20,
                "per_level_scale": per_level_scale,
            },
        )  
        self.sigma_net = tcnn.Network(
            n_input_dims=40,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # Network: Intensity.
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_lidar_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "degree": 12,
            },
        )
        self.in_dim_lidar_color = self.encoder_lidar_dir.n_output_dims + self.geo_feat_dim
        self.lidar_color_net = tcnn.Network(
            n_input_dims=self.in_dim_lidar_color,
            n_output_dims=self.out_lidar_color_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    # save pose for analysis
    def save_pose(self,idx,pose):
        folder_path = f"./log/initial_pose/{self.opt.workspace}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_path = f"./log/refined_pose/{self.opt.workspace}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_path = f"./log/record/{self.opt.workspace}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        se3_refine_all=torch.cat([self.se3_refine_rot.weight,self.se3_refine_trans.weight],dim=-1)
        se3_refine = se3_refine_all[idx]
        pose_refine = self.lie.se3_to_SE3(se3_refine)
        pose=pose@self.pose_noise[idx]
        if self.opt.no_gt_pose:
            p=torch.eye(4)
            pose=p.unsqueeze(0)
            pose=pose.to(self.opt.device)

        np.save(f"./log/initial_pose/{self.opt.workspace}/initial_{idx}",pose.cpu().numpy())
        pose = self.lie.compose_pair(pose_refine,pose)
        np.save(f"./log/refined_pose/{self.opt.workspace}/refined_{idx}",pose.cpu().numpy())
    
        with open(f"./log/record/{self.opt.workspace}/loss_record.pkl", 'wb') as file:
            pickle.dump(self.loss_record, file)
        with open(f"./log/record/{self.opt.workspace}/rre_record.pkl", 'wb') as file:
            pickle.dump(self.rre, file)
        with open(f"./log/record/{self.opt.workspace}/rte_record.pkl", 'wb') as file:
            pickle.dump(self.rte, file)
        return pose

    def get_pose(self,idx,pose):
        if self.rot and self.trans:
            se3_refines=torch.cat([self.se3_refine_rot.weight,self.se3_refine_trans.weight],dim=-1)
            self.se3_refines=se3_refines
            se3_refine = se3_refines[idx] #3
            pose_refine = self.lie.se3_to_SE3(se3_refine)
            pose=pose@self.pose_noise[idx]
            if self.opt.no_gt_pose:
                p=torch.eye(4)
                pose=p.unsqueeze(0)
                pose=pose.to(self.opt.device)
            pose_new = self.lie.compose_pair(pose_refine,pose)
        elif self.rot:
            se3_trans=torch.zeros([self.opt.dataloader_size,3],device=self.device)
            self.se3_refine=torch.cat([self.se3_refine_rot.weight,se3_trans],dim=-1)
            se3_refine=self.se3_refine[idx]
            pose_refine = self.lie.se3_to_SE3(se3_refine)
            pose=pose@self.pose_noise[idx]
            pose_new = self.lie.compose_pair(pose_refine,pose)
        elif self.trans:
            se3_rot=torch.zeros([self.opt.dataloader_size,3],device=self.device)
            self.se3_refine=torch.cat([se3_rot,self.se3_refine_trans.weight],dim=-1)
            se3_refine=self.se3_refine[idx]
            pose_refine = self.lie.se3_to_SE3(se3_refine)
            pose=pose@self.pose_noise[idx]
            pose_new = self.lie.compose_pair(pose_refine,pose)
        else:
            se3_trans=torch.zeros([self.opt.dataloader_size,3],device=self.device)
            se3_rot=torch.zeros([self.opt.dataloader_size,3],device=self.device)
            self.se3_refine=torch.cat([se3_rot,se3_trans],dim=-1)
            # In this option, pose_noise is set to 0
            pose_new=pose@self.pose_noise[idx]
        return pose_new
    
    def forward(self, x, d):
        pass

    def density(self, x):
        #coarse-to-fine training approach
        start,end = self.opt.c2f
        alpha = ((self.progress.data-start)/(end-start)*20).clamp_(min=0,max=20)
        k = torch.arange(20,dtype=torch.float32,device=self.device)
        weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        weight=weight.repeat_interleave(2)

        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x_hash=self.encoder_hash(x)
        x=x_hash
        x=weight*x
        h = self.sigma_net(x)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        return {
            "sigma": sigma,
            "geo_feat": geo_feat,
        }
    
    # Intensity ((in fact
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        if mask is not None:
            rgbs = torch.zeros(
                mask.shape[0], self.out_dim, dtype=x.dtype, device=x.device
            )  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_lidar_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.lidar_color_net(h)
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h
        return rgbs
    
    # optimizer utils
    def get_params_pose_trans(self,lr):
        if self.opt.no_gt_pose:
            params = [{"params": self.se3_refine_trans.parameters(),"lr":0.1*lr*6}]
        else:
            params = [{"params": self.se3_refine_trans.parameters(),"lr":0.1*lr}]
        return params
    def get_params_pose_rot(self,lr):
        if self.opt.no_gt_pose:
            params = [{"params": self.se3_refine_rot.parameters(),"lr":0.5*lr*6}]
        else:
            if self.opt.dataloader=="kitti360":
                params = [{"params": self.se3_refine_rot.parameters(),"lr":0.5*lr}]
            else:
                params = [{"params": self.se3_refine_rot.parameters(),"lr":0.5*lr}]
        return params
    def get_params(self, lr):
        params = [
            {"params": self.encoder_hash.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.encoder_lidar_dir.parameters(), "lr": lr},
            {"params": self.lidar_color_net.parameters(), "lr": lr},
        ]
        if self.bg_radius > 0:
            params.append({"params": self.encoder_bg.parameters(), "lr": lr})
            params.append({"params": self.bg_net.parameters(), "lr": lr})
        return params
