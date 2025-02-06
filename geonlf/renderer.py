import math
import trimesh
import numpy as np
import torch
import torch.nn as nn
#from data.dataset.base_dataset import get_lidar_rays
from packaging import version as pver

def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples
        ).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print("[visualize points]", pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()
class Lie():
    def compose_pair(self,pose_a,pose_b):
        pose_new=torch.matmul(pose_b, pose_a).to(dtype=torch.float32)
        return pose_new
    def so3_to_SO3(self,w): # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R
    def SO3_to_so3(self,R,eps=1e-7): 
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w
    def se3_to_SE3(self,wu): # [...,3]
        w,u = wu.split([3,3],dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        #Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        Rt = torch.cat([R,(u[...,None])],dim=-1)
        pad=torch.tensor([0,0,0,1]).unsqueeze(0).to(wu.device) #1,4
        Rt=torch.cat([Rt,pad],dim=0)
        return Rt
    def SE3_to_se3(self,Rt,eps=1e-8): # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    
    def skew_symmetric(self,w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx
    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans


class NeRFRenderer(nn.Module):
    def __init__(
        self,
        bound=1,
        density_scale=1, 
        min_near=0.2,
        min_near_lidar=0.2,
        density_thresh=0.01,
        bg_radius=-1,
    ):
        super().__init__()
        self.lie=Lie()
        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.min_near_lidar = min_near_lidar
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius  # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer("aabb_train", aabb_train)
        self.register_buffer("aabb_infer", aabb_infer)

    # we need complete these functions in network_tcnn.py (NeRFNetwork)
    def forward(self, x, d):
        raise NotImplementedError()
    def density(self, x):
        raise NotImplementedError()
    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()
    def save_pose(self,idx,pose):
        raise NotImplementedError()
    def get_pose(self,idx,pose):
        raise NotImplementedError()   

    @torch.cuda.amp.autocast(enabled=False)
    def get_lidar_rays(self,poses, intrinsics, H, W, N, patch_size):
        """
        Get lidar rays.
        Args:
            poses: [B, 4, 4], lidar2world
            intrinsics: [2]
            H, W, N: int
        Returns:
            rays_o, rays_d: [B, N, 3]
            inds: [B, N]
        """
        device = poses.device
        B = poses.shape[0]

        i, j = custom_meshgrid(
            torch.linspace(0, W - 1, W, device=device),
            torch.linspace(0, H - 1, H, device=device),
        )  
        i = i.t().reshape([1, H * W]).expand([B, H * W])
        j = j.t().reshape([1, H * W]).expand([B, H * W])
        results = {}
        if N > 0:
            N = min(N, H * W)
            if isinstance(patch_size, int):
                patch_size_x, patch_size_y = patch_size, patch_size
            elif len(patch_size) == 1:
                patch_size_x, patch_size_y = patch_size[0], patch_size[0]
            else:
                patch_size_x, patch_size_y = patch_size
            
            if patch_size_x > 0:
                # random sample left-top cores.
                # NOTE: this impl will lead to less sampling on the image corner
                # pixels... but I don't have other ideas.
                num_patch = N // (patch_size_x * patch_size_y)
                inds_x = torch.randint(0, H - patch_size_x, size=[num_patch], device=device) #np,1
                inds_y = torch.randint(0, W - patch_size_y, size=[num_patch], device=device) #np,1
                inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

                # create meshgrid for each patch，每个patch([2,8])。
                pi, pj = custom_meshgrid(
                    torch.arange(patch_size_x, device=device),
                    torch.arange(patch_size_y, device=device),
                )
                offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [px*py, 2]

                inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # np,1,2 + 1,px*py,2=[np, px*py, 2] 
                inds = inds.view(-1, 2)  # [N, 2]
                inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten
                inds = inds.expand([B, N])

            else:
                inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
                inds = inds.expand([B, N])

            i = torch.gather(i, -1, inds)
            j = torch.gather(j, -1, inds)
            results["inds"] = inds

        else:
            inds = torch.arange(H * W, device=device).expand([B, H * W])
            results["inds"] = inds

        fov_up, fov = intrinsics
        beta = -(i - W / 2) / W * 2 * np.pi
        alpha = (fov_up - j / H * fov) / 180 * np.pi

        directions = torch.stack(
            [
                torch.cos(alpha) * torch.cos(beta),
                torch.cos(alpha) * torch.sin(beta),
                torch.sin(alpha),
            ],
            -1,
        )

        poses=poses.to(dtype=torch.float32)
        rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
        rays_o = poses[..., :3, 3]  # [B, 3]
        rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

        results["rays_o"] = rays_o
        results["rays_d"] = rays_d

        return results
    
    def run(
        self,
        data,        
        rays_o,
        rays_d,
        image_lidar_sampled,
        num_steps=768,      
        perturb=False,
        **kwargs
    ):
        self.out_dim = self.out_lidar_color_dim
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device
        # 1. Choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer
        
        # 2. Generate xyzs
        nears = (torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.min_near_lidar)
        fars = (torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.min_near_lidar * 81.0) 
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)
        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = (z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist)
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # 3. Obtain density and depth 
        density_outputs=self.density(xyzs.reshape(-1, 3))
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)
        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]
        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])
        mask = weights > 1e-4  # hard coded 
        depth = torch.sum(weights * z_vals, dim=-1)
        depth = depth.view(*prefix)

        # 4. Obtain intensity
        rgbs = self.color(
            xyzs.reshape(-1, 3),
            dirs.reshape(-1, 3),
            mask=mask.reshape(-1),
            **density_outputs
        )

        rgbs = rgbs.view(N, -1, self.out_dim)  # [N, T+t, 2]
        intensity = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]
        intensity = intensity.view(*prefix, self.out_dim)

        return {
            "image_lidar":data["image_lidar"],
            "image_lidar_sampled":image_lidar_sampled,
            "depth_lidar": depth,
            "intensity": intensity,
        }
    

    def render(
        self,
        data,
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):
        # 1. Obtain the refined pose.
        pose=self.get_pose(data["index"],data["pose"]) # 1 4 4

        # 2. Obtain the origin and direction of rays.
        #    Given the limited GPU memory, we sample 4096 rays at a time.
        rays_lidar=self.get_lidar_rays(pose,data["intrinsics_lidar"],data["H_lidar"],data["W_lidar"],data["num_rays_lidar"],data["patch"])
        rays_o = rays_lidar["rays_o"]  # [B, N, 3]
        rays_d = rays_lidar["rays_d"]  # [B, N, 3]
        _run=self.run
        
        # 3. Since we have the sampled rays, we can locate the corresponding GT in the original image.
        image_lidar=data["image_lidar"]
        B=  image_lidar.shape[0]
        C = image_lidar.shape[-1] 
        image_lidar_sampled = torch.gather(image_lidar.view(B, -1, C),1,torch.stack(C * [rays_lidar["inds"]], -1))  # [B, N, 3]

        device = rays_o.device
        B , N = rays_o.shape[0:2]
        if staged:
            # If we need render all pixels of a range image
            out_dim = self.out_lidar_color_dim
            depth = torch.empty((B, N), device=device)
            intensity = torch.empty((B, N, out_dim), device=device)
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(
                        data,
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        image_lidar_sampled,
                        **kwargs
                    )
                    depth[b : b + 1, head:tail] = results_["depth_lidar"]
                    intensity[b : b + 1, head:tail] = results_["intensity"]
                    head += max_ray_batch

            results = {}
            results["depth_lidar"] = depth
            results["intensity"] = intensity
            results["image_lidar"] = results_["image_lidar"]
            results["image_lidar_sampled"]=image_lidar_sampled

        else:
            # Just N (4096 in our experiment) rays
            results = _run(data, rays_o, rays_d, image_lidar_sampled, **kwargs)
        return results
    