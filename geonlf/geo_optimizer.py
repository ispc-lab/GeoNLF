import open3d as o3d
import numpy as np
import torch
import tqdm
from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.convert import pano_to_lidar
from utils.recorder import recorder

class Geo_optimizer():
    """
    npoints: all point clouds are downsampled to n points
    n_connected: each frame has 2*n_connected edges, except for the first and last n_connected frames.
    """
    def __init__(self,opt,loader,model,npoints=24000,n_connectd=2):
        self.opt = opt
        self.loader = loader
        self.model=model
        self.npoints = npoints
        self.n_connected = n_connectd
        self.pcds = None
        self.fp16 = opt.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.get_all_pcds()
        self.recorder=recorder(model)
        self.chamLoss = chamfer_3DDist() 

    def downsample(self,pcd):
        N = pcd.shape[0]
        #o3d is based on CPU ...
        p=o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pcd[:,:3])
        p.voxel_down_sample(voxel_size=0.2)
        pcd=np.asarray(p.points)
        pcd=np.concatenate([pcd,np.ones((pcd.shape[0],1))],axis=1)
        if N >= self.npoints:
            sample_idx = np.random.choice(N, self.npoints, replace=False)
        else:
            sample_idx = np.concatenate((np.arange(N), np.random.choice(N, self.npoints-N, replace=True)), axis=-1)
        pcd = pcd[sample_idx, :].astype('float32')
        pcd_on_cpu=pcd[None, ...] #1 N 4
        pcd_on_cpu = torch.FloatTensor(pcd_on_cpu)
        pcd_on_gpu = pcd_on_cpu.to(self.opt.device)
        return pcd_on_gpu
    
    def get_all_pcds(self):
        if self.opt.dataloader=="kitti360":
            intrinsics=[2.0, 26.9]
        elif self.opt.dataloader=="nuscenes":
            intrinsics=[10.0, 40.0]
        else:
            raise Exception("If you have new dataset, please specify the FoV of this dataset here.")

        pcds=torch.zeros(self.opt.dataloader_size,self.npoints,4,device=self.opt.device)
        for data in self.loader:
            idx=data["index"]
            rangemap=data["image_lidar"] #1 32 1080 3
            depth=rangemap[0,:,:,2].cpu().numpy()
            pcd = pano_to_lidar(depth, intrinsics)/self.opt.scale #(N, 3), float32, in lidar frame.
            pcd_on_gpu_downsampled = self.downsample(pcd)
            pcds[idx,:,:]=pcd_on_gpu_downsampled

        self.pcds=pcds       

    def update_all_poses(self):        
        poses=torch.zeros(self.opt.dataloader_size,4,4,device=self.opt.device)
        for data in self.loader:
            idx=data["index"] 
            pose=self.model.get_pose(data["index"],data["pose"]) #1 4 4
            pose[:,:3,3]=pose[:,:3,3]/self.opt.scale
            poses[idx,:,:]=pose
        self.poses=poses
            
    def matrix_construct(self):
        # all point clouds have the same number of points, 
        # so graph-based RCD can be computed in one pass.
        self.update_all_poses()
        pcds=self.pcds
        poses=self.poses
        transpose_poses=poses.permute(0, 2, 1)
        new_pcds_=torch.matmul(pcds, transpose_poses) # 36 N 4
        new_pcds=new_pcds_[:,:,:3] # 36 N 3

        N=new_pcds.shape[0]
        target=[]
        source=[]
        for i in range(1,self.n_connected+1):
            matrix_source=new_pcds[i:,:,:]
            matrix_target=new_pcds[:N-i,:,:]  
            target.append(matrix_target)
            source.append(matrix_source)

        matrix_targets=torch.cat(target,dim=0)   #69 N 3
        matrix_sources=torch.cat(source,dim=0) #69 N 3
        
        dist1, dist2, _, _ = self.chamLoss(matrix_sources,matrix_targets)

        if self.opt.dataloader=="kitti360":
            t_control=self.model.progress.data*0.3
            d=0.15
        else:
            t_control=self.model.progress.data*0.5
            d=0.15

        dist1_=dist1**0.5
        dist1_[dist1_ <= d] = d
        dist1_to_weight=torch.exp(t_control/dist1_)
        sum_dist1_to_weight=torch.sum(dist1_to_weight,dim=1,keepdim=True)
        weight1=dist1_to_weight/sum_dist1_to_weight
        new_dist1_soft_mean=weight1*dist1

        dist2_=dist2**0.5 #torch.sqrt()#dist2
        dist2_[dist2_ <= d] = d
        dist2_to_weight=torch.exp(t_control/dist2_)
        sum_dist2_to_weight=torch.sum(dist2_to_weight,dim=1,keepdim=True)
        weight2=dist2_to_weight/sum_dist2_to_weight
        new_dist2_soft_mean=weight2*dist2

        robust_cd=torch.sum(new_dist1_soft_mean)/dist1.shape[0]+torch.sum(new_dist2_soft_mean)/dist2.shape[0]
        #print("Graph based robust CD :", robust_cd)
        return robust_cd
    
    def graph_based_train_step(self):
        self.optimizer_graph_trans.zero_grad()
        self.optimizer_graph_rot.zero_grad()

        loss=self.matrix_construct()
        l=loss.item()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer_graph_trans)

        self.scaler.step(self.optimizer_graph_rot)
        self.scaler.update()
        self.recorder.cal_pose_error_when_graph_optim(self.loader)  
        return l   

    def geo_optimize(self,epoch,lr_trans,lr_rot):
        if self.opt.no_gt_pose:
            itv1,itv2,itv3=6,10,25
            ep1,ep2,ep3=75,20,20
            reweight_graph=10
            bound1,bound2,bound3=151,650,1300
        else:
            itv1,itv2,itv3=1,1,1 
            ep1,ep2,ep3=10,5,1
            bound1,bound2,bound3=100,350,900
            reweight_graph=3
 
        if self.opt.graph_optim and epoch<=bound1 and epoch%itv1==0:
            self.optimizer_graph_trans=torch.optim.Adam(self.model.get_params_pose_trans(reweight_graph*8*lr_trans), betas=(0.9, 0.99), eps=1e-15)
            self.optimizer_graph_rot=torch.optim.Adam(self.model.get_params_pose_rot(reweight_graph*5*lr_rot), betas=(0.9, 0.99), eps=1e-15)
            pbar = tqdm.tqdm(
                total=10,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )       
            total_loss=0
            local_step=0
            for ep in range(ep1):
                local_step+=1
                loss=self.graph_based_train_step()
                total_loss+=loss
                pbar.set_description(f"loss={loss} ({total_loss/local_step:.4f})")
                pbar.update(1)
                 
            self.recorder.save_train_pose(self.loader)
        
        if self.opt.graph_optim and bound2>epoch>bound1 and epoch%itv2==0:
            self.optimizer_graph_trans=torch.optim.Adam(self.model.get_params_pose_trans(reweight_graph*4*lr_trans), betas=(0.9, 0.99), eps=1e-15)
            self.optimizer_graph_rot=torch.optim.Adam(self.model.get_params_pose_rot(reweight_graph*2*lr_rot), betas=(0.9, 0.99), eps=1e-15)
            pbar = tqdm.tqdm(
                total=5,
                bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )       
            total_loss=0
            local_step=0           
            for ep in range(ep2):
                local_step+=1
                loss=self.graph_based_train_step()
                total_loss+=loss
                pbar.set_description(f"loss={loss} ({total_loss/local_step:.4f})")
                pbar.update(self.loader.batch_size) 
            self.recorder.save_train_pose(self.loader)

        if self.opt.graph_optim and bound2<=epoch<=bound3 and epoch%itv3==0: 
            self.optimizer_graph_trans=torch.optim.Adam(self.model.get_params_pose_trans(reweight_graph*2*lr_trans), betas=(0.9, 0.99), eps=1e-15)
            self.optimizer_graph_rot=torch.optim.Adam(self.model.get_params_pose_rot(reweight_graph*1*lr_rot), betas=(0.9, 0.99), eps=1e-15)
            pbar = tqdm.tqdm(
                total=1,
                bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )       
            total_loss=0
            local_step=0   
            for ep in range(ep3):
                local_step+=1
                loss=self.graph_based_train_step()
                total_loss+=loss
                pbar.set_description(f"loss={loss:.4f} ({total_loss/local_step:.4f})")
                pbar.update(self.loader.batch_size) 
            self.recorder.save_train_pose(self.loader)
        