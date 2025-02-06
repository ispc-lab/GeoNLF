import torch
import configargparse
import os
import numpy as np
import random
from GeoNLF.geo_optimizer import Geo_optimizer
    
from GeoNLF.trainer import Trainer
from utils.metrics import (
    RaydropMeter,
    IntensityMeter,
    DepthMeter,
    PointsMeter,
    TrajectoryMeter,
)
def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark=False 
    torch.backends.cudnn.deterministic=True
def get_arg_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config",is_config_file=True,default="configs/nus_samples_0.txt")
    parser.add_argument(
        "--cluster_summary_path",
        type=str,
        default="/summary",
        help="Overwrite default summary path if on cluster",
    )

#Dataset and sequences
    # start of the sequence
    parser.add_argument("--start",type=int)
    #evaluate all frames(not just testset): both poses and NVS
    parser.add_argument("--all_eval",action="store_true")
    #dataset path
    parser.add_argument("--path", type=str, default='./data/nuscenes')
    #dataset
    parser.add_argument("--dataloader", type=str, choices=("kitti360","nuscenes"), default="nuscenes")

#Our method
    # using Selective-Reweighting Strategy
    parser.add_argument("--reweight",action="store_true")
    # using Geometry-Constraints
    parser.add_argument("--geo_loss",action="store_true")
    # using Geo-optimizer
    parser.add_argument("--graph_optim",action="store_true")
    # coarse-to-fine 
    parser.add_argument("--c2f", type=float, nargs=2, default=[0, 0.8])

#Initialization
    # optimize rotation parameters
    parser.add_argument("--rot", action="store_true")
    # optimize translation parameters
    parser.add_argument("--trans", action="store_true")
    # add disturbance to GT poses
    parser.add_argument("--noise_rot", action="store_true")
    parser.add_argument("--noise_trans", action="store_true")
    parser.add_argument("--rot_value", default=0.151, type=float) # An average rotation error of 8.65 degrees in each axis.  20 degrees
    parser.add_argument("--trans_value", default=2.0, type=float) # An average translation error of 2 meters in each axis.   3.46 meters
    # all poses are initialized to Identity matrix
    parser.add_argument("--no_gt_pose",action="store_true")


#Network 
    ### lidar-nerf
    #depth
    parser.add_argument("--alpha_d", type=float, default=1e3)
    #raydrop
    parser.add_argument("--alpha_r", type=float, default=1)
    #intensity
    parser.add_argument("--alpha_i", type=float, default=1)
    #hash-grid
    parser.add_argument("--desired_resolution",type=int,default=2048,help="TCN finest resolution at the smallest scale")
    parser.add_argument("--log2_hashmap_size", type=int, default=19)
    parser.add_argument("--n_features_per_level", type=int, default=2)
    #sigmanet
    parser.add_argument("--num_layers", type=int, default=2, help="num_layers of sigmanet")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden_dim of sigmanet")
    parser.add_argument("--geo_feat_dim", type=int, default=15, help="geo_feat_dim of sigmanet")
    parser.add_argument("--num_rays_lidar",type=int,default=4096,help="num rays sampled per image for each training step")
    #test/eval
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--test_eval", action="store_true", help="test and eval mode")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--seed", type=int, default=0)
    ### network backbone options
    parser.add_argument("--fp16", action="store_true", help="use amp mixed precision training")
    parser.add_argument("--tcnn", action="store_true", help="use TCNN backend")

#loss
    parser.add_argument(
        "--depth_loss", type=str, default="l1", help="l1, bce, mse, huber"
    )
    parser.add_argument(
        "--intensity_loss", type=str, default="mse", help="l1, bce, mse, huber"
    )
    parser.add_argument(
        "--raydrop_loss", type=str, default="mse", help="l1, bce, mse, huber"
    )

# Method of sample rays: this allows different ways of sampling, e.g. sample patches rather than random pixels
# NOTE: It is important for geometric constraints.
    parser.add_argument(
        "--patch_size_lidar",
        type=int,
        default=1,
        help="[experimental] render patches in training. "
        "1 means disabled, use [64, 32, 16] to enable",
    )
    parser.add_argument(
        "--change_patch_size_lidar",
        nargs="+",
        type=int,
        default=[32, 128],
        help="[experimental] render patches in training. "
        "1 means disabled, use [64, 32, 16] to enable, change during training",
    )
    parser.add_argument(
        "--change_patch_size_epoch",
        type=int,
        default=2,
        help="change patch_size intenvel",
    )

    # training options
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--iters", type=int,default=30000,help="training iters")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--num_rays", type=int,default=4096,help="num rays sampled per image for each training step")
    parser.add_argument("--num_steps", type=int, default=768, help="num steps sampled per ray")
    parser.add_argument("--upsample_steps", type=int, default=64, help="num steps up-sampled per ray")
    parser.add_argument("--max_ray_batch",type=int,default=4096,help="batch size of rays at inference to avoid OOM)")
    parser.add_argument("--color_space",type=str,default="srgb",help="Color space, supports (linear, srgb)")
    parser.add_argument("--preload",action="store_true",help="preload all data into GPU, accelerate training but use more GPU memory")

    # others
    parser.add_argument("--bound",type=float,default=2,help="assume the scene is bounded in box[-bound, bound]^3,if > 1, will invoke adaptive ray marching.")
    parser.add_argument("--scale",type=float,default=0.33,help="scale location into box[-bound, bound]^3")
    parser.add_argument("--offset",type=float,nargs="*",default=[0, 0, 0],help="offset of camera location")
    parser.add_argument("--min_near", type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument("--min_near_lidar",type=float,default=0.01,help="minimum near distance for LiDAR")
    parser.add_argument("--density_thresh",type=float,default=10,help="threshold for density grid to be occupied")
    parser.add_argument("--bg_radius",type=float,default=-1,help="if positive, use a background model at sphere(bg_radius)")

    return parser


def set_opt_device_dataset_model():
    set_seed(1)
    parser = get_arg_parser()
    opt = parser.parse_args()    
    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    opt.device=device
    opt.fp16 = True
    opt.tcnn = True
    opt.preload = True
    opt.min_near_lidar = opt.scale
    assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"


    # specify dataloader size
    if opt.dataloader=="kitti360":
        opt.dataloader_size=36
    elif opt.dataloader=="nuscenes":
        opt.dataloader_size=36
    else:
        raise RuntimeError("Please specify the dataset")

    # specify dataloader
    if opt.dataloader == "kitti360":
        from data.dataset.kitti360_dataset_barf import kitti360Dataset as NeRFDataset
    elif opt.dataloader == "nuscenes":
        from data.dataset.nus_dataset_barf import  NusDataset as NeRFDataset
    else:
        raise RuntimeError("Should not reach here.")
    
    # save args
    os.makedirs(opt.workspace, exist_ok=True)
    f = os.path.join(opt.workspace, "args.txt")
    with open(f, "w") as file:
        for arg in vars(opt):
            attr = getattr(opt, arg)
            file.write("{} = {}\n".format(arg, attr))


    # specify model
    from GeoNLF.network_tcnn import NeRFNetwork
    model = NeRFNetwork(
        opt,
        device=device,
        desired_resolution=opt.desired_resolution,
        log2_hashmap_size=opt.log2_hashmap_size,
        n_features_per_level=opt.n_features_per_level,
        num_layers=opt.num_layers,
        hidden_dim=opt.hidden_dim,
        geo_feat_dim=opt.geo_feat_dim,
        bound=opt.bound,
        density_scale=1,
        min_near=opt.min_near,
        min_near_lidar=opt.min_near_lidar,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    return opt,device,NeRFDataset,model

def set_loss(opt):
    loss_dict = {
        "mse": torch.nn.MSELoss(reduction="none"),
        "l1": torch.nn.L1Loss(reduction="none"),
        "bce": torch.nn.BCEWithLogitsLoss(reduction="none"),
        "huber": torch.nn.HuberLoss(reduction="none", delta=0.2 * opt.scale),
        "cos": torch.nn.CosineSimilarity(),
    }
    # depth_loss = l1,depth_grad_loss = l1,intensity_loss = mse,raydrop_loss = mse
    criterion = {
        "depth": loss_dict[opt.depth_loss],
        "raydrop": loss_dict[opt.raydrop_loss],
        "intensity": loss_dict[opt.intensity_loss],
    }
    return criterion

def test_mode(opt,device,NeRFDataset,model,criterion):

    if opt.dataloader == "nuscenes":
        intrinsics=[10.0,40.0]
    elif opt.dataloader == "kitti360":
        intrinsics=[2.0,26.9]
    else:
        raise RuntimeError("Please specify the dataset")
    
    test_loader = NeRFDataset(
            device=device,
            split="test",
            root_path=opt.path,
            sequence_id=opt.start,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
        ).dataloader()
    

    depth_metrics = [
        RaydropMeter(ratio=0.5),
        IntensityMeter(scale=1.0),
        DepthMeter(scale=opt.scale),
        IntensityMeter(scale=1.0),
        TrajectoryMeter(scale=opt.scale,offset=opt.offset),
        DepthMeter(scale=opt.scale),
        PointsMeter(scale=opt.scale,intrinsics=intrinsics),
    ]

    trainer = Trainer(
        "lidar_nerf",
        opt,
        model,
        test_loader,
        Geo_optimizer,
        device=device,
        workspace=opt.workspace,
        criterion=criterion,
        fp16=opt.fp16,
        depth_metrics=depth_metrics,
        use_checkpoint=opt.ckpt,
    )
    if test_loader.has_gt and opt.test_eval:
        trainer.evaluate(test_loader)  # blender has gt, so evaluate it.
    else:
        pass

    trainer.test(test_loader)  # test

def train_mode(opt,device,NeRFDataset,model,criterion):

    if opt.dataloader == "nuscenes":
        intrinsics=[10.0,40.0]
    elif opt.dataloader == "kitti360":
        intrinsics=[2.0,26.9]
    else:
        raise RuntimeError("Please specify the dataset")
    
    optimizer = lambda model: torch.optim.Adam(
            model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15
        )
    optimizer_pose_trans=lambda model: torch.optim.Adam(
        model.get_params_pose_trans(opt.lr), betas=(0.9, 0.99), eps=1e-15
    )
    optimizer_pose_rot=lambda model: torch.optim.Adam(
        model.get_params_pose_rot(opt.lr), betas=(0.9, 0.99), eps=1e-15
    )

    # scheduler：
    # decay to 0.1 * init_lr at last iter step for NeRF， 
    # decay to 0.01 * init_lr at last iter step for Pose
    scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1)
    )
    if opt.no_gt_pose:
        scheduler_pose_rot = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.01 ** min(iter / opt.iters, 1) 
        )
        schedeuler_pose_trans = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.01 ** min(iter / opt.iters, 1)
        ) 
    else:
        scheduler_pose_rot = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.01 ** min(iter / opt.iters, 1) 
        )
        schedeuler_pose_trans = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.01 ** min(iter / opt.iters, 1)
        ) 


    # train_loader
    train_loader = NeRFDataset(
        device=device,
        split="train",
        root_path=opt.path,
        sequence_id=opt.start,
        preload=opt.preload,
        scale=opt.scale,
        offset=opt.offset,
        fp16=opt.fp16,
        patch_size_lidar=opt.patch_size_lidar,
        num_rays_lidar=opt.num_rays_lidar,
    ).dataloader()

    valid_loader = NeRFDataset(
        device=device,
        split="val",
        root_path=opt.path,
        sequence_id=opt.start,
        preload=opt.preload,
        scale=opt.scale,
        offset=opt.offset,
        fp16=opt.fp16,
        patch_size_lidar=opt.patch_size_lidar,
        num_rays_lidar=opt.num_rays_lidar,
    ).dataloader()

    test_loader = NeRFDataset(
        device=device,
        split="test",
        root_path=opt.path,
        sequence_id=opt.start,
        preload=opt.preload,
        scale=opt.scale,
        offset=opt.offset,
        fp16=opt.fp16,
        patch_size_lidar=opt.patch_size_lidar,
        num_rays_lidar=opt.num_rays_lidar,
    ).dataloader()

    depth_metrics = [
        RaydropMeter(ratio=0.5),
        IntensityMeter(scale=1.0),
        DepthMeter(scale=opt.scale),
        IntensityMeter(scale=1.0),
        TrajectoryMeter(scale=opt.scale,offset=opt.offset),
        DepthMeter(scale=opt.scale),
        PointsMeter(scale=opt.scale,intrinsics=intrinsics),

    ]


    trainer = Trainer(
        "lidar_nerf",
        opt,
        model,
        train_loader,
        Geo_optimizer,
        device=device,
        workspace=opt.workspace,
        optimizer=optimizer,
        optimizer_pose_rot=optimizer_pose_rot,
        optimizer_pose_trans=optimizer_pose_trans,
        criterion=criterion,
        ema_decay=0.95,
        fp16=opt.fp16,
        lr_scheduler=scheduler,
        lr_scheduler_pose_rot=scheduler_pose_rot,
        lr_scheduler_pose_trans=schedeuler_pose_trans,
        scheduler_update_every_step=True,
        depth_metrics=depth_metrics,
        use_checkpoint=opt.ckpt,
        eval_interval=opt.eval_interval,
    )
    
    
    
    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    print(f"max_epoch: {max_epoch}")
    trainer.train(train_loader,test_loader,max_epoch)
    #trainer.recoder.save_train_pose(test_loader)
    trainer.test(test_loader) 
    #trainer.save_mesh(resolution=128, threshold=10)
def main():
    #这个函数就是根据parser来确定device这些量的
    opt,device,NeRFDataset,model=set_opt_device_dataset_model()
    #根据parser来确定损失函数
    criterion=set_loss(opt)
    #如果是测试/评估模式
    if opt.test or opt.test_eval:
        test_mode(opt,device,NeRFDataset,model,criterion)
    else:
        train_mode(opt,device,NeRFDataset,model,criterion)
        
if __name__ == "__main__":
    main()
