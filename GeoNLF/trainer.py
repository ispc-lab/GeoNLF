import glob
import os
import random
import time
import sys
import cv2
import imageio
import lpips
import open3d as o3d
import mcubes
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import trimesh
import time
from rich.console import Console
from skimage.metrics import structural_similarity

from loss.custom_loss import chamfer_distance_low_capacity
from loss.custom_loss import chamfer_based_norm_loss_low_capacity
from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.metrics import fscore
from data.dataset.base_dataset import custom_meshgrid
from utils.convert import pano_to_lidar
from utils.recorder import recoder

def is_ali_cluster():
    import socket
    hostname = socket.gethostname()
    return "auto-drive" in hostname


class Trainer(object):
    def __init__(
        self,
        name,  # name of this experiment
        opt,  # extra conf
        model,  # network
        loader,
        Geo_optimizer,
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        optimizer_pose_rot=None,
        optimizer_pose_trans=None,
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        lr_scheduler_pose_rot=None,
        lr_scheduler_pose_trans=None,
        metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        depth_metrics=[],
        local_rank=0,  # which GPU am I
        world_size=1,  # total num of GPUs
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        eval_interval=50,  # eval once every $ epoch
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metric
        report_metric_at_train=False,  # also report metrics at training
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
    ):
        self.pcds=None
        self.poses=None
        self.it=0
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.depth_metrics = depth_metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()

        self.optim_direction_rot = None
        self.optim_direction_trans = None
        self.optim_gradients_trans=[[] for i in range(self.opt.dataloader_size)]
        self.optim_gradients_rot=[[] for i in range(self.opt.dataloader_size)]

        
        model.to(self.device)
        self.geo_optimizer=Geo_optimizer(opt,loader,model)
        self.recoder=recoder(model)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )
        self.model = model
        

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=0.001, weight_decay=5e-4
            )  # naive adam
        else:
            self.optimizer = optimizer(self.model)
            self.optimizer_pose_rot=optimizer_pose_rot(self.model)
            self.optimizer_pose_trans=optimizer_pose_trans(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)
            self.lr_scheduler_pose_rot=lr_scheduler_pose_rot(self.optimizer_pose_rot)
            self.lr_scheduler_pose_trans=lr_scheduler_pose_trans(self.optimizer_pose_trans)

        if ema_decay is not None:
            #self.ema = ExponentialMovingAverage(
                #self.model.parameters(), decay=ema_decay
            #)
            self.ema=None
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = "min"

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(
            f"[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}"
        )

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()
    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
            "it": self.it,#其实跟global step一样，当时没看到，有点多此一举了
        }

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["optimizer_pose_rot"]=self.optimizer_pose_rot.state_dict()
            state["optimizer_pose_trans"]=self.optimizer_pose_trans.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["lr_scheduler_pose_rot"]=self.lr_scheduler_pose_rot.state_dict()
            state["lr_scheduler_pose_trans"]=self.lr_scheduler_pose_trans.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()
        
        if not best:
            state["model"] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if (
                    self.stats["best_result"] is None
                    or self.stats["results"][-1] < self.stats["best_result"]
                ):
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if "density_grid" in state["model"]:
                        del state["model"]["density_grid"]

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        if model_only:
            return
        self.it = checkpoint_dict["it"]
        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"]
        self.global_step = checkpoint_dict["global_step"]
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.optimizer_pose_rot.load_state_dict(checkpoint_dict["optimizer_pose_rot"])
                self.optimizer_pose_trans.load_state_dict(checkpoint_dict["optimizer_pose_trans"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.lr_scheduler_pose_rot.load_state_dict(checkpoint_dict["lr_scheduler_pose_rot"])
                self.lr_scheduler_pose_trans.load_state_dict(checkpoint_dict["lr_scheduler_pose_trans"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
      
    def train_step(self, data):

        pred_intensity = None
        gt_intensity = None
        pred_depth = None
        gt_depth = None
        loss = 0
        geo_flag=False

        if isinstance(self.opt.patch_size_lidar, int):
            #非patch采样,采样1个pixcel
            patch_size_x, patch_size_y = (
                self.opt.patch_size_lidar,
                self.opt.patch_size_lidar,
            )
        elif len(self.opt.patch_size_lidar) == 1:
            #不会经过这儿
            patch_size_x, patch_size_y = (
                self.opt.patch_size_lidar[0],
                self.opt.patch_size_lidar[0],
            )
        else:
            #patch采样
            patch_size_x, patch_size_y = self.opt.patch_size_lidar
            geo_flag=True

        if not self.opt.geo_loss or not geo_flag:
            outputs_lidar = self.model.render(
                data,
                staged=False,
                perturb=True,
                **vars(self.opt),
            )
            self.it+=1
            self.model.progress.data.fill_(self.it/self.opt.iters)
            self.model.progress.requires_grad=False
            image_lidar_sampled=outputs_lidar["image_lidar_sampled"]
            gt_raydrop = image_lidar_sampled[:, :, 0]
            gt_intensity = image_lidar_sampled[:, :, 1] * gt_raydrop
            gt_depth = image_lidar_sampled[:, :, 2] * gt_raydrop

            pred_raydrop = outputs_lidar["intensity"][:, :, 0]
            pred_intensity = outputs_lidar["intensity"][:, :, 1] * gt_raydrop
            pred_depth = outputs_lidar["depth_lidar"] * gt_raydrop
            lidar_loss = (
                self.opt.alpha_d * self.criterion["depth"](pred_depth, gt_depth)
                + self.opt.alpha_r * self.criterion["raydrop"](pred_raydrop, gt_raydrop)
                + self.opt.alpha_i * self.criterion["intensity"](pred_intensity, gt_intensity)
            )
            loss=lidar_loss
            if len(loss.shape) == 3:  # [K, B, N]
                loss = loss.mean(0)
            loss = loss.mean()


        elif self.opt.geo_loss and geo_flag:
            outputs_lidar = self.model.render(
                data,
                staged=False,
                perturb=True,
                **vars(self.opt),
            )

            self.it+=1
            self.model.progress.data.fill_(self.it/self.opt.iters)
            self.model.progress.requires_grad=False

            image_lidar_sampled=outputs_lidar["image_lidar_sampled"]
            gt_raydrop = image_lidar_sampled[:, :, 0]
            gt_intensity = image_lidar_sampled[:, :, 1] * gt_raydrop
            gt_depth = image_lidar_sampled[:, :, 2] * gt_raydrop

            pred_raydrop = outputs_lidar["intensity"][:, :, 0]
            pred_intensity = outputs_lidar["intensity"][:, :, 1] * gt_raydrop
            pred_depth = outputs_lidar["depth_lidar"] * gt_raydrop
            #raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)

            lidar_loss = (
                self.opt.alpha_d * self.criterion["depth"](pred_depth, gt_depth).mean()
                + self.opt.alpha_r
                * self.criterion["raydrop"](pred_raydrop, gt_raydrop).mean()
                + self.opt.alpha_i
                * self.criterion["intensity"](pred_intensity, gt_intensity).mean()
            )

            loss=lidar_loss
            if len(loss.shape) == 3:  # [K, B, N]
                loss = loss.mean(0)
            loss = loss.mean()

            pcd1 = pano_to_lidar(gt_depth, data["intrinsics_lidar"],is_tensor=True)/self.opt.scale
            pcd2 = pano_to_lidar(pred_depth, data["intrinsics_lidar"],is_tensor=True)/self.opt.scale
            pcd1=pcd1.unsqueeze(0)
            pcd2=pcd2.unsqueeze(0)
            custom_3D_loss_cd,idx1,idx2=chamfer_distance_low_capacity(pcd1,pcd2)
            custom_3D_loss_norm=chamfer_based_norm_loss_low_capacity(pcd1,pcd2,idx1,idx2)

            custom_3D_loss=(custom_3D_loss_cd+custom_3D_loss_norm)
            loss=0.8*loss+0*custom_3D_loss

        else:
            lidar_loss = 0
            loss = lidar_loss

        return (
            pred_intensity,
            gt_intensity,
            pred_depth,
            gt_depth,
            loss,
        )
    def eval_step(self, data):
        pred_intensity = None
        pred_depth = None
        pred_depth_crop = None
        pred_raydrop = None
        gt_intensity = None
        gt_depth = None
        gt_depth_crop = None
        gt_raydrop = None
        loss = 0

        outputs_lidar = self.model.render(
            data,
            cal_lidar_color=True,
            staged=True,
            perturb=False,
            **vars(self.opt),
        )


        image=outputs_lidar["image_lidar"]
        gt_raydrop = image[:, :, :, 0]
        gt_intensity = image[:, :, :, 1] * gt_raydrop
        gt_depth = image[:, :, :, 2] * gt_raydrop
        B_lidar, H_lidar, W_lidar, C_lidar = image.shape
        
        pred_rgb_lidar = outputs_lidar["intensity"].reshape(
            B_lidar, H_lidar, W_lidar, 2
        )
        pred_raydrop = pred_rgb_lidar[:, :, :, 0]
        raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)

        pred_intensity = pred_rgb_lidar[:, :, :, 1]
        pred_depth = outputs_lidar["depth_lidar"].reshape(B_lidar, H_lidar, W_lidar)

        lidar_loss = (
            self.opt.alpha_d * self.criterion["depth"](pred_depth * raydrop_mask, gt_depth).mean()
            + self.opt.alpha_r
            * self.criterion["raydrop"](pred_raydrop, gt_raydrop).mean()
            + self.opt.alpha_i
            * self.criterion["intensity"](pred_intensity * raydrop_mask, gt_intensity).mean()
        )

        loss = lidar_loss

        return (
            pred_intensity,
            pred_depth,
            pred_depth_crop,
            pred_raydrop,
            gt_intensity,
            gt_depth,
            gt_depth_crop,
            gt_raydrop,
            loss,
        )
    def test_step(self, data, bg_color=None, perturb=False):
        pred_raydrop = None
        pred_intensity = None
        pred_depth = None
        

        H_lidar, W_lidar = data["H_lidar"], data["W_lidar"]
        outputs_lidar = self.model.render(
            data,
            cal_lidar_color=True,
            staged=True,
            perturb=False,
            **vars(self.opt),
        )

        pred_rgb_lidar = outputs_lidar["intensity"].reshape(
            -1, H_lidar, W_lidar, 2
        )
        pred_raydrop = pred_rgb_lidar[:, :, :, 0]
        raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)
        pred_intensity = pred_rgb_lidar[:, :, :, 1]
        pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)

        pred_intensity = pred_intensity * raydrop_mask
        pred_depth = pred_depth * raydrop_mask
        
        return pred_raydrop, pred_intensity, pred_depth
        
    def train(self, train_loader, test_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            if is_ali_cluster() and self.opt.cluster_summary_path is not None:
                summary_path = self.opt.cluster_summary_path
            else:
                summary_path = os.path.join(self.workspace, "run", self.name)
            self.writer = tensorboardX.SummaryWriter(summary_path)


        change_dataloder = False
        if self.opt.change_patch_size_lidar[0] > 1:
            change_dataloder = True
        for epoch in range(self.epoch + 1, max_epochs + 1):
            if self.opt.trans or self.opt.rot:
                lr_trans=self.lr_scheduler_pose_trans.get_last_lr()[0]
                lr_rot=self.lr_scheduler_pose_rot.get_last_lr()[0]
                self.geo_optimizer.geo_optimize(self.epoch,lr_trans,lr_rot)
            self.epoch = epoch
            if change_dataloder:
                if self.epoch % self.opt.change_patch_size_epoch == 0:
                    train_loader._data.patch_size_lidar = (
                        self.opt.change_patch_size_lidar
                    )
                    self.opt.patch_size_lidar = self.opt.change_patch_size_lidar
                else:
                    train_loader._data.patch_size_lidar = 1
                    self.opt.patch_size_lidar = 1
            self.train_one_epoch(train_loader)
            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)
            if self.epoch % self.eval_interval == 0: #self.eval_interval
                self.evaluate_one_epoch(test_loader)
                self.save_checkpoint(full=False, best=True)
            if self.epoch%5==0 or self.epoch==1:
                self.recoder.save_train_pose(train_loader)
            #self.geo_optimize(train_loader)

 
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()
    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX
    def test(self, loader, save_path=None, name=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval()

        # if write_video:
        #     all_preds = []
        #     all_preds_depth = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    #_=self.test_step(data)
                    preds_raydrop, preds_intensity, preds_depth = self.test_step(data)
                #'''
                pred_raydrop = preds_raydrop[0].detach().cpu().numpy()
                pred_raydrop = (np.where(pred_raydrop > 0.5, 1.0, 0.0)).reshape(
                    loader._data.H_lidar, loader._data.W_lidar
                )
                pred_raydrop = (pred_raydrop * 255).astype(np.uint8)

                pred_intensity = preds_intensity[0].detach().cpu().numpy()
                pred_intensity = (pred_intensity * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_lidar = pano_to_lidar(
                    pred_depth / self.opt.scale, loader._data.intrinsics_lidar
                )
 
                np.save(
                    os.path.join(save_path, f"test_{name}_{i:04d}_depth_lidar.npy"),
                    pred_lidar,
                )

                pred_depth = (pred_depth * 255).astype(np.uint8)
                # pred_depth = (pred_depth / self.opt.scale).astype(np.uint8)

                # if write_video:
                #     all_preds.append(cv2.cvtColor(cv2.applyColorMap(pred_intensity, 1), cv2.COLOR_BGR2RGB))
                #     all_preds_depth.append(cv2.cvtColor(cv2.applyColorMap(pred_depth, 20), cv2.COLOR_BGR2RGB))
                # else:
                cv2.imwrite(
                    os.path.join(save_path, f"test_{name}_{i:04d}_raydrop.png"),
                    pred_raydrop,
                )
                cv2.imwrite(
                    os.path.join(
                        save_path, f"test_{name}_{i:04d}_intensity.png"
                    ),
                    cv2.applyColorMap(pred_intensity, 1),
                )
                cv2.imwrite(
                    os.path.join(save_path, f"test_{name}_{i:04d}_depth.png"),
                    cv2.applyColorMap(pred_depth, 20),
                )

                pbar.update(loader.batch_size)
                #'''
        #'''
        # if write_video:
        #     all_preds = np.stack(all_preds, axis=0)
        #     all_preds_depth = np.stack(all_preds_depth, axis=0)
        #     imageio.mimwrite(
        #         os.path.join(save_path, f"{name}_lidar_rgb.mp4"),
        #         all_preds,
        #         fps=25,
        #         quality=8,
        #         macro_block_size=1,
        #     )
        #     imageio.mimwrite(
        #         os.path.join(save_path, f"{name}_depth.mp4"),
        #         all_preds_depth,
        #         fps=25,
        #         quality=8,
        #         macro_block_size=1,
        #     )

        self.log(f"==> Finished Test.")
        #'''
    def train_one_epoch(self, loader):
        self.optimizer.param_groups[0]['params'][0].requires_grad=True
        self.optimizer.param_groups[1]['params'][0].requires_grad=True
        self.optimizer.param_groups[2]['params'][0].requires_grad=True
        self.optimizer.param_groups[3]['params'][0].requires_grad=True

        log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log(
            f"[{log_time}] ==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()
            for metric in self.depth_metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        self.local_step = 0
        
        idx_test=[8,20,32]
        # idx_test=list(range(100))

        for data in loader:
            if self.epoch==1:
                self.recoder.cal_pose_error(data)
            self.local_step += 1
            self.global_step += 1

            self.optimizer_pose_rot.zero_grad()
            self.optimizer_pose_trans.zero_grad()
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.fp16):
                (
                    pred_intensity,
                    gt_intensity,
                    pred_depth,
                    gt_depth,
                    loss,
                ) = self.train_step(data)

            self.scaler.scale(loss).backward()
            # save the current learning rate
            current_lr_network=self.lr_scheduler.get_last_lr()[0]

            # the test frame can't contribute to the network, but we optimize their pose
            
            if data["index"] in idx_test:
                # compared to 768, the network can't be optimized because we set the learning rate=0
                # in this way, we can simply just optimize pose as iNeRF.
                # print(self.optimizer.param_groups[0]) 
                self.optimizer.param_groups[0]['lr']=0
                self.optimizer.param_groups[1]['lr']=0
                self.optimizer.param_groups[2]['lr']=0
                self.optimizer.param_groups[3]['lr']=0


            # selective-reweighting
            if self.opt.reweight:
                record=[i[-5:] for i in self.model.loss_record]
                mean_loss=[]
                for l in record:
                    meanl=sum(l)/(len(l)+0.0001)
                    mean_loss.append(meanl)
                sorted_pairs=sorted(zip(mean_loss,[i for i in range(len(mean_loss))]))
                mean_loss,idx=zip(*sorted_pairs)
                indx_loss=idx[-3:]
                if data["index"] not in idx_test and data["index"] in indx_loss and 400>self.epoch:
                    if self.epoch<10:
                        reweight_loss=0.1
                    else:
                        reweight_loss=min(0.15+0.85*self.epoch/400,1)
                    self.optimizer.param_groups[0]['lr']=current_lr_network*reweight_loss
                    self.optimizer.param_groups[1]['lr']=current_lr_network*reweight_loss
                    self.optimizer.param_groups[2]['lr']=current_lr_network*reweight_loss
                    self.optimizer.param_groups[3]['lr']=current_lr_network*reweight_loss


            self.scaler.step(self.optimizer)
            # if data["index"] in idx_test:
            #     print(self.optimizer.param_groups[0])
            self.optimizer.param_groups[0]['lr']=current_lr_network
            self.optimizer.param_groups[1]['lr']=current_lr_network
            self.optimizer.param_groups[2]['lr']=current_lr_network
            self.optimizer.param_groups[3]['lr']=current_lr_network

            if self.opt.rot:
                self.scaler.step(self.optimizer_pose_rot)
            if self.opt.trans:
                self.scaler.step(self.optimizer_pose_trans)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
                if self.opt.rot:
                    self.lr_scheduler_pose_rot.step()
                if self.opt.trans:
                    self.lr_scheduler_pose_trans.step()

            loss_val = loss.item()
            self.recoder.cal_pose_error(data)
            self.model.loss_record[data["index"]].append(loss_val)
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for i, metric in enumerate(self.depth_metrics):
                        if i < 2:  # hard code
                            metric.update(pred_intensity, gt_intensity)
                        else:
                            metric.update(pred_depth, gt_depth)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                pbar.update(loader.batch_size)
        


        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)
        self.log(f"average_loss: {average_loss}.")

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.depth_metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="LiDAR_train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
                if self.opt.rot:
                    self.lr_scheduler_pose_rot.step(average_loss)
                if self.opt.trans:
                    self.lr_scheduler_pose_trans.step(average_loss)
            else:
                self.lr_scheduler.step()
                if self.opt.rot:
                    self.lr_scheduler_pose_rot.step()
                if self.opt.trans:
                    self.lr_scheduler_pose_trans.step()

        self.log(f"==> Finished Epoch {self.epoch}.")
    def evaluate_one_epoch(self, loader, name=None):

        self.log(f"++> Evaluate at epoch {self.epoch} ...")
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()
            for metric in self.depth_metrics:
                metric.clear()

        self.model.eval()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        with torch.no_grad():
            self.local_step = 0
            #prepare for trajectory evaluation
            pred_poses=np.zeros((self.opt.dataloader_size,4,4))
            gt_poses=np.zeros((self.opt.dataloader_size,4,4))
            for data in loader:
                idx=data["index"] 
                pred_pose=self.model.get_pose(data["index"],data["pose"]) #1 4 4
                pred_poses[idx,:,:]=pred_pose.cpu().numpy()
                gt_poses[idx,:,:]=data["pose"].cpu().numpy()
            self.depth_metrics[4].update(pred_poses,gt_poses)
            for data in loader:
                if self.opt.all_eval:
                    eva_idx=[i for i in range(self.opt.dataloader_size)]
                else:
                    eva_idx=[6,18,30]
                if data["index"] in eva_idx:
                    self.local_step += 1

                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        (   
                            preds_intensity,
                            preds_depth,
                            preds_depth_crop,
                            preds_raydrop,
                            gt_intensity,
                            gt_depth,
                            gt_depth_crop,
                            gt_raydrop,
                            loss,
                        ) = self.eval_step(data)
                    

                    preds_mask = torch.where(preds_raydrop > 0.5, 1, 0)
                    gt_mask = gt_raydrop

                    # all_gather/reduce the statistics (NCCL only support all_*)
                    if self.world_size > 1:
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss = loss / self.world_size

                        preds_list = [
                            torch.zeros_like(preds).to(self.device)
                            for _ in range(self.world_size)
                        ]  # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_list, preds)
                        preds = torch.cat(preds_list, dim=0)

                        preds_depth_list = [
                            torch.zeros_like(preds_depth).to(self.device)
                            for _ in range(self.world_size)
                        ]  # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_depth_list, preds_depth)
                        preds_depth = torch.cat(preds_depth_list, dim=0)

                        truths_list = [
                            torch.zeros_like(truths).to(self.device)
                            for _ in range(self.world_size)
                        ]  # [[B, ...], [B, ...], ...]
                        dist.all_gather(truths_list, truths)
                        truths = torch.cat(truths_list, dim=0)

                    loss_val = loss.item()
                    total_loss += loss_val

                    # only rank = 0 will perform evaluation.
                    if self.local_rank == 0:
                        for i, metric in enumerate(self.depth_metrics):
                            if i == 0:  # hard code
                                metric.update(preds_raydrop, gt_raydrop)
                            elif i == 1:
                                metric.update(preds_intensity*gt_mask, gt_intensity)
                            elif i == 2:
                                metric.update(preds_depth*gt_mask, gt_depth)
                            elif i == 3:
                                metric.update(preds_intensity*preds_mask, gt_intensity)
                            elif i==4:
                                pass
                            else:
                                metric.update(preds_depth*preds_mask, gt_depth)
                                #metric.update(preds_depth*gt_mask, gt_depth)

                        save_path_pred = os.path.join(
                            self.workspace,
                            "validation",
                            f"{name}_{self.local_step:04d}.png",
                        )
                        os.makedirs(os.path.dirname(save_path_pred), exist_ok=True)

                        pred_raydrop = preds_raydrop[0].detach().cpu().numpy()
                        # pred_raydrop = (np.where(pred_raydrop > 0.5, 1.0, 0.0)).reshape(
                        #     loader._data.H_lidar, loader._data.W_lidar
                        # )
                        img_raydrop = (pred_raydrop * 255).astype(np.uint8)
                        img_raydrop = cv2.cvtColor(img_raydrop, cv2.COLOR_GRAY2BGR)

                        pred_intensity = preds_intensity[0].detach().cpu().numpy()
                        img_intensity = (pred_intensity * 255).astype(np.uint8)
                        img_intensity = cv2.applyColorMap(img_intensity, 1) #1, 10, 14, 15
                        
                        pred_depth = preds_depth[0].detach().cpu().numpy()
                        img_depth = (pred_depth * 255).astype(np.uint8)
                        # img_depth = (pred_depth / self.opt.scale).astype(np.uint8)
                        img_depth = cv2.applyColorMap(img_depth, 20)

                        preds_mask = preds_mask[0].detach().cpu().numpy()
                        img_mask = (preds_mask * 255).astype(np.uint8)
                        img_raydrop_masked = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

                        img_intensity_masked = (pred_intensity * preds_mask * 255).astype(np.uint8)
                        img_intensity_masked = cv2.applyColorMap(img_intensity_masked, 1) #1, 10, 14, 15
                        
                        img_depth_masked = (pred_depth * preds_mask * 255).astype(np.uint8)
                        img_depth_masked = cv2.applyColorMap(img_depth_masked, 20)

                        img_pred = cv2.vconcat([img_raydrop, img_intensity, img_depth, 
                                                img_raydrop_masked, img_intensity_masked, img_depth_masked])
                        cv2.imwrite(save_path_pred, img_pred)

                        #pred_lidar = pano_to_lidar(pred_depth / self.opt.scale, loader._data.intrinsics_lidar)
                        pred_lidar = pano_to_lidar(pred_depth * preds_mask / self.opt.scale, loader._data.intrinsics_lidar)
        
                        
                        np.save(
                            os.path.join(
                                self.workspace,
                                "validation",
                                f"{name}_{self.local_step:04d}_lidar.npy",
                            ),
                            pred_lidar,
                        )

                        pbar.set_description(
                            f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                        )
                        pbar.update(loader.batch_size)
                else:
                    pass
        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if len(self.depth_metrics) > 0:
                # result = self.metrics[0].measure()
                result = self.depth_metrics[-1].measure()[0]  # hard code
                self.stats["results"].append(
                    result if self.best_mode == "min" else -result
                )  # if max mode, use -result
            else:
                self.stats["results"].append(
                    average_loss
                )  # if no metric, choose best by min loss

            np.set_printoptions(linewidth=150, suppress=True, precision=8)
            for i, metric in enumerate(self.depth_metrics):
                if i == 4:
                    continue
                if i == 1:
                    self.log(f"=== ↓ GT mask ↓ ==== RMSE{' '*6}MedAE{' '*8}a1{' '*10}a2{' '*10}a3{' '*8}LPIPS{' '*8}SSIM{' '*8}PSNR ===")
                if i == 3:
                    self.log(f"== ↓ Final pred ↓ == RMSE{' '*6}MedAE{' '*8}a1{' '*10}a2{' '*10}a3{' '*8}LPIPS{' '*8}SSIM{' '*8}PSNR ===")
                self.log(metric.report(), style="blue")
                metric.clear()
            self.log(self.depth_metrics[4].report(), style="blue")
            self.depth_metrics[4].clear()



        self.log(f"++> Evaluate epoch {self.epoch} Finished.")
    
