import json
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from GeoNLF.dataset.base_dataset import get_lidar_rays, BaseDataset


@dataclass
class kitti360Dataset(BaseDataset):
    sequence_id: int = 1908
    device: str = "cpu"
    split: str = "train"  # train, val, test
    root_path: str = './data/kitti360'
    preload: bool = True  # preload data into GPU
    scale: float = (
        1  # camera radius scale to make sure camera are inside the bounding box.
    )
    offset: list = field(default_factory=list)  # offset
    # bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
    fp16: bool = True  # if preload, load into fp16.
    patch_size_lidar: int = 1  # size of the image to extract from the Lidar.
    num_rays: int = 4096
    num_rays_lidar: int = 4096

    def __post_init__(self):
        self.training = self.split in ["train", "all", "trainval"]
        self.num_rays = self.num_rays if self.training else -1
        self.num_rays_lidar = self.num_rays_lidar if self.training else -1
        #读文件
        # load nerf-compatible format data.
        with open(
            os.path.join(
                self.root_path, f"kitti_transforms_{self.sequence_id}.json"
            ),
            "r",
        ) as f:
            transform = json.load(f)

        if "h_lidar" in transform and "w_lidar" in transform:
            self.H_lidar = int(transform["h_lidar"])
            self.W_lidar = int(transform["w_lidar"])

        # read images
        frames = transform["frames"]
        # frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        #rangemap读取和gt的位姿
        self.poses_lidar = []
        self.images_lidar = []
        for f in tqdm.tqdm(frames, desc=f"Loading {self.split} data"):
            pose_lidar = np.array(f["lidar2world"], dtype=np.float32)  # [4, 4]
            f_lidar_path = f["lidar_file_path"]

            
            # channel1 None, channel2 intensity , channel3 depth
            pc = np.load(os.path.join(self.root_path,f_lidar_path))
            ray_drop = pc.reshape(-1, 3)[:, 2].copy()
            ray_drop[ray_drop > 0] = 1.0
            ray_drop = ray_drop.reshape(self.H_lidar, self.W_lidar, 1)
            #raydrop intensity depth，HxWx3
            image_lidar = np.concatenate(
                [ray_drop, pc[:, :, 1, None], pc[:, :, 2, None] * self.scale],
                axis=-1,
            )
        
            self.poses_lidar.append(pose_lidar)
            self.images_lidar.append(image_lidar)
        #改变平移 将去偏置后缩放
        self.poses_lidar = np.stack(self.poses_lidar, axis=0)
        self.poses_lidar[:, :3, -1] = (
            self.poses_lidar[:, :3, -1] - self.offset
        ) * self.scale
        self.poses_lidar = torch.from_numpy(self.poses_lidar)  # [N, 4, 4]


        #原本是列表，现在改写成数组，NHWC（C=3，为raydrop、instensity、depth）
        if self.images_lidar is not None:
            self.images_lidar = torch.from_numpy(
                np.stack(self.images_lidar, axis=0)
            ).float()  # [N, H, W, C]
        #print(self.images_lidar.shape)

        if self.preload:
            self.poses_lidar = self.poses_lidar.to(self.device)
            if self.images_lidar is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16:
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images_lidar = self.images_lidar.to(dtype).to(self.device)

        self.intrinsics_lidar = (2.0, 26.9)  # fov_up, fov

    def collate(self, index):
        B = len(index)  # a list of length 1

        results = {}

        pose_lidar = self.poses_lidar[index].to(self.device)  # [1, 4, 4]
        results["pose"]=pose_lidar
        results["patch"]=self.patch_size_lidar
        results["intrinsics_lidar"]=self.intrinsics_lidar
        results["num_rays_lidar"]=self.num_rays_lidar
        results["index"]=index[0]
        results["H_lidar"]=self.H_lidar
        results["W_lidar"]=self.W_lidar
        results["image_lidar"]= self.images_lidar[index].to(self.device)  # [1, H, W, 3]
        return results

    def dataloader(self):
        size = len(self.poses_lidar)
        loader = DataLoader(
            list(range(size)),
            batch_size=1,
            collate_fn=self.collate,
            shuffle=self.training,
            num_workers=0,
        )
        loader._data = self
        loader.has_gt = self.images_lidar is not None
        return loader

    def __len__(self):
        """
        Returns # of frames in this dataset.
        """
        num_frames = len(self.poses_lidar)
        return num_frames