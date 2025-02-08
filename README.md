<div align="center">
<h3>GeoNLF: Geometry guided Pose-Free Neural LiDAR Fields</h3>  

Weiyi Xue*, [Zehan Zheng*](https://dyfcalid.github.io/), [Fan Lu](https://fanlu97.github.io/), Haiyun Wei,  [Guang Chen](https://ispc-group.github.io/)†, Changjun Jiang  († Corresponding author)  
**NeurIPS 2024**
**[Paper (arXiv)](https://arxiv.org/abs/2407.05597) | [Paper (NeurIPS)](https://papers.nips.cc/paper_files/paper/2024/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html) | [Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/93231.png?t=1733402073.7734404) **  

This repository is the official PyTorch implementation for GeoNLF.

<h1><img src="https://github.com/ispc-lab/GeoNLF/assets/fig1" width=90%></h1>
<h1><img src="https://github.com/ispc-lab/GeoNLF/assets/fig2" width=90%></h1>
</div>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#changelog">Changelog</a>
    </li>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#getting-started">Getting started</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>


## Changelog
2025-02-07:🎉 Code of GeoNLF (for GT pose with noise) is released.  
2024-09-26:🎉 Our paper is accepted by NeurIPS 2024.  

## Introduction
Although recent efforts have extended Neural Radiance Fields (NeRF) into LiDAR point cloud synthesis, the majority of existing works exhibit a strong dependence on precomputed poses. However, point cloud registration methods struggle to achieve precise global pose estimation, whereas previous pose-free NeRFs overlook geometric consistency in global reconstruction. In light of this, we explore the geometric insights of point clouds, which provide explicit registration priors for reconstruction. Based on this, we propose Geometry guided Neural LiDAR Fields(GeoNLF), a hybrid framework performing alternately global neural reconstruction and pure geometric pose optimization. Furthermore, NeRFs tend to overfit individual frames and easily get stuck in local minima under sparse-view inputs. To tackle this issue, we develop a selective-reweighting strategy and introduce geometric constraints for robust optimization. Extensive experiments on NuScenes and KITTI-360 datasets demonstrate the superiority of GeoNLF in both novel view synthesis and multi-view registration of low-frequency large-scale point clouds.

## Getting started


### 🛠️ Installation

```bash
git clone https://github.com/ispc-lab/GeoNLF.git
cd GeoNLF

conda create -n geonlf python=3.9
conda activate geonlf

# PyTorch
# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA <= 11.7
# pip install torch==2.0.0 torchvision torchaudio

# Dependencies
pip install -r requirements.txt

# Local compile for tiny-cuda-nn
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install

# compile packages in utils
cd utils/chamfer3D
python setup.py install
```


### 📁 Dataset
#### Nuscenes dataset
For convenience, the Nuscenes-mini dataset (approximately 3GB) can be used for quick experiments and can be downloaded directly from the official website. Put them into `data/nuscenes`
(or use symlinks: `ln -s DATA_ROOT/nuscenes-mini ../data/nuscenes/`).

#### KITTI-360 dataset ([Download](https://www.cvlibs.net/datasets/kitti-360/download.php))
We use sequence00 (`2013_05_28_drive_0000_sync`) for experiments in our paper.  
Download KITTI-360 dataset (2D images are not needed) and put them into `data/kitti360`.  
(or use symlinks: `ln -s DATA_ROOT/KITTI-360 ../data/kitti360/`). 

The folder tree is as follows:
```bash
GeoNLF
└── ...

data
└── kitti360
    └── KITTI-360
        ├── calibration
        ├── data_3d_raw
        └── data_poses
└── nuscenes
    └── nuscenes-mini
        ├── samples
        ├── sweeps
        └── v1.0-mini
        └──...
```
##
Next, for Nuscenes-mini dataset, run nuscenes dataset preprocessing: (--start seq_id --samples)
```bash
cd data/preprocess_nus
python generate_train_rangeview.py --start 0 --samples
python nus_to_nerf.py --start 0 --samples
```

Notably, the Nuscenes-mini dataset consists of multiple scenes. The keyframes in the samples directory are sampled at a frequency of 2Hz, while the frames in the sweeps directory have a sampling frequency of 10Hz. Each scene contains approximately 40 keyframes. When selecting a sequence, please ensure that the chosen frames do not span across multiple scenes. For instance, if `start=20` and 36 frames are sampled, crossing two scenes may lead to reconstruction failure. In contrast, selecting start=0, 39, or 79 does not result in scene crossing. 

If the `--samples` flag is disabled, frames will be selected from the sweeps directory instead. The sweeps data also consists of multiple scenes; however, the poses in the sweeps directory of the Nuscenes dataset are not entirely accurate. Additionally, please copy the point cloud files from the keyframes in samples to the sweeps directory for integration. You may also use the `--high_freq` option to set a sampling frequency of 10Hz, which is only applicable when sampling from sweeps.

For KITTI-360 dataset, run KITTI-360 dataset preprocessing: (set the --start seq_id)
```bash
cd data/preprocess_kitti
python generate_train_rangeview.py --start 9999
python kitti360_to_nerf.py --start 9999
# or set the "--high_freq" flag, which indicates a sampling frequency of 10Hz.(In our experiment, sampling frequency is 2Hz.)
# python generate_train_rangeview.py --start 9999 --high_freq
# python kitti360_to_nerf.py --start 9999 --high_freq
```

##
Then obtain the scale and offset of this sequence by running cal_centerpose_bound.py, which prints the `scale`, `offset`.  

For Nuscenes:
```bash
cd data/preprocess_nus
python cal_centerpose_bound.py --start 0 --samples
# python cal_centerpose_bound.py --start 0 
# python cal_centerpose_bound.py --start 0 --high_freq
```
For KITTI-360:
```bash
cd data/preprocess_kitti
python cal_centerpose_bound.py --start 9999
# python cal_centerpose_bound.py --start 9999 --high_freq
```

##
Finally, configure the config file according to the template. After preprocessing, your folder structure should look like this:  
```bash
GeoNLF
  └── configs
      ├── kitti_9999.txt
      ├── nus_samples_0.txt
  └──data
      └── kitti360
          ├── train
          ├── kitti_transforms_9999.json
      └── nuscenes
          ├── train
          ├── nus_transforms_0.json
```

### 🚀 Run GeoNLF

 
Run the following command:
```bash
# Nuscenes
python main_lidarnerf.py --workspace mytest --config configs/nus_samples_0.txt --start 0 --rot --trans --noise_rot --noise_trans --dataloader nuscenes --geo_loss --reweight --graph_optim
# KITTI-360
python main_lidarnerf.py --workspace mytest --config configs/kitti_9999.txt --start 9999 --rot --trans --noise_rot --noise_trans --graph_optim --dataloader kitti360 --geo_loss --reweight
```
<a id="results"></a>


## Acknowledgement
We sincerely appreciate the great contribution of the following works:
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/tree/master)
- [LiDAR-NeRF](https://github.com/tangtaogo/lidar-nerf)
- [NFL](https://research.nvidia.com/labs/toronto-ai/nfl/)
- [LiDAR4D](https://github.com/ispc-lab/LiDAR4D)


## Citation
If you find our repo or paper helpful, feel free to support us with a star 🌟 or use the following citation:  
```bibtex
@inproceedings{NEURIPS2024_86ab6927,
 author = {Xue, Weiyi and Zheng, Zehan and Lu, Fan and Wei, Haiyun and Chen, Guang and jiang, changjun},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {73672--73692},
 publisher = {Curran Associates, Inc.},
 title = {GeoNLF: Geometry guided Pose-Free Neural LiDAR Fields},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/86ab6927ee4ae9bde4247793c46797c7-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```


## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
