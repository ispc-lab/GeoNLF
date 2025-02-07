<div align="center">


<h3>GeoNLF: Geometry guided Pose-Free Neural LiDAR Fields</h3>  

**NeurIPS 2024**


**[Paper (arXiv)](https://arxiv.org/abs/2407.05597) | [Paper (NeurIPS)](https://papers.nips.cc/paper_files/paper/2024/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html) | [Project Page](xx) | [Video](xx) | [Poster](xx) **  

This repository is the official PyTorch implementation for GeoNLF.

</div>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#changelog">Changelog</a>
    </li>
    <li>
      <a href="#demo">Demo</a>
    </li>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#getting-started">Getting started</a>
    </li>
    <li>
      <a href="#results">Results</a>
    </li>
    <li>
      <a href="#simulation">Simulation</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>


## Changelog    
2024-9-26:🎉 Our paper is accepted by NeurIPS 2024.  


## Demo

## Introduction


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

#### KITTI-360 dataset ([Download](https://www.cvlibs.net/datasets/kitti-360/download.php))
We use sequence00 (`2013_05_28_drive_0000_sync`) for experiments in our paper.  

<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/c9f5d5c5-ac48-4d54-8109-9a8b745bbca0" width=65%>  

Download KITTI-360 dataset (2D images are not needed) and put them into `data/kitti360`.  
(or use symlinks: `ln -s DATA_ROOT/KITTI-360 ./data/kitti360/`).  
The folder tree is as follows:  

```bash
data
└── kitti360
    └── KITTI-360
        ├── calibration
        ├── data_3d_raw
        └── data_poses
```

Next, run KITTI-360 dataset preprocessing: (set `DATASET` and `SEQ_ID`)  

```bash
bash preprocess_data.sh
```

After preprocessing, your folder structure should look like this:  

```bash
configs
├── kitti360_{sequence_id}.txt
data
└── kitti360
    ├── KITTI-360
    │   ├── calibration
    │   ├── data_3d_raw
    │   └── data_poses
    ├── train
    ├── transforms_{sequence_id}test.json
    ├── transforms_{sequence_id}train.json
    └── transforms_{sequence_id}val.json
```

### 🚀 Run GeoNLF

Set corresponding sequence config path in `--config` and you can modify logging file path in `--workspace`. Remember to set available GPU ID in `CUDA_VISIBLE_DEVICES`.   
Run the following command:
```bash
# KITTI-360
bash run_kitti_lidar4d.sh
```


<a id="results"></a>


## Acknowledgement
We sincerely appreciate the great contribution of the following works:
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/tree/master)
- [LiDAR-NeRF](https://github.com/tangtaogo/lidar-nerf)
- [NFL](https://research.nvidia.com/labs/toronto-ai/nfl/)
- [K-Planes](https://github.com/sarafridov/K-Planes)


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
