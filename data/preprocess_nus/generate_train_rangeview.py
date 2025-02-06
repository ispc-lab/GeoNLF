import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import shutil
from nus_loader import nus_loader
import argparse
import configargparse

def lidar_to_pano_with_intensities(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    max_depth=80,
    min_depth=1.8 #for nus
):
    """
    Convert lidar frame to pano frame with intensities.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
        intensities: (H, W), float32.
    """
    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    #Intensity of nuscenes : 0-255
    local_point_intensities=local_point_intensities/255
    
    fov_up, fov = lidar_K
    fov_down = fov - fov_up
    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)
    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    for local_points, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth and min depth
        if dist >= max_depth or dist<=min_depth:
            continue

        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity

    return pano, intensities

def LiDAR_2_Pano(
    local_points_with_intensities, lidar_H, lidar_W, intrinsics, max_depth=80.0
):
    pano, intensities = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    return range_view


def generate_train_data(
    H,
    W,
    intrinsics,
    lidar_paths,
    out_dir,
    points_dim,
):
    """
    Args:
        H: Heights of the range view.
        W: Width of the range view.
        intrinsics: (fov_up, fov) of the range view.
        out_dir: Output directory.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lidar_path in tqdm(lidar_paths):
        point_cloud = np.fromfile(lidar_path, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, points_dim))
        pano = LiDAR_2_Pano(point_cloud, H, W, intrinsics)
        frame_name = lidar_path.split("/")[-1] #n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin
        suffix = frame_name.split(".")[-1]
        frame_name = frame_name.replace(suffix, "npy")
        np.save(out_dir / frame_name, pano)

def create_nus_rangeview(frames_path_select):
    out_dir='../nuscenes/train'
    H = 32
    W = 1080
    intrinsics = (10.0, 40.0)  # fov_up, fov
    
    generate_train_data(
    H=H,
    W=W,
    intrinsics=intrinsics,
    lidar_paths=frames_path_select,
    out_dir=out_dir,
    points_dim=5,
    )


def get_arg_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--samples",action="store_true")
    parser.add_argument("--high_freq",action="store_true")

    parser.add_argument(
        "--start",
        type=int, 
        default=40,
        help="choose start",
    )

    return parser

def main():
    parser = get_arg_parser()
    opt = parser.parse_args()
    root= '../../../data/nuscenes/nuscenes-mini'
    if opt.samples:
        frames_root='../../../data/nuscenes/nuscenes-mini/samples/LIDAR_TOP'
    else:
        frames_root='../../../data/nuscenes/nuscenes-mini/sweeps/LIDAR_TOP'
    T,F=nus_loader(root,frames_root)

    if opt.samples:
        F=F[opt.start:opt.start+36]
    elif opt.high_freq:
        F=F[opt.start:opt.start+36]
    else:
        #sample from sweeps-folder
        F=F[opt.start:opt.start+180:5]

    frames_path_select=[os.path.join(frames_root,frame_name) for frame_name in F]
    create_nus_rangeview(frames_path_select)



if __name__ == "__main__":
    main()

