import numpy as np
import configargparse
np.set_printoptions(suppress=True)
import os
import json
import tqdm

def pano_to_lidar_with_intensities(pano: np.ndarray, intensities, lidar_K):
    """
    Args:
        pano: (H, W), float32.
        intensities: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points_with_intensities: (N, 4), float32, in lidar frame.
    """
    fov_up, fov = lidar_K

    H, W = pano.shape
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    beta = -(i - W / 2) / W * 2 * np.pi
    alpha = (fov_up - j / H * fov) / 180 * np.pi
    dirs = np.stack(
        [
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ],
        -1,
    )
    local_points = dirs * pano.reshape(H, W, 1)

    # local_points: (H, W, 3)
    # intensities : (H, W)
    # local_points_with_intensities: (H, W, 4)
    local_points_with_intensities = np.concatenate(
        [local_points, intensities.reshape(H, W, 1)], axis=2
    )

    # Filter empty points.
    idx = np.where(pano != 0.0)
    local_points_with_intensities = local_points_with_intensities[idx]

    return local_points_with_intensities

def pano_to_lidar(pano, lidar_K):
    """
    Args:
        pano: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points: (N, 3), float32, in lidar frame.
    """
    local_points_with_intensities = pano_to_lidar_with_intensities(
        pano=pano,
        intensities=np.zeros_like(pano),
        lidar_K=lidar_K,
    )
    return local_points_with_intensities[:, :3]


def cal_centerpose_bound_scale(
    lidar_rangeview_paths, lidar2worlds, intrinsics, bound=1.0
):
    near = 200
    far = 0
    points_world_list = []
    for i, lidar_rangeview_path in enumerate(lidar_rangeview_paths):
        pano = np.load(lidar_rangeview_path)
        point_cloud = pano_to_lidar(pano=pano[:, :, 2], lidar_K=intrinsics)
        point_cloud = np.concatenate(
            [point_cloud, np.ones(point_cloud.shape[0]).reshape(-1, 1)], -1
        )
        dis = np.sqrt(
            point_cloud[:, 0] ** 2 + point_cloud[:, 1] ** 2 + point_cloud[:, 2] ** 2
        )
        near = min(min(dis), near)
        far = max(far, max(dis))
        points_world = (point_cloud @ lidar2worlds[i].T)[:, :3]
        points_world_list.append(points_world)
    print("near, far:", near, far)
    pc_all_w = np.concatenate(points_world_list)[:, :3]

    centerpose = [
        (np.max(pc_all_w[:, 0]) + np.min(pc_all_w[:, 0])) / 2.0,
        (np.max(pc_all_w[:, 1]) + np.min(pc_all_w[:, 1])) / 2.0,
        (np.max(pc_all_w[:, 2]) + np.min(pc_all_w[:, 2])) / 2.0,
    ]
    print("centerpose: ", centerpose)
    pc_all_w_centered = pc_all_w - centerpose

    norms = np.linalg.norm(pc_all_w_centered , ord=2, axis=1)
    # Step 2: Square the norms to get the sum of squares for each row
    bound_ori = [
        np.max(pc_all_w_centered[:, 0]),
        np.max(pc_all_w_centered[:, 1]),
        np.max(pc_all_w_centered[:, 2]),
    ]
    scale = bound / np.max(bound_ori)
    print("scale: ", scale)

def get_arg_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--samples",action="store_true")
    parser.add_argument("--high_freq",action="store_true")
    parser.add_argument(
        "--start",
        type=int, 
        default=480,
        help="choose start",
    )
    return parser

def get_path_pose_from_json(root_path, sequence_id):
    with open(
        os.path.join(root_path, f"nus_transforms_{sequence_id}.json"), "r"
        #f"../../data/nuscenes/{opt.start}_transforms_train_swps.json"
    ) as f:
        transform = json.load(f)
    frames = transform["frames"]
    poses_lidar = []
    paths_lidar = []
    for f in tqdm.tqdm(frames, desc=f"Loading {type} data"):
        pose_lidar = np.array(f["lidar2world"], dtype=np.float32)  # [4, 4]
        f_lidar_path=os.path.join('../',f["lidar_file_path"])
        #print(f_lidar_path)
        #f_lidar_path = os.path.join(root_path, f["lidar_file_path"])
        poses_lidar.append(pose_lidar)
        paths_lidar.append(f_lidar_path)
    return paths_lidar, poses_lidar


def main():
    parser = get_arg_parser()
    opt = parser.parse_args()
    root_path = "../nuscenes"
    sequence_id=opt.start
    lidar_rangeview_paths, lidar2worlds = get_path_pose_from_json(root_path, sequence_id)
    intrinsics = (10.0, 40.0)  # fov_up, fov
    cal_centerpose_bound_scale(lidar_rangeview_paths, lidar2worlds, intrinsics)


if __name__ == "__main__":
    main()
