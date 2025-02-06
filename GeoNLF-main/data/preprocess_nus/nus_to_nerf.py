#nus_to_nerf
import os
from nus_loader import nus_loader
import numpy as np
import json
from pathlib import Path
import configargparse

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

def main():
    parser = get_arg_parser()
    opt = parser.parse_args()
    current_path = Path(__file__)
    project_root= current_path.parent.parent.parent
    root = project_root.parent / "data" / "nuscenes" / "nuscenes-mini"
    if opt.samples:
        frames_root= project_root.parent / "data" / "nuscenes" / "nuscenes-mini" / "samples" / "LIDAR_TOP"   # '../../../data/nuscenes/nuscenes-mini/samples/LIDAR_TOP'
    else:
        frames_root= project_root.parent / "data" / "nuscenes" / "nuscenes-mini" / "sweeps" / "LIDAR_TOP"    # '../../../data/nuscenes/nuscenes-mini/sweeps/LIDAR_TOP'

    # all poses, frames of nuscenes dataset
    T_list,frames_list=nus_loader(root,frames_root)
    # select sequence
    if opt.samples:
        T_list=T_list[opt.start:opt.start+36]
        frames_list=frames_list[opt.start:opt.start+36]
    elif opt.high_freq:
        T_list=T_list[opt.start:opt.start+36]
        frames_list=frames_list[opt.start:opt.start+36]
    else:
        T_list=T_list[opt.start:opt.start+180:5]
        frames_list=frames_list[opt.start:opt.start+180:5]
    
    current_path = Path(__file__)
    range_view_dir = current_path.parent.parent / "nuscenes" / "train"

    range_view_paths_train=[os.path.join(range_view_dir,"{}.npy".format(frame.rsplit('.bin', 1)[0])) for frame in frames_list]

    # get image dimensions, assume all images have the same dimensions.
    lidar_range_image = np.load(range_view_paths_train[0])
    lidar_h, lidar_w, _ = lidar_range_image.shape

    #
    lidar_paths_split_train=[path for path in range_view_paths_train]
    lidar2world_split_train=[T_list[i] for i in range(len(T_list))]
    lidar2world_split_train=np.array(lidar2world_split_train)


    lidar_paths_split=lidar_paths_split_train
    lidar2world_split=lidar2world_split_train

    json_dict = {
            "w_lidar": lidar_w,
            "h_lidar": lidar_h,
            "aabb_scale": 2,
            "frames": [
                {
                    "lidar_file_path": str(
                        lidar_path
                    ),
                    "lidar2world": lidar2world.tolist(),
                }
                for (
                    lidar_path,
                    lidar2world,
                ) in zip(
                    lidar_paths_split,
                    lidar2world_split,
                )
            ],
        }



    json_path = os.path.join('../nuscenes',f"nus_transforms_{opt.start}.json")

    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)
        print("Saved")
    


if __name__ == "__main__":
    main()
