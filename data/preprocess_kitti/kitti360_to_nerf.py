from pathlib import Path

from kitti360_loader import KITTI360Loader
import camtools as ct
import numpy as np
import json
import configargparse
def get_arg_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=int, 
        default=1908,
        help="choose start",
    )
    parser.add_argument(
    "--high_freq",
    action="store_true",
    )

    return parser.parse_args()

def main():
    opt=get_arg_parser()
    current_path=Path(__file__)
    project_root = Path(__file__).parent.parent.parent
    kitti_360_root = project_root.parent / "data" / "kitti360" / "KITTI-360"



    kitti_360_parent_dir = kitti_360_root.parent
    out_dir=project_root / "data" / "kitti360"

    # Specify frames and splits.
    sequence_name = "2013_05_28_drive_0000"
    frame_ids = list(range(opt.start, opt.start + 180,5))
    if opt.high_freq:
        frame_ids=list(range(opt.start,opt.start+50))

    k3 = KITTI360Loader(kitti_360_root)
    # Get lidar paths (range view not raw data).
    range_view_dir = current_path.parent.parent / "kitti360" / "train"
    range_view_paths = [
        range_view_dir / "{:010d}.npy".format(int(frame_id)) for frame_id in frame_ids
    ]
    # Get lidar2world.
    lidar2world = k3.load_lidars(sequence_name, frame_ids)
    # Get range image dimensions, assume all images have the same dimensions.
    lidar_range_image = np.load(range_view_paths[0])
    lidar_h, lidar_w, _ = lidar_range_image.shape

    lidar_paths_split = [range_view_paths[i] for i in range(len(frame_ids))]
    lidar2world_split = [lidar2world[i] for i in range(len(frame_ids))]

    json_dict = {
        "w_lidar": lidar_w,
        "h_lidar": lidar_h,
        "aabb_scale": 2,
        "frames": [
            {
                "lidar_file_path": str(
                    lidar_path #.relative_to(kitti_360_parent_dir)
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
    json_path = out_dir / f"kitti_transforms_{opt.start}.json"

    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)
        print(f"Saved {json_path}.")


if __name__ == "__main__":
    main()
