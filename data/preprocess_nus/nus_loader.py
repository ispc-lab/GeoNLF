from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import numpy as np
import os.path as osp
import json
import os
import camtools as ct
class Nuscenes:
    def __init__(self,root):
        self.root=root
        self.table_root=osp.join(root,'v1.0-mini')
        self.scene = self.__load_table__('scene')
        self.sample = self.__load_table__('sample')
        self.sample_data = self.__load_table__('sample_data')
        self.sample_annotation = self.__load_table__('sample_annotation')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
    def __load_table__(self,table_name) -> dict:
        with open(osp.join(self.table_root,'{}.json'.format(table_name))) as f:
            table=json.load(f)
        return table

def nus_loader(root,frames_root):
    frames_list=os.listdir(frames_root)
    nus=Nuscenes(root)
    nuscenes = NuScenes(version='v1.0-mini', dataroot=root, verbose=True)
    sample_data=nus.sample_data

    new_dict = {}
    for dic in sample_data:
        filename=dic['filename']
        filename=filename.split('/')[-1]
        new_dict[filename]=dic
    
    pcd_list=[]
    T_list=[]
    for point_cloud_filename in frames_list:
        pcd_list.append(point_cloud_filename)
        # point_cloud_filename=osp.join('sweeps/LIDAR_TOP',point_cloud_filename)
        infor_idx=new_dict[point_cloud_filename]

        # lidar2car
        calibrated_sensor = nuscenes.get('calibrated_sensor', infor_idx['calibrated_sensor_token'])
        lidar2car_trans = calibrated_sensor['translation']
        lidar2car_rot = calibrated_sensor['rotation']
        lidar2car = transform_matrix(lidar2car_trans,Quaternion(lidar2car_rot),inverse=False)
        
        # intrinsics
        ego_pose = nuscenes.get('ego_pose', infor_idx['ego_pose_token'])
        car2world_trans = ego_pose['translation']
        car2world_rot = ego_pose['rotation']
        car2world = transform_matrix(car2world_trans,Quaternion(car2world_rot),inverse=False)
        
        # T
        T=car2world@lidar2car
        T_list.append(T)

    time=[]
    for frame in frames_list:
        frame_time=int(frame.split('_')[-1].split('.')[0])
        time.append(frame_time)
    sorted_pairs_frames = sorted(zip(time, frames_list))
    sorted_pairs_T=sorted(zip(time, T_list))
    _, sorted_frames_list = zip(*sorted_pairs_frames)
    _, sorted_T_list = zip(*sorted_pairs_T)
        
    return sorted_T_list,sorted_frames_list

def main():
    root= '../../../data/nuscenes/nuscenes-mini'
    frames_root='../../../data/nuscenes/nuscenes-mini/samples/LIDAR_TOP'
    T,F=nus_loader(root,frames_root)
    #np.save('T',T)
    #np.save('Frames_name',F)

if __name__ == "__main__":
    main()


