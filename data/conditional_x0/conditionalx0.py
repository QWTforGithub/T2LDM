import random
from nuscenes import NuScenes
import pickle
from torch.utils.data import Dataset,DataLoader
from utils import common
import numpy as np
from pathlib import Path


class ConditionalX0(Dataset):
    def __init__(
            self,
            conditionalx0_lidar_path,
            conditionalx0_lidar_description,
            conditionalx0_lidar_semantic,

            semantic_class_num=17.0,

            training=True,
            aug=["rotation", "flip"],

            resolution=(32, 1024),
            depth_range=[1.45, 80.0],
            fov=[3, -25],

            type="nuScenes",
            print_info=True
    ):

        self.aug = aug
        self.resolution = resolution
        self.depth_range = depth_range
        self.fov = fov
        self.type = type
        self.training = training
        self.semantic_class_num = semantic_class_num - 1

        self.transform = common.get_lidar_transform(self.aug, self.training)
        self.learning_map = common.get_semantickitti_learning_map(20)

        self.conditionalx0_lidar_path = conditionalx0_lidar_path
        self.conditionalx0_lidar_description = conditionalx0_lidar_description
        self.conditionalx0_lidar_semantic = conditionalx0_lidar_semantic

        if(print_info):
            print(f" ---- ConditionalX0 Dataset with {len(self.conditionalx0_lidar_path)} ---- ")

        return

    def __len__(self):
        return len(self.conditionalx0_lidar_path)

    def __getitem__(self, idx):

        sample = None

        lidar_path = self.conditionalx0_lidar_path[idx]
        lidar_description = self.conditionalx0_lidar_description[idx]
        lidar_semantic = self.conditionalx0_lidar_semantic[idx]

        if(self.type == "nuScenes"):
            points = common.get_lidar_sweep(lidar_path, return_intensity=True, return_time=True, dim=5)

            semantic = np.expand_dims(lidar_semantic, axis=1)
            points = np.concatenate([points, semantic], axis=-1)

            if self.transform:
                points[:, :3], _ = self.transform(points[:, :3])

            range_image = common.points_as_images(
                points,
                size=self.resolution,
                fov=self.fov,
                depth_range=self.depth_range,
                return_all=True,
            ).transpose(2, 0, 1)

            sample = {
                "id": lidar_path,
                "batch": [len(points), ],
                "points": points[:, :3],  # (N,3)
                "xyz": range_image[:3],  # (3 H, W)
                "reflectance": common.reflectance_norm(range_image[[3]]),  # (1, H, W)
                "time": range_image[[4]],  # (1, H, W)
                "semantic": range_image[[5]] / self.semantic_class_num,  # (1, H, W)
                "depth": range_image[[6]],  # (1, H, W)
                "mask": range_image[[7]],  # (1, H, W)
                "text": lidar_description,  # String
                "semantic_org": self.conditionalx0_lidar_semantic[idx],
            }

        elif(self.type == "kitti_360"):

            points = common.get_lidar_sweep(lidar_path, return_intensity=True)

            if self.transform:
                points[:, :3], _ = self.transform(points[:, :3])

            range_image = common.points_as_images(
                points,
                size=self.resolution,
                fov=self.fov,
                depth_range=self.depth_range,
                return_all=True
            )

            range_image = range_image.transpose(2, 0, 1)

            sample = {
                "id": lidar_path,
                "batch": [len(points), ],
                "points": points[:, :3],  # (N,3)
                "xyz": range_image[:3],  # (3 H, W)
                "reflectance": range_image[[3]],  # (1, H, W)
                "depth": range_image[[4]],  # (1, H, W)
                "mask": range_image[[5]],  # (1, H, W)
            }

        elif(self.type == "kitti_semantic"):

            points = common.get_lidar_sweep(lidar_path, return_intensity=True)
            if self.transform:
                points[:, :3], _ = self.transform(points[:, :3])

            with open(lidar_semantic, "rb") as a:
                semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
                semantic = np.vectorize(self.learning_map.__getitem__)(
                    semantic & 0xFFFF
                ).astype(np.int32)

            semantic = np.expand_dims(semantic, axis=1)
            points = np.concatenate([points, semantic], axis=-1)

            range_image = common.points_as_images(
                points,
                size=self.resolution,
                fov=self.fov,
                depth_range=self.depth_range,
                return_all=True,
            ).transpose(2, 0, 1)

            sample = {
                "id": lidar_path,
                "points": points[:, :3],  # (N,3)
                "batch": [len(points), ],
                "xyz": range_image[:3],  # (3  H, W)
                "reflectance": range_image[[3]],  # (1, H, W)
                "semantic": range_image[[4]] / self.semantic_class_num,  # (1, H, W)
                "depth": range_image[[5]],  # (1, H, W)
                "mask": range_image[[6]],  # (1, H, W)
                "text": lidar_description,  #
            }

        return sample


if __name__ == '__main__':

    from utils import lidar

    resolution = (32,1024) # (32,1024) (64, 1024)
    depth_range = (0.01,50.0) # (0.0001,50.0) (1.45, 80.0)
    fov = (3,-25)

    li = lidar.LiDARUtility(
        resolution=resolution,
        depth_range=depth_range,
        fov=fov,
        project_dir="/data/qwt/temp"
    )
    li = li.cuda()

    data_root = "/data/qwt/dataset/nuscenes/raw"
    version = "v1.0-trainval"
    pkl = "nuscenes_infos_10sweeps_description.pkl"
    dataset = ConditionalX0(
        data_root=data_root,
        version=version,
        pkl=pkl
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        collate_fn=common.collate_fn
    )

    for batch in dataloader:
        common.load_data_to_gpu(batch)
        depth = batch["depth"]
        depth = li.convert_depth(depth)
        depth = li.normalize(depth)
        li.sample_to_lidar(metric=depth)
        break
        # print(batch)
    pass