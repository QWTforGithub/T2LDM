import glob

import os
from torch.utils.data import Dataset,DataLoader
from utils import common
import numpy as np

# class ConditionalX0(Dataset):
#     def __init__(
#             self,
#             data_root='/data/qwt/dataset/conditionalx0',
#             training=True,
#
#             resolution=(32,1024),
#             depth_range=[1.45,80.0],
#             fov=[3,-25],
#     ):
#
#         self.data_root = data_root
#         self.training = training
#
#         self.resolution = resolution
#         self.depth_range = depth_range
#         self.fov = fov
#
#         self.lidar_paths = sorted(glob.glob(os.path.join(data_root,'*.pcd.bin')))
#         print(f" ----Conditional Dataset with {len(self.lidar_paths)} ---- ")
#         return
#
#     def __len__(self):
#         return len(self.lidar_paths)
#
#     def __getitem__(self, idx):
#         lidar_path = self.lidar_paths[idx]
#         points = common.get_lidar_sweep(lidar_path, return_intensity=True, return_time=True, dim=5)
#
#         range_image = common.points_as_images(
#             points,
#             size = self.resolution,
#             fov = self.fov,
#             depth_range = self.depth_range,
#             return_all=True,
#         ).transpose(2, 0, 1)
#
#         sample = {
#             "xyz": range_image[:3],                                  # (3 H, W)
#             "reflectance": common.reflectance_norm(range_image[[3]]),# (1, H, W)
#             "time": range_image[[4]],                                # (1, H, W)
#             "depth": range_image[[5]],                               # (1, H, W)
#             "mask": range_image[[6]],                                # (1, H, W)
#         }
#
#         return sample
#
#
# if __name__ == '__main__':
#
#     dataset = ConditionalX0()
#
#     dataloader = DataLoader(
#         dataset,
#         batch_size=2,
#         shuffle=False,
#         num_workers=4,
#         drop_last=True,
#         pin_memory=True,
#         collate_fn=common.collate_fn
#     )
#
#     for batch in dataloader:
#         common.load_data_to_gpu(batch)
#         print(batch)
#     pass

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
            data_root='/data/qwt/dataset/nuscenes/raw',
            pkl='',
            version='v1.0-trainval',
            training=True,
            aug=["rotation", "flip"],

            resolution=(32,1024),
            depth_range=[1.45,80.0],
            fov=[3,-25],

            only_class=-1,
            text_keys="text_l5 text_l7 text_l8",
            semantic_class_num=20.0,
            use_semantic=True,

            type="nuScenes",

            print_info=True
    ):

        self.data_root = data_root
        self.pkl = pkl
        self.training = training
        self.aug = aug

        self.resolution = resolution
        self.depth_range = depth_range
        self.fov = fov

        self.semantic_class_num = semantic_class_num - 1

        self.type = type

        self.transform = common.get_lidar_transform(self.aug, self.training)

        self.lidar_paths = []
        self.lidar_description = []
        self.lidar_description_class = []
        self.lidar_description_l0 = []
        self.lidar_description_l1 = []
        self.lidar_description_l2 = []
        self.lidar_description_l3 = []
        self.lidar_description_l4 = []

        if(type == "nuScenes"):
            self.semantic = []

            if(".pkl" in pkl):
                pkl_path = f"{data_root}/{version}/{pkl}"
                with open(pkl_path, 'rb') as f:
                    infos = pickle.load(f)

                lists = [
                    8145, 9136, 10245, 11234, 13478,
                    15423, 16789, 17789, 18788, 19745,
                    0, 174, 356, 1024, 4132,
                    5124, 6154, 6657, 7145, 7542,
                ]

                new_paths = []
                for l in lists:
                    new_paths.append(infos[l])

                infos = new_paths

                # infos_1 =  infos[174:176]
                # infos_2 =  infos[:2]
                # infos_3 = infos[16187:16189]
                # infos_4 = infos[16198:16200]
                #
                # infos_5 =  infos[90:92]
                # infos_6 =  infos[2:5]
                # infos_7 = infos[92:95]
                # infos_8 = infos[17:20]
                #
                # infos = infos_1 + infos_2 + infos_3 +infos_4 + infos_5 + infos_6 + infos_7 + infos_8

                # infos1 = common.get_sample_by_text(infos=infos, text_keys="Less than", num=35)
                # infos2 = common.get_sample_by_text(infos=infos, text_keys="More than", num=35) # pedestrian/barrier/truck

                # infos1 = common.get_sample_by_text(infos=infos, text_keys="pedestrian", num=25, name="text_aim")
                # infos2 = common.get_sample_by_text(infos=infos, text_keys="barrier", num=25, name="text_aim")
                # infos3 = common.get_sample_by_text(infos=infos, text_keys="truck", num=25, name="text_aim")
                #
                # infos = infos1 + infos2 + infos3

                # infos = random.sample(infos, 80)

                for info in infos:
                    lidar_path = info["lidar_path"]
                    self.lidar_paths.append(f"{data_root}/{version}/{lidar_path}")
                    # print(lidar_path)

                    description = common.cat_descriptor(info=info, keys=text_keys)
                    self.lidar_description.append(description)

                    semantic = info["semantic"]
                    if(only_class >= 0):
                        semantic[...] = (semantic == only_class)
                    self.semantic.append(semantic)

                if(print_info):
                    print(f" ---- ConditionalX0 {type} Dataset with {len(infos)} ---- ")

            else:
                nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

                for scene in nusc.scene[0:850]:
                    sample = nusc.get('sample', scene['first_sample_token'])
                    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    self.lidar_paths.append(lidar['filename'])
                    self.lidar_description.append(scene['description'])
                    self.lidar_description_class.append(scene['description_class'])
                    for i in range(scene['nbr_samples'] - 1):
                        sample = nusc.get('sample', sample['next'])
                        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                        self.lidar_paths.append(lidar['filename'])
                        self.lidar_description.append(scene['description'])  # get all samples
                        self.lidar_description_class.append(scene['description_class'])  # get all samples
        elif(type == "kitti_360"):
            SCENES = [0, 2, 3, 4, 5, 6, 7, 9, 10]

            file_paths = []
            for scene in SCENES:
                wildcard = f"*_{scene:04d}_sync/velodyne_points/data/*.bin"
                file_paths += sorted(Path(data_root).glob(wildcard))

            lists = [
                0,          174,        356,        1024,       4132,
                5124,       6154,       6657,       7145,       7542,
                8145,       9136,       10245,      11234,      13478,
                15423,      16789,      17789,      18788,      19745,
            ]

            new_paths = []
            for l in lists:
                new_paths.append(file_paths[l])
            self.file_paths = file_paths

            self.items = list(zip(range(len(file_paths)), file_paths))

            if (print_info):
                print(f" ---- ConditionalX0 {type} Dataset with {len(self.items)} ---- ")

        elif(type == "kitti_semantic"):
            SCENES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            self.learning_map = common.get_semantickitti_learning_map(self.semantic_class_num)

            lists = [
                0,          174,        356,        1024,       4132,
                5124,       6154,       6657,       7145,       7542,
                8145,       9136,       10245,      11234,      13478,
                15423,      16789,      17789,      18788,      19745,
            ]

            bin_paths = []
            semantic_paths = []
            for scene in SCENES:
                wildcard = f"{scene:02d}/velodyne/*.bin"
                bin_paths += sorted(Path(data_root).glob(wildcard))

                wildcard = f"{scene:02d}/labels/*.label"
                semantic_paths += sorted(Path(data_root).glob(wildcard))

            new_paths = []
            new_semantic_paths = []
            for l in lists:
                new_paths.append(bin_paths[l])
                new_semantic_paths.append(semantic_paths[l])

            self.bin_paths = new_paths
            self.semantic_paths = new_semantic_paths

            if (print_info):
                print(f" ---- ConditionalX0 {type} Dataset with {len(self.bin_paths)} ---- ")

        return

    def __len__(self):
        if(self.type == "nuScenes"):
            return len(self.lidar_paths)
        elif(self.type == "kitti_360"):
            return len(self.items)
        elif(self.type == "kitti_semantic"):
            return len(self.bin_paths)
        else:
            return 0

    def __getitem__(self, idx):

        sample = None

        if(self.type == "nuScenes"):
            lidar_path = self.lidar_paths[idx]
            lidar_description = self.lidar_description[idx]
            points = common.get_lidar_sweep(lidar_path, return_intensity=True, return_time=True, dim=5)

            # common.save_points(points[:, :3], name=f"/data/qwt/models/temp/{idx}.ply")

            semantic = np.expand_dims(self.semantic[idx], axis=1)
            points = np.concatenate([points, semantic], axis=-1)

            if self.transform:
                points[:,:3],_ = self.transform(points[:,:3])

            range_image = common.points_as_images(
                points,
                size = self.resolution,
                fov = self.fov,
                depth_range = self.depth_range,
                return_all=True,
            ).transpose(2, 0, 1)

            sample = {
                "id": lidar_path,
                "batch": [len(points), ],
                "points": points[:, :3],                                    # (N,3)
                "xyz": range_image[:3],                                     # (3 H, W)
                "reflectance": common.reflectance_norm(range_image[[3]]),   # (1, H, W)
                "time": range_image[[4]],                                   # (1, H, W)
                "semantic": range_image[[5]] / self.semantic_class_num,     # (1, H, W)
                "depth": range_image[[6]],                                  # (1, H, W)
                "mask": range_image[[7]],                                   # (1, H, W)
                "text": lidar_description,                                  # String
                "semantic_org": self.semantic[idx],
            }

        elif(self.type == "kitti_360"):
            sample_id, file_path = self.items[idx]
            points = common.get_lidar_sweep(file_path, return_intensity=True)

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
                "id": file_path,
                "batch": [len(points), ],
                "points": points[:, :3],                    # (N,3)
                "xyz": range_image[:3],                     # (3 H, W)
                "reflectance": range_image[[3]],            # (1, H, W)
                "depth": range_image[[4]],                  # (1, H, W)
                "mask": range_image[[5]],                   # (1, H, W)
            }

        elif(self.type == "kitti_semantic"):
            bin_path = self.bin_paths[idx]
            semantic_path = self.semantic_paths[idx]

            points = common.get_lidar_sweep(bin_path, return_intensity=True)
            if self.transform:
                points[:, :3], _ = self.transform(points[:, :3])

            with open(semantic_path, "rb") as a:
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
                "id": bin_path,
                "points": points[:, :3],                                    # (N,3)
                "batch": [len(points), ],
                "xyz": range_image[:3],                                     # (3  H, W)
                "reflectance": range_image[[3]],                            # (1, H, W)
                "semantic": range_image[[4]] / self.semantic_class_num,     # (1, H, W)
                "depth": range_image[[5]],                                  # (1, H, W)
                "mask": range_image[[6]],                                   # (1, H, W)
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