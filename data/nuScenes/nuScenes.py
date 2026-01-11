from nuscenes import NuScenes
import pickle
from torch.utils.data import Dataset,DataLoader
from utils import common
import numpy as np
import torch

class NuScenesDataset(Dataset):
    def __init__(
            self,
            data_root='/data/qwt/dataset/nuscenes/raw',
            pkl='',
            version='v1.0-trainval',
            training=True,
            aug=["rotation", "flip"],

            resolution=(32,1024),
            depth_range=[0.01,50.0],
            fov=[3,-25],

            only_class=-1,
            text_keys="text_l5 text_l7 text_l8",
            semantic_class_num=17.0,

            print_info = True
    ):

        self.data_root = data_root
        self.pkl = pkl
        self.training = training
        self.aug = aug

        self.resolution = resolution
        self.depth_range = depth_range
        self.fov = fov

        self.semantic_class_num = semantic_class_num - 1

        self.transform = common.get_lidar_transform(self.aug, self.training)

        self.lidar_paths = []
        self.lidar_description = []

        self.semantic = []

        if(".pkl" in pkl):
            pkl_path = f"{data_root}/{version}/{pkl}"
            with open(pkl_path, 'rb') as f:
                infos = pickle.load(f)

            for info in infos:
                lidar_path = info["lidar_path"]
                self.lidar_paths.append(f"{data_root}/{version}/{lidar_path}")

                description = common.cat_descriptor(info=info, keys=text_keys)
                self.lidar_description.append(description)

                semantic = info["semantic"]
                if(only_class >= 0):
                    semantic[...] = (semantic == only_class)
                self.semantic.append(semantic)

            if(print_info):
                print(f" ---- nuScenes Dataset with {len(infos)} ---- ")

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

        return

    def __len__(self):
        return len(self.lidar_paths)

    def __getitem__(self, idx):
        lidar_path = self.lidar_paths[idx]
        lidar_description = self.lidar_description[idx]

        points = common.get_lidar_sweep(lidar_path, return_intensity=True, return_time=True, dim=5)

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
            "batch": [len(points),],
            "points": points[:, :3],                                        # (N,3)
            "xyz": range_image[:3],                                         # (3 H, W)
            "reflectance": common.reflectance_norm(range_image[[3]]),       # (1, H, W)
            "time": range_image[[4]],                                       # (1, H, W)
            "semantic": range_image[[5]] / self.semantic_class_num,         # (1, H, W)
            "depth": range_image[[6]],                                      # (1, H, W)
            "mask": range_image[[7]],                                       # (1, H, W)
            "text": lidar_description,                                      # String
            "semantic_org": self.semantic[idx],
        }

        return sample


if __name__ == '__main__':
    data_root = "/ihoment/youjie10/qwt/dataset/nuScenes"
    version = "v1.0-trainval"
    pkl = "nuscenes_infos_10sweeps_description.pkl"
    dataset = NuScenesDataset(
        data_root=data_root,
        version=version,
        pkl=pkl,
        text_keys=("text_aim")
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        collate_fn=common.collate_fn
    )

    for batch in dataloader:
        common.load_data_to_gpu(batch)
        print(batch)

    # xx = '/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin'
    # points = common.get_lidar_sweep(xx, return_intensity=True, return_time=True, dim=5)
    #
    # range_image = common.points_as_images(
    #     points,
    #     size=(32, 1024),
    #     fov=(3, -25),
    #     depth_range=(0.01,50.0),
    #     return_all=True,
    # ).transpose(2, 0, 1)
    pass