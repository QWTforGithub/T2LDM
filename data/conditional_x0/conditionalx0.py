import random
from nuscenes import NuScenes
import pickle
from torch.utils.data import Dataset, DataLoader
from utils import common
import numpy as np
from pathlib import Path
from data.kitti_semantic.descriptor import read_pkl

KITTI360_SCENES = [0, 2, 3, 4, 5, 6, 7, 9, 10]
SEMANTICKITTI_SCENES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class ConditionalX0(Dataset):
    def __init__(
            self,
            conditionalx0_lidar_path=None,
            conditionalx0_lidar_description=None,
            conditionalx0_lidar_semantic=None,

            data_root='/root/dataset/rsd_data/nuscenes',
            pkl='nuscenes_infos_10sweeps_description.pkl',
            version='v1.0-trainval',
            text_keys="text_aim",
            text_path=None,

            use_seg=False,
            semantic_class_num=17.0,

            training=True,
            aug=["rotation", "flip"],

            resolution=(32, 1024),
            depth_range=[1.45, 80.0],
            fov=[3, -25],

            type="nuScenes",
            random_num=256, # 256
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

        if (self.conditionalx0_lidar_path is None):
            self.lidar_path = []
            self.lidar_description = []
            self.lidar_semantic = []

            lists = [
                8145, 9136, 10245, 11234, 13478,
                15423, 16789, 17789, 18788, 19745,
                0, 174, 356, 1024, 4132,
                5124, 6154, 6657, 7145, 7542,
            ]
            lists = [lists] * 10
            lists = [x for sub in lists for x in sub]

            self.conditionalx0_lidar_path = []
            self.conditionalx0_lidar_description = []
            self.conditionalx0_lidar_semantic = []

            if (type == "nuScenes"):

                pkl_path = f"{data_root}/{version}/{pkl}"
                with open(pkl_path, 'rb') as f:
                    infos = pickle.load(f)

                for info in infos:
                    if (not info.keys().__contains__("lidar_path")):
                        continue

                    lidar_path = info["lidar_path"]
                    self.lidar_path.append(f"{data_root}/{version}/{lidar_path}")

                    # description = common.cat_descriptor(info=info, keys=text_keys)
                    description = info[text_keys]
                    self.lidar_description.append(description)

                    if (info.keys().__contains__("semantic")):
                        semantic = info["semantic"]
                    else:
                        semantic = None

                    self.lidar_semantic.append(semantic)

            elif (type == "KITTI360"):
                for scene in KITTI360_SCENES:
                    wildcard = f"*_{scene:04d}_sync/velodyne_points/data/*.bin"
                    self.lidar_path += sorted(Path(data_root).glob(wildcard))

                for _ in self.lidar_path:
                    self.lidar_description.append(None)
                    self.lidar_semantic.append(None)

            elif (type == "SemanticKITTI"):
                if (text_path is None):
                    for scene in SEMANTICKITTI_SCENES:
                        wildcard = f"{scene:02d}/velodyne/*.bin"
                        self.lidar_path += sorted(Path(data_root).glob(wildcard))

                        wildcard = f"{scene:02d}/labels/*.label"
                        self.lidar_semantic += sorted(Path(data_root).glob(wildcard))

                    for _ in self.lidar_path:
                        self.lidar_description.append(None)

                else:
                    infos = read_pkl(text_path)
                    for info in infos:
                        self.lidar_path.append(Path(info["bin_path"]))
                        self.lidar_description.append(info["text"])
                        self.lidar_semantic.append(Path(info["semantic_path"]))


            if(random_num > 0 and use_seg):
                lists = random.sample(range(len(self.lidar_path)), random_num)
                print(f"list: {lists}")

            for l in lists:
                self.conditionalx0_lidar_path.append(self.lidar_path[l])
                self.conditionalx0_lidar_description.append(self.lidar_description[l])
                self.conditionalx0_lidar_semantic.append(self.lidar_semantic[l])

        if (print_info):
            print(f" ---- ConditionalX0 Dataset with {len(self.conditionalx0_lidar_path)} ---- ")

        return

    def __len__(self):
        return len(self.conditionalx0_lidar_path)

    def __getitem__(self, idx):

        sample = None

        lidar_path = self.conditionalx0_lidar_path[idx]
        lidar_description = self.conditionalx0_lidar_description[idx]
        lidar_semantic = self.conditionalx0_lidar_semantic[idx]

        if (self.type == "nuScenes"):
            points = common.get_lidar_sweep(lidar_path, return_intensity=True, return_time=True, dim=5)

            if (lidar_semantic is not None):
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

            if (lidar_semantic is not None):
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
                    "semantic_org": range_image[[5]],
                }
            else:
                sample = {
                    "id": lidar_path,
                    "batch": [len(points), ],
                    "points": points[:, :3],  # (N,3)
                    "xyz": range_image[:3],  # (3 H, W)
                    "reflectance": common.reflectance_norm(range_image[[3]]),  # (1, H, W)
                    "time": range_image[[4]],  # (1, H, W)
                    "depth": range_image[[5]],  # (1, H, W)
                    "mask": range_image[[6]],  # (1, H, W)
                    "text": lidar_description,  # String
                }

        elif (self.type == "kitti_360"):

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

        elif (self.type == "kitti_semantic"):

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
                "text": lidar_description,
                "semantic_org": range_image[[4]]  # self.lidar_semantic[idx],
            }

        return sample


if __name__ == '__main__':

    from utils import lidar

    resolution = (32, 1024)  # (32,1024) (64, 1024)
    depth_range = (0.01, 50.0)  # (0.0001,50.0) (1.45, 80.0)
    fov = (3, -25)

    li = lidar.LiDARUtility(
        resolution=resolution,
        depth_range=depth_range,
        fov=fov,
        project_dir="/data/qwt/temp"
    )
    li = li.cuda()

    data_root = "/root/dataset/rsd_data/nuscenes"
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
        # li.sample_to_lidar(generation=depth)
        break
        # print(batch)

    pass