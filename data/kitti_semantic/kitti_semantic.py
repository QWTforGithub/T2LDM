from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from utils import common
import numpy as np

SCENES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class KITTISemanticDataset(Dataset):

    def __init__(
            self,
            data_root="/data/qwt/dataset/semanticKITTI/dataset/sequences",
            training=True,
            aug=["rotation", "flip"],

            resolution=(64, 1024),
            depth_range=[1.45, 80.0],
            fov=[3, -25],

            semantic_class_num=20.0,

            print_info=True
    ):
        super().__init__()
        self.data_root = data_root
        self.training = training
        self.aug = aug

        self.resolution = resolution
        self.depth_range = depth_range
        self.fov = fov

        self.semantic_class_num = semantic_class_num - 1

        self.transform = common.get_lidar_transform(self.aug, self.training)

        self.learning_map = common.get_semantickitti_learning_map(semantic_class_num)

        bin_paths = []
        semantic_paths = []
        for scene in SCENES:
            wildcard = f"{scene:02d}/velodyne/*.bin"
            bin_paths += sorted(Path(data_root).glob(wildcard))

            wildcard = f"{scene:02d}/labels/*.label"
            semantic_paths += sorted(Path(data_root).glob(wildcard))

        self.bin_paths = bin_paths
        self.semantic_paths = semantic_paths

        if(print_info):
            print(f" ---- SemanticKITTI Dataset with {len(bin_paths)} ---- ")

    def __len__(self):
        return len(self.bin_paths)

    def __getitem__(self, idx):
        bin_path = self.bin_paths[idx]
        semantic_path = self.semantic_paths[idx]

        points = common.get_lidar_sweep(bin_path, return_intensity=True)
        if self.transform:
            points[:,:3],_ = self.transform(points[:,:3])

        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(self.learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        semantic = np.expand_dims(semantic, axis=1)
        points = np.concatenate([points, semantic], axis=-1)

        range_image = common.points_as_images(
            points,
            size = self.resolution,
            fov = self.fov,
            depth_range = self.depth_range,
            return_all=True,
        ).transpose(2, 0, 1)

        sample = {
            "id": bin_path,
            "points": points[:, :3],  # (N,3)
            "batch": [len(points), ],
            "xyz": range_image[:3],                                      # (3  H, W)
            "reflectance": range_image[[3]],                             # (1, H, W)
            "semantic": range_image[[4]] / self.semantic_class_num,      # (1, H, W)
            "depth": range_image[[5]],                                   # (1, H, W)
            "mask": range_image[[6]],                                    # (1, H, W)
        }

        return sample

if __name__ == '__main__':

    dataset = KITTISemanticDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        collate_fn=common.collate_fn
    )

    for batch in dataloader:
        common.load_data_to_gpu(batch)
        xyz = batch["xyz"]
        reflectance = batch["reflectance"]
        semantic = batch["semantic"]
        depth = batch["depth"]
        mask = batch["mask"]

        print(f"xyz : {xyz.shape}, max : {xyz.max()}, min : {xyz.min()}")
        print(f"reflectance : {reflectance.shape}, max : {reflectance.max()}, min : {reflectance.min()}")
        print(f"semantic : {semantic.shape}, max : {semantic.max()}, min : {semantic.min()}")
        print(f"depth : {depth.shape}, max : {depth.max()}, min : {depth.min()}")
        print(f"mask : {mask.shape}")

        break

    # import open3d
    # # bin_path = "/data/qwt/dataset/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000000.bin"
    # # points = common.get_lidar_sweep(bin_path, return_intensity=False)
    # # pc = o3d.geometry.PointCloud()
    # # pc.points = o3d.utility.Vector3dVector(points)
    # # o3d.io.write_point_cloud(filename="/data/qwt/temp/test.ply", pointcloud=pc)
    #
    # from utils import lidar
    # import torch
    #
    # resolution = (64,1024) # (32,1024) (64, 1024)
    # depth_range = (1.45, 80.0) # (0.0001,50.0) (1.45, 80.0)
    # fov = (3,-25)
    #
    # li = lidar.LiDARUtility(
    #     resolution=resolution,
    #     depth_range=depth_range,
    #     fov=fov,
    #     project_dir="/data/qwt/temp"
    # )
    #
    # ply_path = "/data/qwt/temp/test.ply"
    #
    # pc = open3d.io.read_point_cloud(ply_path)
    # points = np.asarray(pc.points)
    #
    # points = torch.from_numpy(points)
    #
    # points = common.midpoint_interpolate(points.unsqueeze(0).permute(0,2,1).cuda(), up_rate=2).cpu()
    # points = points.permute(0,2,1).squeeze()
    #
    # range_img = common.points_as_images_torch(points, size=resolution).permute(2, 0, 1)
    #
    # range_img = range_img.unsqueeze_(0)
    #
    # range_img = li.convert_depth(range_img)
    # range_img = li.normalize(range_img)
    #
    # # ---- 保存 ----
    # li.sample_to_lidar(metric=range_img)
    # # ---- 保存 ----

    pass