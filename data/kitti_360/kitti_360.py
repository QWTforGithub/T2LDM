from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from utils import common

SCENES = [0, 2, 3, 4, 5, 6, 7, 9, 10]

class KITTI360Dataset(Dataset):
    def __init__(
            self,
            data_root="/outputs/kitti360/KITTI-360/data_3d_raw",
            training=True,
            aug=["rotation", "flip"],

            resolution=(64,1024),
            depth_range=[1.45,80.0],
            fov=[3,-25],

            print_info=True
    ):
        super().__init__()

        self.data_root = data_root
        self.training = training
        self.aug = aug

        self.resolution = resolution
        self.depth_range = depth_range
        self.fov = fov

        self.transform = common.get_lidar_transform(self.aug, self.training)

        file_paths = []
        for scene in SCENES:
            wildcard = f"*_{scene:04d}_sync/velodyne_points/data/*.bin"
            file_paths += sorted(Path(data_root).glob(wildcard))

        if(print_info):
            print(f" ---- KITTI-360 Dataset with {len(file_paths)} ---- ")
        self.items = list(zip(range(len(file_paths)), file_paths))

        return

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample_id, file_path = self.items[idx]
        # data_path = "/outputs/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0003_sync/velodyne_points/data/0000000049.bin"
        points = common.get_lidar_sweep(file_path, return_intensity=True)

        # common.save_points(points[:,:3], name=f"{idx}.ply")

        if self.transform:
            points[:,:3],_ = self.transform(points[:,:3])

        range_image = common.points_as_images(
            points,
            size = self.resolution,
            fov = self.fov,
            depth_range = self.depth_range,
            return_all=True
        )

        range_image = range_image.transpose(2, 0, 1)

        sample = {
            "id": file_path,
            "batch": [len(points), ],
            "points": points[:, :3],             # (N,3)
            "xyz": range_image[:3],              # (3 H, W)
            "reflectance": range_image[[3]],     # (1, H, W)
            "depth": range_image[[4]],           # (1, H, W)
            "mask": range_image[[5]],            # (1, H, W)
        }

        return sample


if __name__ == '__main__':

    # range_map = load_points_as_images(
    #     file_path,
    #     scan_unfolding=(self.projection == "unfolding"),
    #     W=self.width,
    #     H=self.height,
    #     min_depth=self.min_depth,
    #     max_depth=self.max_depth
    # )

    data_root = "/outputs/kitti360/KITTI-360/data_3d_raw"
    dataset = KITTI360Dataset(data_root=data_root)

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
        #print(batch)

    # import random
    #
    # file_paths = []
    # for scene in SCENES:
    #     wildcard = f"*_{scene:04d}_sync/velodyne_points/data/*.bin"
    #     file_paths += sorted(Path(data_root).glob(wildcard))
    #
    # file_paths = random.sample(file_paths, 1024)
    #
    # print(f" ---- KITTI-360 Dataset with {len(file_paths)} ---- ")
    #
    # for idx, file_path in enumerate(file_paths):
    #     points = common.get_lidar_sweep(file_path, return_intensity=True)
    #
    #     common.save_points(points[:,:3], name=f"plys/{idx}.ply")

    pass