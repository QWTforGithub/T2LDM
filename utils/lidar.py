from typing import Literal
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import open3d
import glob
import pickle
from utils.common import get_hdl64e_linear_ray_angles,get_color, encode_strings

class LiDARUtility(nn.Module):
    def __init__(
        self,
        resolution = (64, 1024),
        image_format: Literal["log_depth", "inverse_depth", "depth"] = "log_depth",
        depth_range = (1.45,80.0),
        fov = (3, -25),
        ray_angles: torch.Tensor = None,
        project_dir: str = "temp"
    ):
        super().__init__()
        assert image_format in ("log_depth", "inverse_depth", "depth")
        self.resolution = resolution
        self.image_format = image_format
        self.min_depth = depth_range[0]
        self.max_depth = depth_range[1]
        self.project_dir = project_dir
        self.fov = fov

        if ray_angles is None:
            ray_angles = get_hdl64e_linear_ray_angles(
                resolution=resolution,
                fov=fov
            )
        else:
            assert ray_angles.ndim == 4 and ray_angles.shape[1] == 2
        ray_angles = F.interpolate(
            ray_angles,
            size=self.resolution,
            mode="nearest-exact",
        )
        self.register_buffer("ray_angles", ray_angles)

    @staticmethod
    def denormalize(x):
        """Scale from [-1, +1] to [0, 1]"""
        return (x + 1) / 2

    @staticmethod
    def normalize(x):
        """Scale from [0, 1] to [-1, +1]"""
        return x * 2 - 1

    @torch.no_grad()
    def to_xyz(self, metric):
        assert len(metric.shape) == 4
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        phi = self.ray_angles[:, [0]]
        theta = self.ray_angles[:, [1]]
        grid_x = metric * phi.cos() * theta.cos()
        grid_y = metric * phi.cos() * theta.sin()
        grid_z = metric * phi.sin()
        xyz = torch.cat((grid_x, grid_y, grid_z), dim=1)
        xyz = xyz * mask.float()
        return xyz

    @torch.no_grad()
    def points_4dim_to_3dim(self, metric):
        assert len(metric.shape) == 4
        B, C, H, W = metric.shape
        if(isinstance(metric, torch.Tensor)):
            xyz = metric.permute(0, 2, 3, 1).reshape(B, H * W, C).cpu().numpy()
        else:
            xyz = metric.transpose(0, 2, 3, 1).reshape(B, H * W, C)
        return xyz

    @torch.no_grad()
    def save_pointcloud(self, xyz, colors=None, name="point_cloud.ply"):
        assert len(xyz.shape) == 2
        if (isinstance(xyz, torch.Tensor)):
            xyz = xyz.cpu().numpy()
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(xyz)
        if(colors is not None):
            pc.colors = open3d.utility.Vector3dVector(colors)
        open3d.io.write_point_cloud(name, pc)
        print(f"---- Saved point cloud to {name} ----")

    @torch.no_grad()
    def save_noise(self, noise, name="noise.npy"):
        assert len(noise.shape) == 5
        if (isinstance(noise, torch.Tensor)):
            noise = noise.cpu().numpy()
        np.save(file=name, arr=noise)
        print(f"---- Saved noise to {name} ----")

    def save_semantic(self, semantic, name="semantic.npy"):
        assert len(semantic.shape) == 2
        if (isinstance(semantic, torch.Tensor)):
            semantic = semantic.cpu().numpy()
        np.save(file=name, arr=semantic)
        print(f"---- Saved semantic to {name} ----")

    def save_pkl(self, info, name="pkl.pkl"):
        with open(name, "wb") as f:
            pickle.dump(info, f)
        print(f"---- Saved pkl to {name} ----")

    @torch.no_grad()
    def sample_to_lidar(
            self,
            generation=None,        # denosing net生成的range image [B,1,H,W]
            upsampling=None,        # 上采样生成的range image [B,1,H,W]
            downsampling=None,      # 下采样生成的range image [B,1,H,W]

            num_step=0,             # 训练迭代次数
            num_sample=0,           # 采样次数

            g=None,                 # guidence net生成的range image [B,1,H,W]
            noise=None,             # 采样的噪声
            text=None,              # 文本，列表 ["xxx", "xxx"]
            batches=None,           # 每个点云的点数量, [B,]
            points=None,            # 所有点云的堆叠, [N,3]（需要通过batches来区分每个点云）
            semantic=None,          # 语义map [B,1,H,W]
            xyz=None,               # 与conditional net对应的range image [B,1,H,W]
            rank=0,                 # rank
            dataset="nuscenes"      # 数据集
    ):

        if(generation is not  None):
            assert len(generation.shape) == 4

            generation = self.denormalize(generation) # 将取值从 [-1,1]规划嗷[0,1]
            generation = self.revert_depth(generation)  # 将image_format形式 range image转为正常形式
            generation = self.to_xyz(generation) # range image转换为点云
            generation = self.points_4dim_to_3dim(generation) # dim4点云[B,C,H,W]转回dim2点云[B,N,3]

            generation_dir = f"{self.project_dir}/generation"
            os.makedirs(generation_dir, exist_ok=True)

            # 保存每个batch的点云
            for i in range(generation.shape[0]):
                self.save_pointcloud(xyz=generation[i], colors=None, name=f"{generation_dir}/generation_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}_batch_{i}.ply")

        if(upsampling is not  None):
            assert len(upsampling.shape) == 4

            upsampling = self.denormalize(upsampling) # 将取值从 [-1,1]规划嗷[0,1]
            upsampling = self.revert_depth(upsampling)  # 将image_format形式 range image转为正常形式
            upsampling = self.to_xyz(upsampling) # range image转换为点云
            upsampling = self.points_4dim_to_3dim(upsampling) # dim4点云[B,C,H,W]转回dim2点云[B,N,3]

            upsampling_dir = f"{self.project_dir}/upsampling"
            os.makedirs(upsampling_dir, exist_ok=True)

            # 保存每个batch的点云
            for i in range(upsampling.shape[0]):
                self.save_pointcloud(xyz=upsampling[i], colors=None, name=f"{upsampling_dir}/upsampling_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}_batch_{i}.ply")

        if(downsampling is not  None):
            assert len(downsampling.shape) == 4

            downsampling = self.denormalize(downsampling) # 将取值从 [-1,1]规划嗷[0,1]
            downsampling = self.revert_depth(downsampling)  # 将image_format形式 range image转为正常形式
            downsampling = self.to_xyz(downsampling) # range image转换为点云
            downsampling = self.points_4dim_to_3dim(downsampling) # dim4点云[B,C,H,W]转回dim2点云[B,N,3]

            downsampling_dir = f"{self.project_dir}/downsampling"
            os.makedirs(downsampling_dir, exist_ok=True)

            # 保存每个batch的点云
            for i in range(downsampling.shape[0]):
                self.save_pointcloud(xyz=downsampling[i], colors=None, name=f"{downsampling_dir}/downsampling_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}_batch_{i}.ply")

        if(xyz is not None):
            assert len(xyz.shape) == 4

            xyz_dir = f"{self.project_dir}/xyz"
            os.makedirs(xyz_dir, exist_ok=True)

            xyz = self.points_4dim_to_3dim(xyz)
            for i in range(xyz.shape[0]):
                self.save_pointcloud(xyz=xyz[i], colors=None, name=f"{xyz_dir}/xyz_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}_batch_{i}.ply")

        if(semantic is not None):

            assert len(semantic.shape) == 4 and len(xyz.shape) == 4

            semantic_dir = f"{self.project_dir}/semantic"
            os.makedirs(semantic_dir, exist_ok=True)

            semantic = self.points_4dim_to_3dim(semantic).astype(int)
            semantic = np.squeeze(semantic)
            self.save_semantic(semantic=semantic, name=f"{semantic_dir}/semantic_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}.npy")

            colors = get_color()

            for i,labels in enumerate(semantic):

                gt_color = []
                for label in labels:
                    gt_color.append(colors[label])
                gt_color = np.asarray(gt_color)

                self.save_pointcloud(xyz=xyz[i], colors=gt_color, name=f"{semantic_dir}/xyzcolor_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}_batch_{i}.ply")
                self.save_pointcloud(xyz=generation[i], colors=gt_color, name=f"{semantic_dir}/generationcolor_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}_batch_{i}.ply")

        if(g is not None):
            assert len(g.shape) == 4

            g = self.denormalize(g) # 将取值从 [-1,1]规划嗷[0,1]
            g = self.revert_depth(g)  # 将image_format形式 range image转为正常形式
            g = self.to_xyz(g) # range image转换为点云
            g = self.points_4dim_to_3dim(g) # dim4点云[B,C,H,W]转回dim2点云[B,N,3]

            conditional_dir = f"{self.project_dir}/conditional"
            os.makedirs(conditional_dir, exist_ok=True)

            # 保存每个batch的点云
            for i in range(g.shape[0]):
                self.save_pointcloud(xyz=g[i], colors=None, name=f"{conditional_dir}/guidence_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}_batch_{i}.ply")

        if(points is not None):
            assert len(points.shape) == 2

            points_dir = f"{self.project_dir}/points"
            os.makedirs(points_dir, exist_ok=True)

            start = 0
            end = 0
            for i,batch in enumerate(batches):
                end += batch
                self.save_pointcloud(xyz=points[start:end,:], colors=None, name=f"{points_dir}/points_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}_batch_{i}.ply")
                start += batch

        if(noise is not None):
            assert len(noise.shape) == 5

            noise_dir = f"{self.project_dir}/noise"
            os.makedirs(noise_dir, exist_ok=True)

            self.save_noise(noise=noise, name=f"{noise_dir}/noise_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}.npy")

        if(text is not None):

            # 字符编码，读取时需要利用common.decode_tensor解码
            text = encode_strings(str_list=text, max_len=64)

            text_dir = f"{self.project_dir}/text"
            os.makedirs(text_dir, exist_ok=True)

            for i in range(generation.shape[0]):
                self.save_pkl(info=text[[i]], name=f"{text_dir}/text_rank_{rank}_dataset_{dataset}_step_{num_step}_sample_{num_sample}_batch_{i}.pkl")


    @torch.no_grad()
    def convert_depth(
        self,
        metric: torch.Tensor,
        mask: torch.Tensor | None = None,
        image_format: str = None,
    ) -> torch.Tensor:
        """
        Convert metric depth in [0, `max_depth`] to normalized depth in [0, 1].
        """
        if image_format is None:
            image_format = self.image_format
        if mask is None:
            mask = self.get_mask(metric)
        if image_format == "log_depth":
            normalized = torch.log2(metric + 1 + 0.0001) / np.log2(self.max_depth + 1 + 0.0001)
        elif image_format == "inverse_depth":
            normalized = self.min_depth / metric.add(1e-8)
        elif image_format == "depth":
            normalized = metric.div(self.max_depth)
        else:
            raise ValueError
        normalized = normalized.clamp(0, 1) * mask
        return normalized

    @torch.no_grad()
    def revert_depth(
        self,
        normalized: torch.Tensor,
        image_format: str = None,
    ) -> torch.Tensor:
        """
        Revert normalized depth in [0, 1] back to metric depth in [0, `max_depth`].
        """
        if image_format is None:
            image_format = self.image_format
        if image_format == "log_depth":
            metric = torch.exp2(normalized * np.log2(self.max_depth + 1 + 0.0001)) - 1 - 0.0001
        elif image_format == "inverse_depth":
            metric = self.min_depth / normalized.add(1e-8)
        elif image_format == "depth":
            metric = normalized.mul(self.max_depth)
        else:
            raise ValueError
        return metric * self.get_mask(metric)

    def get_mask(self, metric):
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        return mask.float()

def get_lidars(
        root_dir: str = "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP"
):
    names = glob.glob(f"{root_dir}/*.bin")

    return names

if __name__ == '__main__':

    # from utils import common
    #
    # resolution = (32,2048) # (32,1024) (64, 1024)
    # depth_range = (0.01,50.0) # (0.0001,50.0) (1.45, 80.0)
    # fov = (3,-25)
    #
    # li = LiDARUtility(
    #     resolution=resolution,
    #     depth_range=depth_range,
    #     fov=fov,
    #     project_dir="/data/qwt/temp"
    # )
    #
    # ply_path = "/data/qwt/temp/1526915630897851.pcd.bin.ply"
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
    # # ---- 保存 ----
    # depth = range_img.squeeze().numpy()
    # common.save_img("/data/qwt/temp/depth.png", img=depth, depth_color=True)
    # # ---- 保存 ----
    #
    # range_img = range_img.unsqueeze_(0)
    #
    # range_img = li.convert_depth(range_img)
    # range_img = li.normalize(range_img)
    #
    # # ---- 保存 ----
    # li.sample_to_lidar(metric=range_img)
    # # ---- 保存 ----

    from utils import common

    resolution = (32,1024) # (64,1024)
    depth_range = (0.01,50.0)  # (1.45,80.0)
    fov = (3,-25)

    li = LiDARUtility(
        resolution=resolution,
        depth_range=depth_range,
        fov=fov,
        project_dir="/data/qwt/temp"
    )

    # 532,1024,11458,
    lidar_filename = '/data/qwt/dataset/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000011458.bin'
    lidar_filename = '/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-09-18-13-10-39-0400__LIDAR_TOP__1537290847200028.pcd.bin'
    # lidar_filename = '/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-09-18-14-43-59-0400__LIDAR_TOP__1537296489198821.pcd.bin'
    # lidar_filename = '/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-08-28-16-16-48-0400__LIDAR_TOP__1535487741196823.pcd.bin'
    # lidar_filename = '/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-08-29-16-04-13-0400__LIDAR_TOP__1535573300398646.pcd.bin'


    # points = common.get_lidar_sweep(lidar_filename, return_intensity=True, return_time=True, dim=5)
    points = common.get_lidar_sweep(lidar_filename,dim=5)

    range_img = common.points_as_images(
        points=points,
        size=resolution,
        depth_range=depth_range,
        fov=fov,
        return_all=False
    ).transpose(2, 0, 1)

    range_img = np.expand_dims(range_img, axis=0)
    range_img = torch.from_numpy(range_img)

    range_img = li.convert_depth(range_img)
    range_img = li.normalize(range_img)

    li.sample_to_lidar(range_img)

    name = lidar_filename.split("/")[-1].split("__")[-1]

    li.save_pointcloud(xyz=points[:,:3], name=f"{li.project_dir}/{name}.ply")

    pass

