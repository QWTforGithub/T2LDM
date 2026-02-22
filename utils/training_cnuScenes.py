import dataclasses
import math
from typing import Literal

import torch
from torch.optim.lr_scheduler import LambdaLR


@dataclasses.dataclass
class TrainingConfig:

    # ---- dataset setting ----
    dataset: Literal["kitti_raw", "kitti_360", "nuScenes"] = "nuScenes"
    data_root: str = "/ihoment/youjie10/qwt/dataset/nuscenes"
    aug: tuple[int, int] = ("-", "-") # ("rotation", "flip")
    resolution: tuple[int, int] = (32, 1024)
    depth_range: tuple[int, int] = (0.01, 50.0) # 0.01
    fov: tuple[int, int] = (3, -25)
    batch_size_train: int = 4 # 2 # 16
    batch_size_eval: int = 4
    num_workers: int = 4# 2 # 16
    text_name: str = "text_l1"
    pkl: str = "nuscenes_infos_10sweeps_description.pkl"
    version: str = "v1.0-trainval"
    only_class: int = 0
    # ---- dataset setting ----

    # ---- model setting ----
    model_name: str = "circular_unet"
    in_channel: int = 2
    out_channel: int = 2
    base_channel: int = 64
    channels: tuple[int, int, int, int] = (1, 2, 4, 4)
    strides: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((1, 2), (2, 2), (2, 2))
    dropout: float = 0.0
    resamp_with_conv: bool = True

    # 以下设置表示使用无条件生成
    attn_types: tuple[str, str, str, str] = ("linear", "linear", "linear", "vanilla")
    norm_types: tuple[str, str, str, str] = ("gn", "gn", "gn", "gn")
    use_text: bool = False
    midd_attn_type: str = 'vanilla'
    midd_norm_type: str = 'gn'
    use_seg: bool = False

    # 以下设置表示使用条件生成
    # attn_types: tuple[str, str, str, str] = ("nn", "nn", "nn", "nn")
    # norm_types: tuple[str, str, str, str] = ("ln", "ln", "ln", "ln")
    # use_text: bool = True
    # midd_attn_type: str = 'nn'
    # midd_norm_type: str = 'ln'
    # use_seg: bool = False

    use_axialwconv: bool = False
    use_rope: bool = True
    # res_connection: tuple[bool, bool, bool, bool] = (False, False, False, False)
    res_connection: tuple[bool, bool, bool, bool] = (True, True, True, True)
    use_norm: tuple[bool, bool, bool, bool] = (True, True, True, True)
    midd_use_norm: bool = True
    midd_res_connection: bool = True
    midd_heads: int = 8
    midd_channels: tuple[int, int, int] = (4, 8, 4)
    heads: tuple[int, int, int, int] = (2, 4, 8, 8)
    use_timestep: bool = True
    use_pe: bool = False
    skip_connection_scale: str = "sqrt(2)" # "equal" "sqrt(2)" "scalelong"
    model_gn_num_groups: int = 32 // 4
    model_gn_eps: float = 1e-6

    clip_mode: str = "ViT-L/14" # "ViT-B/32",
    clip_pool_features: bool = False # True -> [B,768/512], False -> [B,77,768/512]
    clip_channels: int = 768
    # ---- model setting ----

    # ---- training setting ----
    image_format: str = "log_depth"
    lidar_projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    train_depth: bool = True
    train_reflectance: bool = True
    train_mask: bool = True # True
    num_steps: int = 400_000#300_000 #625_000 # 625_0000 # 300_000
    save_image_steps: int = 5_000 # 1
    save_model_steps: int = 5_000 # 1
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4 # 1e-4， 1e-5(效果一般)
    lr_warmup_steps: int = 2_000 # 16_000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999 # 0.99
    adam_weight_decay: float = 0.01 # 0.0
    adam_epsilon: float = 1e-8
    ema_decay: float = 0.9997 # 0.995
    ema_update_every: int = 1 # 10 # 20
    output_dir: str = "logs/diffusion"
    seed: int = 0
    mixed_precision: str = "no" # "fp16", "no"
    dynamo_backend: str = None # "inductor", "no", None
    checkpoint_dir: str = None
    # ---- training setting ----

    # ---- diffusion setting ----
    criterion: str = "l2"
    diffusion_num_training_steps: int = 1024 #2048
    diffusion_num_sampling_steps: int = 1024 #2048
    diffusion_objective: Literal["eps", "v", "x0"] = "v" # "v"
    diffusion_beta_schedule: str = "cosine"
    diffusion_timesteps_type: Literal["continuous", "discrete"] = "continuous"
    diffusion_sampling: str = "ddpm" # "ddpm", "ddim"
    diffusion_min_snr_loss_weight: bool = True
    diffusion_min_snr_gamma: float = 5.0
    diffusion_clip_sample: bool = True
    diffusion_clip_sample_range: float = 1.0

    diffusion_use_x_0_condition_guide: bool = True
    diffusion_use_condition_guide: bool = True
    diffusion_condition_guide_step_interval: int = 100_000
    diffusion_condition_guide_weights: tuple[float, float, float, float] = (0.001, 0.01, 0.1, 1.0)

    # 以下设置表示不使用classifier_free_scale训练和推理
    diffusion_classifier_free_scale: float = None  # 7.5，设置为非None表示使用classifier_free_scale推理
    diffusion_classifier_dropout: float = 0.0 # [0.0, 0.9] ==> 0.0意味着完整文本条件; [0.1,0.9]意味着存在[""]文本条件

    # # 以下设置表示使用classifier_free_scale训练和推理
    # diffusion_classifier_free_scale: float = 7.5  # 7.5，设置为非None表示使用classifier_free_scale推理
    # diffusion_classifier_dropout: float = 0.1 # [0.0, 0.9] ==> 0.0意味着完整文本条件; [0.1,0.9]意味着存在[""]文本条件
    #
    # # 以下设置表示使用classifier_free_scale训练，但不使用classifier_free_scale推理
    # diffusion_classifier_free_scale: float = None  # 7.5，设置为非None表示使用classifier_free_scale推理
    # diffusion_classifier_dropout: float = 0.1 # [0.0, 0.9] ==> 0.0意味着完整文本条件; [0.1,0.9]意味着存在[""]文本条件

    # ---- diffusion setting ----
