import argparse
import json
import os
import dataclasses
import torch
import datetime
import utils.render
import shutil
import inspect
from pathlib import Path
from accelerate import Accelerator
from utils import common
import torch.distributed as dist
from simple_parsing import ArgumentParser
import time

from data.conditional_x0.conditionalx0 import ConditionalX0
from torch.utils.data import DataLoader
from models.CLIP.clip import clip

from models.T2LDM import CircularUNet
from utils.config_unconditional_nuScenes import TrainingConfig

import utils.inference_mgpus_unconditional_nuScenes as inference_mgpus

def main(args, cfg):

    task = inspect.getfile(TrainingConfig).split("/")[-1].split("_")[1]
    project_dir = "test"
    project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    project_name = f"{project_name}_{args.num_steps}_{args.sampling_mode}{args.sampling_steps}_{task}_{cfg.dataset}_{args.seed}"
    dest_path = os.path.join(project_dir, project_name)
    os.makedirs(dest_path, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="no",
        log_with=["tensorboard"],
        project_dir=dest_path,
        dynamo_backend=cfg.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True,
    )

    device = accelerator.device

    if accelerator.is_main_process:

        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / "training_config.json", "w"),
            indent=4,
        )

        config_path = inspect.getfile(TrainingConfig)
        config_dest_path = os.path.join(dest_path, config_path.split("/")[-1])
        shutil.copy(config_path, config_dest_path)
        print(f"Coping config file to {config_dest_path}")

        net_path = inspect.getfile(CircularUNet)
        net_dest_path = os.path.join(dest_path, net_path.split("/")[-1])
        shutil.copy(net_path, net_dest_path)
        print(f"Coping network file to {net_dest_path}")

        inference_path = str(Path(inference_mgpus.__file__).resolve())
        inference_dest_path = os.path.join(dest_path, inference_path.split("/")[-1])
        shutil.copy(inference_path, inference_dest_path)
        print(f"Coping inference file to {inference_dest_path}")

        generate_path = str(Path(__file__).resolve())
        generate_dest_path = os.path.join(dest_path, generate_path.split("/")[-1])
        shutil.copy(generate_path, generate_dest_path)
        print(f"Coping generate file to {generate_dest_path}")

        train_name = config_path.split("/")[-1].replace("config_","train_")
        training_path = f"{str(Path(__file__).parent.resolve())}/{train_name}"
        training_dest_path = os.path.join(dest_path, training_path.split("/")[-1])
        shutil.copy(training_path, training_dest_path)
        print(f"Coping train file to {training_dest_path}")

        print("\nAccelerator配置信息: ")
        print(accelerator.state)

    # =================================================================================
    # Load pre-trained model
    # =================================================================================

    ddpm, lidar_utils, cfg = inference_mgpus.setup_model(
        ckpt_path=args.ckpt,
        device=args.device,
        sampling_mode=args.sampling_mode,
        project_dir=dest_path
    )

    ddpm = accelerator.prepare(ddpm)

    seed = common.setup_seed(args.seed) # ---- 设置随机种子 ----
    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"[Rank {rank}] Random check: {torch.randint(0, 10000, (1,))}")

    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================
    text_features = None
    text_null_features = None
    text_original = None
    semantic = None
    batches = None
    semantic_org = None
    points = None
    xyz = None

    if(cfg.use_text or cfg.use_text):
        condition_guide_dataset = ConditionalX0(
            data_root=cfg.data_root,
            training=False,
            aug=cfg.aug,
            resolution=cfg.resolution,
            depth_range=cfg.depth_range,
            fov=cfg.fov,
            type=cfg.dataset
        )

        condition_guide_dataloader = DataLoader(
            condition_guide_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=common.collate_fn
        )

        for batch in condition_guide_dataloader:
            conditional_x_0, text_input, text_original, semantic, points, batches, semantic_org, xyz = common.preprocess(
                batch=batch,
                classifier_dropout=cfg.diffusion_classifier_dropout,
                use_text=cfg.use_text,
                use_semantic=cfg.use_seg,
                train_depth=cfg.train_depth,
                train_reflectance=cfg.train_reflectance,
                lidar_utils=lidar_utils
            )
            break

        if(cfg.use_text):
            clip_model = clip.load(cfg.clip_mode, device=device)

            def get_text_features(text, text_null):
                text_emb = clip.tokenize(text).to(device)
                text_null_emb = clip.tokenize(text_null).to(device)
                with torch.no_grad():
                    text_features = clip_model.encode_text(text_emb, cfg.clip_pool_features)
                    text_null_features = clip_model.encode_text(text_null_emb, cfg.clip_pool_features)
                return text_features, text_null_features

            text_null = [""] * len(text_original)
            print(f"\ntext_all: {text_original}")
            print(f"\ntext_null: {text_null}")
            text_features, text_null_features = get_text_features(text_original, text_null)
            text_original = common.encode_strings(str_list=text_original, max_len=64).cuda()

    start_time = time.time()
    x, noise = accelerator.unwrap_model(ddpm).sample(
        batch_size=args.batch_size,
        num_steps=args.sampling_steps,
        return_noise=False,
        mode=args.sampling_mode,
        text_features=text_features,
        text_null_features=text_null_features,
        semantic=semantic
    )
    all_time = time.time() - start_time
    avg_time = all_time / 4 / args.batch_size
    print(f"all time : {all_time}, avg time : {avg_time}")

    x = x.clamp(-1, 1)

    x = accelerator.gather(x)
    if(cfg.use_seg):
        semantic_org = accelerator.gather(semantic_org)
        xyz = accelerator.gather(xyz)

    if(cfg.use_text):
        text_original = accelerator.gather(text_original)

    # ---- save point cloud ----
    if accelerator.is_main_process:
        sample, _ = common.split_channels(cfg.train_depth, cfg.train_reflectance, x)
        lidar_utils.sample_to_lidar(
            sample,
            num_step=args.num_steps,
            num_sample=args.sampling_steps,
            rank=seed,
            noise=noise,
            text=text_original,
            semantic=semantic_org,
            xyz=xyz,
            dataset="nuscenes"
        )
        common.remove_empty_dirs(root=project_dir)
    # ---- save point cloud ----


if __name__ == "__main__":

    # log= "20260212T225739"
    log= "20260217T004549" # 当前的
    name = "diffusion_0000800000.pth"

    ckpt = f"/ihoment/youjie10/qwt/model/T2LDM/logs/diffusion/unconditional_nuScenes/{log}/models/{name}"

    seed = 22# 42#22
    batch_size = 8 # 64
    sampling_steps = 1024 # 1024
    sampling_mode = "ddpm" # ddpm ddim
    num_steps = int(ckpt.split("?")[-1].split("_")[-1].split(".")[0])

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=ckpt)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--sampling_steps", type=int, default=sampling_steps)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--num_steps", type=int, default=num_steps)
    parser.add_argument("--sampling_mode", type=str, default=sampling_mode)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    parser_cfg = ArgumentParser()
    parser_cfg.add_arguments(TrainingConfig, dest="cfg")
    cfg: TrainingConfig = parser_cfg.parse_args().cfg

    main(args, cfg)
