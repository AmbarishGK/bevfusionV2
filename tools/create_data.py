#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import pickle
from typing import Dict, List, Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

LIDAR_SENSOR = "LIDAR_TOP"
CAM_SENSORS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def _pose_to_4x4(rot, trans) -> np.ndarray:
    w, x, y, z = rot
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=np.float32,
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array(trans, dtype=np.float32)
    return T

def collect_sweeps(
    nusc: NuScenes, lidar_sd, max_sweeps: int
) -> list[dict]:
    """Collect up to max_sweeps previous non-keyframe lidar sweeps."""
    sweeps: list[dict] = []
    base_time = lidar_sd["timestamp"]

    curr_sd = lidar_sd
    num = 0
    while num < max_sweeps:
        prev_token = curr_sd["prev"]
        if prev_token == "":
            break
        prev_sd = nusc.get("sample_data", prev_token)
        # stop at previous keyframe; BEVFusion/CenterPoint stay inside keyframe history
        if prev_sd["is_key_frame"]:
            break

        prev_cs = nusc.get("calibrated_sensor", prev_sd["calibrated_sensor_token"])
        prev_ep = nusc.get("ego_pose", prev_sd["ego_pose_token"])

        sweeps.append(
            {
                "data_path": osp.join(nusc.dataroot, prev_sd["filename"]),
                "lidar2ego": _pose_to_4x4(prev_cs["rotation"], prev_cs["translation"]),
                "ego2global": _pose_to_4x4(prev_ep["rotation"], prev_ep["translation"]),
                # time lag (seconds) relative to keyframe; negative
                "time_lag": (prev_sd["timestamp"] - base_time) * 1e-6,
            }
        )

        curr_sd = prev_sd
        num += 1

    return sweeps

def collect_single_sample_info(
    nusc: NuScenes, sample_token: str, max_sweeps: int
) -> Dict[str, Any]:
    sample = nusc.get("sample", sample_token)

    # lidar
    lidar_sd_token = sample["data"][LIDAR_SENSOR]
    lidar_sd = nusc.get("sample_data", lidar_sd_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    lidar2ego = _pose_to_4x4(lidar_cs["rotation"], lidar_cs["translation"])
    ego2global = _pose_to_4x4(lidar_ep["rotation"], lidar_ep["translation"])
    lidar_path = osp.join(nusc.dataroot, lidar_sd["filename"])
    sweeps = collect_sweeps(nusc, lidar_sd, max_sweeps)


    # cameras
    cam_infos: Dict[str, Any] = {}
    for cam_name in CAM_SENSORS:
        cam_sd_token = sample["data"].get(cam_name, None)
        if cam_sd_token is None:
            continue
        cam_sd = nusc.get("sample_data", cam_sd_token)
        cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_ep = nusc.get("ego_pose", cam_sd["ego_pose_token"])

        sensor2ego = _pose_to_4x4(cam_cs["rotation"], cam_cs["translation"])
        ego2global_cam = _pose_to_4x4(cam_ep["rotation"], cam_ep["translation"])
        cam_path = osp.join(nusc.dataroot, cam_sd["filename"])
        intrinsics = np.array(cam_cs["camera_intrinsic"], dtype=np.float32)

        cam_infos[cam_name] = {
            "data_path": cam_path,
            "sensor2ego": sensor2ego,
            "ego2global": ego2global_cam,
            "intrinsics": intrinsics,
        }

    # annotations
    ann_infos: List[Dict[str, Any]] = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        ann_infos.append(
            {
                "translation": np.array(ann["translation"], dtype=np.float32),
                "size": np.array(ann["size"], dtype=np.float32),
                "rotation": np.array(ann["rotation"], dtype=np.float32),
                "velocity": np.array(nusc.box_velocity(ann["token"]), dtype=np.float32),
                "category_name": ann["category_name"],
                "num_lidar_pts": ann["num_lidar_pts"],
                "num_radar_pts": ann["num_radar_pts"],
            }
        )

        info = {
        "sample_token": sample_token,
        "scene_token": sample["scene_token"],
        "timestamp": sample["timestamp"],
        "lidar_path": lidar_path,
        "lidar2ego": lidar2ego,
        "ego2global": ego2global,
        "cams": cam_infos,
        "anns": ann_infos,
        "sweeps": sweeps,
    }

    return info


def create_nuscenes_infos(
    version: str, root: str, out_dir: str, max_sweeps: int = 10
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=root, verbose=True)
    splits = create_splits_scenes()

    if "mini" in version:
        train_scenes = set(splits["mini_train"])
        val_scenes = set(splits["mini_val"])
    else:
        train_scenes = set(splits["train"])
        val_scenes = set(splits["val"])

    scene_split: Dict[str, str] = {}
    for scene in nusc.scene:
        name = scene["name"]
        if name in train_scenes:
            scene_split[scene["token"]] = "train"
        elif name in val_scenes:
            scene_split[scene["token"]] = "val"

    train_infos: List[Dict[str, Any]] = []
    val_infos: List[Dict[str, Any]] = []

    for sample in nusc.sample:
        scene_token = sample["scene_token"]
        split = scene_split.get(scene_token, None)
        if split is None:
            continue

        info = collect_single_sample_info(nusc, sample["token"], max_sweeps)
        if split == "train":
            train_infos.append(info)
        elif split == "val":
            val_infos.append(info)

    train_path = osp.join(out_dir, "infos_train.pkl")
    val_path = osp.join(out_dir, "infos_val.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(train_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(val_path, "wb") as f:
        pickle.dump(val_infos, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(train_infos)} train infos to {train_path}")
    print(f"Saved {len(val_infos)} val infos to {val_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create nuScenes BEVFusion data infos")
    parser.add_argument("dataset", choices=["nuscenes"])
    parser.add_argument(
        "--root",
        required=True,
        help="Root of nuScenes (folder containing v1.0-mini, samples, sweeps, maps, etc.).",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version (e.g. v1.0-mini, v1.0-trainval).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output folder for processed infos (e.g. data/nuscenes_mini_bevfusion).",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=10,
        help="Number of previous lidar sweeps to aggregate (OG BEVFusion uses 10).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "nuscenes":
        create_nuscenes_infos(
            version=args.version,
            root=args.root,
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported yet.")


if __name__ == "__main__":
    main()
