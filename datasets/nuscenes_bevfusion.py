import os.path as osp
import pickle
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def invert_4x4(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform T to Nx3 points."""
    N = pts.shape[0]
    homog = np.concatenate([pts, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N,4]
    out = (T @ homog.T).T  # [N,4]
    return out[:, :3]

# BEVFusion-style detection classes
DETECTION_CLASSES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(DETECTION_CLASSES)}


def nusc_category_to_detection_name(cat: str) -> str | None:
    """
    Map nuScenes category_name to one of DETECTION_CLASSES.
    Returns None for categories we want to ignore.
    """
    if cat.startswith("vehicle.car"):
        return "car"
    if cat.startswith("vehicle.truck"):
        return "truck"
    if cat.startswith("vehicle.bus"):
        return "bus"
    if cat.startswith("vehicle.trailer"):
        return "trailer"
    if cat.startswith("vehicle.construction"):
        return "construction_vehicle"

    if cat.startswith("human.pedestrian"):
        return "pedestrian"

    if cat.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if cat.startswith("vehicle.bicycle"):
        return "bicycle"

    if cat.startswith("movable_object.trafficcone"):
        return "traffic_cone"
    if cat.startswith("static_object.barricade"):
        return "barrier"

    # everything else (e.g. animals, unknown) – ignore for detection
    return None

def anns_to_boxes_and_labels(anns: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw nuScenes anns to:
      boxes: [M, 7] (x, y, z, dx, dy, dz, yaw) in global frame
      labels: [M] int64 (class indices)
    """
    boxes = []
    labels = []

    for ann in anns:
        det_name = nusc_category_to_detection_name(ann["category_name"])
        if det_name is None:
            continue  # skip classes we don't train on

        cls_idx = CLASS_TO_IDX[det_name]

        center = np.asarray(ann["translation"], dtype=np.float32)  # [3]
        size = np.asarray(ann["size"], dtype=np.float32)           # [3] (w,l,h)
        rot = np.asarray(ann["rotation"], dtype=np.float32)        # [4] (w,x,y,z)

        # Convert quaternion to yaw (heading around z)
        w, x, y, z = rot
        # rotation matrix R from quaternion
        R = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ],
            dtype=np.float32,
        )
        # yaw from rotation matrix (assuming z-up, nuScenes convention)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))

        # nuScenes size is [w, l, h]; we keep that order as dx, dy, dz for now
        dx, dy, dz = size[0], size[1], size[2]

        boxes.append([center[0], center[1], center[2], dx, dy, dz, yaw])
        labels.append(cls_idx)

    if len(boxes) == 0:
        boxes = np.zeros((0, 7), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)
    else:
        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

    return boxes, labels


CAM_SENSORS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def load_point_cloud_bin(path: str) -> np.ndarray:
    """Load nuScenes .bin point cloud: float32 [x, y, z, intensity]."""
    points = np.fromfile(path, dtype=np.float32)
    points = points.reshape(-1, 5)  # nuScenes lidar is x,y,z,intensity,ring
    # drop ring, keep x,y,z,intensity
    return points[:, :4]


def load_image(path: str, resize: Tuple[int, int] = None) -> np.ndarray:
    """Load image as float32 CHW in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        w, h = resize  # (W, H)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    # HWC -> CHW
    img = np.transpose(img, (0, 2, 1)) if img.shape[2] == 1 else np.transpose(img, (2, 0, 1))
    return img


class NuScenesBEVFusionDataset(Dataset):
    """
    Minimal nuScenes dataset for BEVFusion:
    - uses precomputed infos_{train,val}.pkl
    - loads lidar + 6 camera images
    - returns tensors + raw annotations
    """

    def __init__(
        self,
        info_path: str,
        image_size: Tuple[int, int] = (1600, 900),
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        point_cloud_range: list[float] | None = None,
    ) -> None:
        """
        Args:
            info_path: path to infos_train.pkl or infos_val.pkl
            image_size: (W, H) to resize images to
        """
        super().__init__()
        self.info_path = info_path
        self.image_size = image_size

        with open(info_path, "rb") as f:
            self.infos: List[Dict[str, Any]] = pickle.load(f)
        if image_mean is not None and image_std is not None:
            mean = np.asarray(image_mean, dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.asarray(image_std, dtype=np.float32).reshape(1, 3, 1, 1)
        else:
            mean, std = None, None

        self.image_mean = mean
        self.image_std = std
        if point_cloud_range is not None:
            self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        else:
            self.point_cloud_range = None

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.infos[idx]

        # lidar
        lidar_path = info["lidar_path"]
        lidar_points = load_point_cloud_bin(lidar_path)  # [N,4] np.float32
        if self.point_cloud_range is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
            pts_xyz = lidar_points[:, :3]
            mask = (
                (pts_xyz[:, 0] >= x_min)
                & (pts_xyz[:, 0] <= x_max)
                & (pts_xyz[:, 1] >= y_min)
                & (pts_xyz[:, 1] <= y_max)
                & (pts_xyz[:, 2] >= z_min)
                & (pts_xyz[:, 2] <= z_max)
            )
            lidar_points = lidar_points[mask]


        # cameras: fixed order in CAM_SENSORS
        imgs = []
        cam_meta = {}
        for cam_name in CAM_SENSORS:
            cam_info = info["cams"].get(cam_name, None)
            if cam_info is None:
                # some scenes might be missing a camera; fill with zeros
                w, h = self.image_size
                imgs.append(np.zeros((3, h, w), dtype=np.float32))
                cam_meta[cam_name] = None
                continue

            img = load_image(cam_info["data_path"], resize=self.image_size)  # [3,H,W]
            imgs.append(img)
            cam_meta[cam_name] = {
                "sensor2ego": cam_info["sensor2ego"],
                "ego2global": cam_info["ego2global"],
                "intrinsics": cam_info["intrinsics"],
            }

        imgs = np.stack(imgs, axis=0)  # [num_cams, 3, H, W]
        if self.image_mean is not None and self.image_std is not None:
            # broadcast: [1,3,1,1] over [num_cams,3,H,W]
            imgs = (imgs - self.image_mean) / self.image_std

        # annotations: keep raw for now
        anns = info["anns"]
        gt_boxes_global, gt_labels_np = anns_to_boxes_and_labels(anns)  # global frame

        # --- transform boxes center + yaw from global -> lidar frame ---
        lidar2ego = info["lidar2ego"]     # 4x4, lidar -> ego
        ego2global = info["ego2global"]   # 4x4, ego -> global

        # Build global -> lidar transform:
        #   P_global = ego2global @ lidar2ego @ P_lidar
        #   => T_lidar_global = (ego2global @ lidar2ego)^(-1)
        ego2global_mat = np.asarray(ego2global, dtype=np.float32)
        lidar2ego_mat = np.asarray(lidar2ego, dtype=np.float32)

        global2ego = invert_4x4(ego2global_mat)   # ego <- global
        ego2lidar = invert_4x4(lidar2ego_mat)     # lidar <- ego
        T_lidar_global = ego2lidar @ global2ego   # lidar <- global

        # centers in global frame
        centers_global = gt_boxes_global[:, :3]   # [M,3]
        centers_lidar = transform_points(T_lidar_global, centers_global)  # [M,3]

        # yaw: transform global heading into lidar frame
        yaw_global = gt_boxes_global[:, 6]        # [M]
        R_lidar_global = T_lidar_global[:3, :3]
        # yaw of the transform itself (rotation from global to lidar)
        yaw_offset = float(np.arctan2(R_lidar_global[1, 0], R_lidar_global[0, 0]))
        yaw_lidar = yaw_global + yaw_offset       # [M]

        sizes = gt_boxes_global[:, 3:6]           # [M,3] (dx, dy, dz same in any frame)

        gt_boxes_lidar = np.concatenate(
            [centers_lidar, sizes, yaw_lidar[:, None]], axis=1
        )  # [M,7]

        # --- BEV-range filtering in lidar frame ---
        if self.point_cloud_range is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
            cx, cy, cz = (
                gt_boxes_lidar[:, 0],
                gt_boxes_lidar[:, 1],
                gt_boxes_lidar[:, 2],
            )
            mask = (
                (cx >= x_min)
                & (cx <= x_max)
                & (cy >= y_min)
                & (cy <= y_max)
                & (cz >= z_min)
                & (cz <= z_max)
            )
            gt_boxes_lidar = gt_boxes_lidar[mask]
            gt_labels_np = gt_labels_np[mask]
        # --- multi-sweep aggregation (OG BEVFusion style) ---
        all_points = [lidar_points]  # list of [Ni,4] arrays, in keyframe lidar frame

        sweeps = info.get("sweeps", [])
        for sweep in sweeps:
            sweep_points = load_point_cloud_bin(sweep["data_path"])  # [Ns,4]
            if sweep_points.shape[0] == 0:
                continue

            sweep_lidar2ego = np.asarray(sweep["lidar2ego"], dtype=np.float32)
            sweep_ego2global = np.asarray(sweep["ego2global"], dtype=np.float32)

            # transform sweep points from sweep lidar -> global
            T_global_sweep = sweep_ego2global @ sweep_lidar2ego  # global <- sweep_lidar
            pts_xyz = sweep_points[:, :3]
            pts_global = transform_points(T_global_sweep, pts_xyz)  # [Ns,3]

            # then global -> keyframe lidar
            pts_lidar_ref = transform_points(T_lidar_global, pts_global)  # [Ns,3]

            sweep_points_ref = np.concatenate(
                [pts_lidar_ref, sweep_points[:, 3:4]], axis=1
            )  # [Ns,4]

            # BEV-range filter for sweep points
            if self.point_cloud_range is not None:
                x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
                mask = (
                    (pts_lidar_ref[:, 0] >= x_min)
                    & (pts_lidar_ref[:, 0] <= x_max)
                    & (pts_lidar_ref[:, 1] >= y_min)
                    & (pts_lidar_ref[:, 1] <= y_max)
                    & (pts_lidar_ref[:, 2] >= z_min)
                    & (pts_lidar_ref[:, 2] <= z_max)
                )
                sweep_points_ref = sweep_points_ref[mask]

            if sweep_points_ref.shape[0] > 0:
                all_points.append(sweep_points_ref)

        if len(all_points) > 1:
            lidar_points_agg = np.concatenate(all_points, axis=0)
        else:
            lidar_points_agg = lidar_points

        lidar_points_tensor = torch.from_numpy(lidar_points_agg)

        # convert to torch
        gt_boxes_lidar_t = torch.from_numpy(gt_boxes_lidar)   # [M,7]
        gt_labels_t = torch.from_numpy(gt_labels_np)          # [M]


        sample = {
            "sample_token": info["sample_token"],
            "scene_token": info["scene_token"],
            "timestamp": info["timestamp"],
            "lidar_points": lidar_points_tensor,           # aggregated multi-sweep
            "images": torch.from_numpy(imgs),
            "cams": cam_meta,
            "anns": anns,
            "gt_boxes": torch.from_numpy(gt_boxes_lidar), # lidar frame, in-range
            "gt_labels": torch.from_numpy(gt_labels_np),
            "lidar2ego": torch.from_numpy(lidar2ego_mat),
            "ego2global": torch.from_numpy(ego2global_mat),
        }


        return sample
