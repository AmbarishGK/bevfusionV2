from torch.utils.data import DataLoader
import torch

from config import Config
from datasets.nuscenes_bevfusion import NuScenesBEVFusionDataset
from models.lidar_bev import SimpleLidarBEVDetector


def main():
    cfg = Config.from_yaml("config/nuscenes_mini.yaml")
    data_cfg = cfg["data"]
    geom_cfg = cfg["geometry"]
    aug_cfg = cfg.get("augment", {})

    dataset = NuScenesBEVFusionDataset(
        info_path=data_cfg["info_train"],
        image_size=tuple(geom_cfg["image_size"]),
        image_mean=aug_cfg.get("image_mean"),
        image_std=aug_cfg.get("image_std"),
        point_cloud_range=geom_cfg.get("point_cloud_range"),
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    lidar_points = batch["lidar_points"]  # [1, N, 4]

    print("Input lidar_points:", lidar_points.shape)

    point_cloud_range = geom_cfg["point_cloud_range"]
    voxel_size = geom_cfg["voxel_size"]
    bev_size = geom_cfg["bev_size"]
    num_classes = len(data_cfg["classes"])

    model = SimpleLidarBEVDetector(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        bev_size=bev_size,
        num_classes=num_classes,
        bev_out_channels=64,
    )

    model.eval()
    with torch.no_grad():
        out = model(lidar_points)

    print("bev_feat:", out["bev_feat"].shape)
    print("cls_logits:", out["cls_logits"].shape)
    print("box_reg:", out["box_reg"].shape)


if __name__ == "__main__":
    main()
