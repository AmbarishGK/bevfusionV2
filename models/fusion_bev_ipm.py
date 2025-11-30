import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lidar_bev import SimpleLidarBEVEncoder
from models.camera_bev import CameraIPMEncoder


class FusionIPMDetector(nn.Module):
    """
    LiDAR + camera BEV detector with IPM-based camera BEV encoder.
    """

    def __init__(
        self,
        point_cloud_range,
        voxel_size,
        bev_size,
        num_classes: int = 10,
        bev_out_channels: int = 64,
        cam_bev_channels: int = 32,
    ) -> None:
        super().__init__()

        self.lidar_encoder = SimpleLidarBEVEncoder(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            bev_size=bev_size,
            in_channels=3,
            out_channels=bev_out_channels,
        )

        self.cam_encoder = CameraIPMEncoder(
            point_cloud_range=point_cloud_range,
            bev_size=bev_size,
            bev_out_channels=cam_bev_channels,
            pretrained_backbone=True,
        )

        # fuse at stride-4 resolution, keep stride-4
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(bev_out_channels + cam_bev_channels, bev_out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(bev_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_out_channels, bev_out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(bev_out_channels),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Conv2d(bev_out_channels, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(bev_out_channels, 8, kernel_size=1)

    def forward(
        self,
        lidar_points: torch.Tensor,   # [B, N, 4]
        images: torch.Tensor,         # [B, num_cams, 3, H, W]
        cams,
        lidar2ego: torch.Tensor,
        ego2global: torch.Tensor,
    ) -> dict:
        bev_lidar = self.lidar_encoder(lidar_points)       # [B, C_l, H4, W4]
        bev_cam_full = self.cam_encoder(images, cams, lidar2ego, ego2global)  # [B, C_c, H, W]

        _, _, H4, W4 = bev_lidar.shape
        bev_cam_down = F.interpolate(bev_cam_full, size=(H4, W4), mode="bilinear", align_corners=False)

        bev_fused = torch.cat([bev_lidar, bev_cam_down], dim=1)  # [B, C_l+C_c, H4, W4]
        bev_feat = self.fuse_conv(bev_fused)                     # [B, C_l, H4, W4]

        cls_logits = self.cls_head(bev_feat)
        box_reg = self.reg_head(bev_feat)

        return {
            "bev_feat": bev_feat,
            "cls_logits": cls_logits,
            "box_reg": box_reg,
        }
