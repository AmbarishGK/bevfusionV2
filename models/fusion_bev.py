import torch
import torch.nn as nn
from torchvision import models as tv_models

from models.lidar_bev import SimpleLidarBEVEncoder


class SimpleCameraBackbone(nn.Module):
    """
    Simple camera backbone using ResNet-18.
    - Takes 6-camera images: [B, 6, 3, H, W]
    - Produces a global camera context vector per batch: [B, cam_out_channels]
    """

    def __init__(self, cam_out_channels: int = 32) -> None:
        super().__init__()
        # ResNet-18 up to conv5 (no avgpool, no fc)
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # [B,512,H',W']
        self.proj = nn.Conv2d(512, cam_out_channels, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 6, 3, H, W]

        Returns:
            cam_ctx: [B, cam_out_channels] global camera context
        """
        B, num_cams, C, H, W = images.shape
        assert num_cams == 6, f"expected 6 cameras, got {num_cams}"

        x = images.view(B * num_cams, C, H, W)  # [B*6,3,H,W]
        feat = self.encoder(x)                  # [B*6,512,H',W']
        feat = self.proj(feat)                  # [B*6,cam_out,H',W']

        # Global average pool per camera
        feat = feat.mean(dim=[2, 3])            # [B*6,cam_out]
        feat = feat.view(B, num_cams, -1)       # [B,6,cam_out]

        # Average across cameras → one context vector per batch
        cam_ctx = feat.mean(dim=1)              # [B,cam_out]
        return cam_ctx


class LidarCameraBEVDetector(nn.Module):
    """
    Simple fusion detector:
      - LiDAR BEV encoder
      - Camera backbone -> global context
      - Fuse camera context into BEV by tiling & conv
      - Same heads: heatmap + residual box regression
    """

    def __init__(
        self,
        point_cloud_range,
        voxel_size,
        bev_size,
        num_classes: int = 10,
        bev_out_channels: int = 64,
        cam_out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.lidar_encoder = SimpleLidarBEVEncoder(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            bev_size=bev_size,
            in_channels=3,
            out_channels=bev_out_channels,
        )

        self.cam_backbone = SimpleCameraBackbone(cam_out_channels=cam_out_channels)

        # fuse [bev_out_channels + cam_out_channels] -> bev_out_channels
        self.fusion_conv = nn.Conv2d(
            bev_out_channels + cam_out_channels, bev_out_channels, kernel_size=1
        )

        self.cls_head = nn.Conv2d(bev_out_channels, num_classes, kernel_size=1)
        # 8-dim residuals: dx_res, dy_res, z, log(dx), log(dy), log(dz), sin(yaw), cos(yaw)
        self.reg_head = nn.Conv2d(bev_out_channels, 8, kernel_size=1)

    def forward(self, lidar_points: torch.Tensor, images: torch.Tensor) -> dict:
        """
        Args:
            lidar_points: [B, N, 4]
            images: [B, 6, 3, H, W]

        Returns:
            dict with:
              - bev_feat: [B, C, H', W']
              - cls_logits: [B, num_classes, H', W']
              - box_reg: [B, 8, H', W']
        """
        bev_lidar = self.lidar_encoder(lidar_points)  # [B, C_l, H', W']
        B, C_l, H_out, W_out = bev_lidar.shape

        cam_ctx = self.cam_backbone(images)          # [B, C_cam]
        C_cam = cam_ctx.shape[1]

        # Tile camera context over BEV grid
        cam_ctx_map = cam_ctx.view(B, C_cam, 1, 1).expand(B, C_cam, H_out, W_out)

        fused_bev = torch.cat([bev_lidar, cam_ctx_map], dim=1)  # [B, C_l+C_cam, H', W']
        fused_bev = self.fusion_conv(fused_bev)                 # [B, C_l, H', W']

        cls_logits = self.cls_head(fused_bev)
        box_reg = self.reg_head(fused_bev)

        return {
            "bev_feat": fused_bev,
            "cls_logits": cls_logits,
            "box_reg": box_reg,
        }
