import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CameraIPMEncoder(nn.Module):
    """
    Camera -> BEV encoder using simple inverse perspective mapping:
      - ResNet18 backbone (per camera)
      - Project rays through pixel centers to ground plane z=0 in lidar frame
      - Scatter-add features into BEV grid
    """

    def __init__(
        self,
        point_cloud_range: List[float],
        bev_size: List[int],
        bev_out_channels: int = 64,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = point_cloud_range
        self.bev_w, self.bev_h = bev_size

        # ResNet18 backbone, keep spatial features
        backbone = resnet18(weights="IMAGENET1K_V1" if pretrained_backbone else None)
        # remove avgpool + fc
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.backbone_out_channels = 512  # ResNet18 layer4 output

        # project backbone features into BEV channels
        self.bev_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, bev_out_channels, kernel_size=1),
            nn.BatchNorm2d(bev_out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        images: torch.Tensor,                 # [B, num_cams, 3, H, W]
        cams: List[Dict[str, dict]],          # list of length B, each is dict(cam_name -> meta)
        lidar2ego: torch.Tensor,             # [B, 4, 4]
        ego2global: torch.Tensor,            # [B, 4, 4]
    ) -> torch.Tensor:
        B, num_cams, _, H, W = images.shape
        device = images.device

        # flatten cameras: [B*num_cams, 3, H, W]
        imgs_flat = images.view(B * num_cams, 3, H, W)

        feats_flat = self.backbone(imgs_flat)  # [B*num_cams, C, Hf, Wf]
        feats_flat = self.bev_head(feats_flat) # [B*num_cams, Cb, Hf, Wf]
        _, Cb, Hf, Wf = feats_flat.shape

        feats = feats_flat.view(B, num_cams, Cb, Hf, Wf)

        # infer stride between image and feature map
        stride_y = H / float(Hf)
        stride_x = W / float(Wf)

        bev = images.new_zeros((B, Cb, self.bev_h, self.bev_w))  # BEV feature map

        # precompute BEV scaling
        dx = (self.x_max - self.x_min) / float(self.bev_w)
        dy = (self.y_max - self.y_min) / float(self.bev_h)

        for b in range(B):
            # lidar <- global
            lidar2ego_b = lidar2ego[b].cpu().numpy()
            ego2global_b = ego2global[b].cpu().numpy()

            # invert to get:
            #   global2ego = (ego2global)^-1
            #   ego2lidar  = (lidar2ego)^-1
            ego2global_mat = ego2global_b
            lidar2ego_mat = lidar2ego_b

            ego2global_inv = _invert_4x4(ego2global_mat)
            lidar2ego_inv = _invert_4x4(lidar2ego_mat)

            # lidar <- global
            T_lidar_global = lidar2ego_inv @ ego2global_inv

            cams_b = cams[b]  # dict(cam_name -> meta)
            cam_names = list(cams_b.keys())
            cam_names.sort()  # ensure deterministic order

            for cam_idx, cam_name in enumerate(cam_names):
                meta = cams_b[cam_name]
                if meta is None:
                    continue

                sensor2ego = meta["sensor2ego"]   # 4x4, ego <- cam
                ego2global_cam = meta["ego2global"]  # 4x4, global <- ego
                K = meta["intrinsics"]           # 3x3

                # lidar <- cam: T_lidar_cam = T_lidar_global @ T_global_cam
                # T_global_cam = ego2global_cam @ sensor2ego
                T_global_cam = ego2global_cam @ sensor2ego
                T_lidar_cam = T_lidar_global @ T_global_cam

                R_lidar_cam = torch.from_numpy(T_lidar_cam[:3, :3]).to(device=device, dtype=torch.float32)
                t_lidar_cam = torch.from_numpy(T_lidar_cam[:3, 3]).to(device=device, dtype=torch.float32)

                K_torch = torch.from_numpy(K).to(device=device, dtype=torch.float32)

                feat = feats[b, cam_idx]  # [Cb, Hf, Wf]

                # pixel coords in feature map
                ys, xs = torch.meshgrid(
                    torch.arange(Hf, device=device),
                    torch.arange(Wf, device=device),
                    indexing="ij",
                )  # [Hf,Wf]

                # map to original image pixel centers
                u = (xs.float() + 0.5) * stride_x
                v = (ys.float() + 0.5) * stride_y

                # backproject rays in camera frame: direction d_cam ~ K^-1 [u,v,1]
                ones = torch.ones_like(u)
                pix = torch.stack([u, v, ones], dim=-1)  # [Hf,Wf,3]
                K_inv = torch.inverse(K_torch)
                d_cam = pix @ K_inv.T                 # [Hf,Wf,3]
                d_cam = d_cam / torch.norm(d_cam, dim=-1, keepdim=True).clamp(min=1e-6)

                # transform directions to lidar frame
                d_cam_flat = d_cam.view(-1, 3).T      # [3,N]
                d_lidar_flat = (R_lidar_cam @ d_cam_flat).T  # [N,3]
                d_lidar = d_lidar_flat.view(Hf, Wf, 3)

                # ray origin in lidar frame (same for all pixels)
                o_lidar = t_lidar_cam  # [3]

                # intersect ray with ground plane z=0 in lidar frame:
                #   o_z + t * d_z = 0 => t = -o_z / d_z
                o_z = o_lidar[2]
                d_z = d_lidar[..., 2]  # [Hf,Wf]
                # avoid division by zero
                mask_valid = d_z.abs() > 1e-3

                t = torch.zeros_like(d_z)
                t[mask_valid] = -o_z / d_z[mask_valid]

                # only keep points in front of camera (t > 0)
                mask_valid = mask_valid & (t > 0.0)

                # coordinates in lidar frame
                o_xy = o_lidar[:2]  # [2]
                d_xy = d_lidar[..., :2]  # [Hf,Wf,2]
                pts_xy = o_xy.view(1, 1, 2) + d_xy * t.unsqueeze(-1)  # [Hf,Wf,2]

                # valid mask for BEV range
                x = pts_xy[..., 0]
                y = pts_xy[..., 1]

                mask_valid = (
                    mask_valid &
                    (x >= self.x_min) & (x <= self.x_max) &
                    (y >= self.y_min) & (y <= self.y_max)
                )

                if mask_valid.sum() == 0:
                    continue

                x_valid = x[mask_valid]
                y_valid = y[mask_valid]

                # BEV indices
                ix = ((x_valid - self.x_min) / dx).long()
                iy = ((y_valid - self.y_min) / dy).long()

                # clamp just in case
                ix = ix.clamp(0, self.bev_w - 1)
                iy = iy.clamp(0, self.bev_h - 1)

                # gather features at valid pixels
                feat_valid = feat[:, mask_valid]  # [Cb, Nv]
                # flatten BEV indices
                lin_idx = iy * self.bev_w + ix    # [Nv]

                # scatter-add into BEV
                bev_b = bev[b].view(Cb, -1)      # [Cb, bev_h*bev_w]
                bev_b.index_add_(1, lin_idx, feat_valid)
                bev[b] = bev_b.view(Cb, self.bev_h, self.bev_w)

        return bev


def _invert_4x4(T):
    import numpy as np
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv
