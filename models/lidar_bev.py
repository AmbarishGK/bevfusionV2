import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLidarBEVEncoder(nn.Module):
    """
    Very simple LiDAR BEV encoder:
      - Rasterizes points to a BEV grid using mean height, mean intensity, and point count.
      - Runs a small CNN over the BEV feature map.
    """

    def __init__(
        self,
        point_cloud_range,
        voxel_size,
        bev_size,
        in_channels: int = 3,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = point_cloud_range
        self.vx, self.vy, self.vz = voxel_size
        # bev_size: [W, H]
        self.bev_w, self.bev_h = bev_size

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Slightly deeper CNN backbone (still overall stride 4)
        self.backbone = nn.Sequential(
            # Block 1 (no downsample)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2 (downsample x2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3 (no downsample)
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4 (downsample x2 again → total /4)
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Block 5 (no downsample, extra capacity)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lidar_points: [B, N, 4] in lidar frame (x, y, z, intensity), already range-filtered.

        Returns:
            bev_feat: [B, C, H_out, W_out] BEV feature map.
        """
        assert lidar_points.dim() == 3, f"expect [B,N,4], got {lidar_points.shape}"
        B, N, C = lidar_points.shape
        assert C >= 4, "lidar_points must have at least 4 channels (x,y,z,intensity)"

        device = lidar_points.device

        # Build initial BEV feature map per batch: [B, in_channels, H, W]
        bev = lidar_points.new_zeros((B, self.in_channels, self.bev_h, self.bev_w))

        for b in range(B):
            pts = lidar_points[b]  # [N,4]
            xs = pts[:, 0]
            ys = pts[:, 1]
            zs = pts[:, 2]
            intensities = pts[:, 3]

            # Compute BEV indices
            ix = ((xs - self.x_min) / self.vx).long()
            iy = ((ys - self.y_min) / self.vy).long()

            # Mask valid voxels inside [0, W) and [0, H)
            valid_mask = (
                (ix >= 0) & (ix < self.bev_w) &
                (iy >= 0) & (iy < self.bev_h)
            )
            if valid_mask.sum() == 0:
                continue

            ix = ix[valid_mask]
            iy = iy[valid_mask]
            zs = zs[valid_mask]
            intensities = intensities[valid_mask]

            lin_idx = iy * self.bev_w + ix  # [M]

            # Allocate accumulators
            num_cells = self.bev_h * self.bev_w
            sum_z = torch.zeros(num_cells, device=device, dtype=pts.dtype)
            sum_i = torch.zeros(num_cells, device=device, dtype=pts.dtype)
            count = torch.zeros(num_cells, device=device, dtype=pts.dtype)

            sum_z.index_add_(0, lin_idx, zs)
            sum_i.index_add_(0, lin_idx, intensities)
            count.index_add_(0, lin_idx, torch.ones_like(zs))

            # Avoid div-by-zero; cells with count=0 stay 0
            count_clamped = torch.clamp(count, min=1.0)

            mean_z = sum_z / count_clamped
            mean_i = sum_i / count_clamped

            # Reshape to [H,W]
            mean_z = mean_z.view(self.bev_h, self.bev_w)
            mean_i = mean_i.view(self.bev_h, self.bev_w)
            count_map = count.view(self.bev_h, self.bev_w)

            # Stack into BEV feature map: [3,H,W]
            bev[b, 0] = mean_z
            bev[b, 1] = mean_i
            bev[b, 2] = count_map

        # Pass through CNN backbone
        bev_feat = self.backbone(bev)
        return bev_feat


class SimpleLidarBEVDetector(nn.Module):
    """
    Minimal LiDAR-only BEV detector:
      - BEV encoder backbone
      - heatmap + residual box heads
    """

    def __init__(
        self,
        point_cloud_range,
        voxel_size,
        bev_size,
        num_classes: int = 10,
        bev_out_channels: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = SimpleLidarBEVEncoder(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            bev_size=bev_size,
            in_channels=3,
            out_channels=bev_out_channels,
        )

        self.cls_head = nn.Conv2d(bev_out_channels, num_classes, kernel_size=1)
        # 8-dim residuals: dx_res, dy_res, z, log(dx), log(dy), log(dz), sin(yaw), cos(yaw)
        self.reg_head = nn.Conv2d(bev_out_channels, 8, kernel_size=1)

    def forward(self, lidar_points: torch.Tensor) -> dict:
        bev_feat = self.encoder(lidar_points)
        cls_logits = self.cls_head(bev_feat)
        box_reg = self.reg_head(bev_feat)

        return {
            "bev_feat": bev_feat,
            "cls_logits": cls_logits,  # [B, C, H, W]
            "box_reg": box_reg,        # [B, 8, H, W]
        }

