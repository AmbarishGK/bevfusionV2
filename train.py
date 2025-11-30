import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from datasets.nuscenes_bevfusion import NuScenesBEVFusionDataset
# from models.lidar_bev import SimpleLidarBEVDetector
# from models.fusion_bev import LidarCameraBEVDetector
from models.fusion_bev_ipm import FusionIPMDetector


def gaussian2d(shape, sigma_x, sigma_y):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y = torch.arange(-m, m + 1)
    x = torch.arange(-n, n + 1)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.exp(- (xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))


def draw_gaussian(heatmap, center, radius, k=1.0):
    """Draw a 2D Gaussian on heatmap at 'center' with given integer radius."""
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[-2:]

    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    gaussian = gaussian2d((diameter, diameter), sigma, sigma).to(heatmap.device)

    left, right = min(x, radius), min(width - x - 1, radius)
    top, bottom = min(y, radius), min(height - y - 1, radius)

    if right < 0 or left < 0 or top < 0 or bottom < 0:
        return heatmap

    masked_h = slice(y - top, y + bottom + 1)
    masked_w = slice(x - left, x + right + 1)
    g_y = slice(radius - top, radius + bottom + 1)
    g_x = slice(radius - left, radius + right + 1)

    heatmap[..., masked_h, masked_w] = torch.maximum(
        heatmap[..., masked_h, masked_w],
        gaussian[g_y, g_x] * k,
    )
    return heatmap


def gaussian_radius(box_hw, min_overlap=0.5):
    """
    Rough CenterNet-style radius given box size in feature map cells.
    box_hw: (h, w) in output feature map pixels.
    """
    h, w = box_hw
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return torch.min(torch.min(r1, r2), r3)

def focal_loss_heatmap(pred, gt, alpha=2.0, beta=4.0):
    pred_sigmoid = pred.sigmoid().clamp(min=1e-6, max=1 - 1e-6)

    pos_mask = gt.eq(1.0).float()
    neg_mask = gt.lt(1.0).float()

    pos_loss = - (1 - pred_sigmoid) ** alpha * torch.log(pred_sigmoid) * pos_mask
    neg_loss = - (pred_sigmoid ** alpha) * ((1 - gt) ** beta) * torch.log(1 - pred_sigmoid) * neg_mask

    loss = (pos_loss + neg_loss).mean()
    return loss


def build_dataloader(cfg: Config, split: str = "train"):
    data_cfg = cfg["data"]
    geom_cfg = cfg["geometry"]
    aug_cfg = cfg.get("augment", {})

    info_path = data_cfg["info_train"] if split == "train" else data_cfg["info_val"]

    dataset = NuScenesBEVFusionDataset(
        info_path=info_path,
        image_size=tuple(geom_cfg["image_size"]),
        image_mean=aug_cfg.get("image_mean"),
        image_std=aug_cfg.get("image_std"),
        point_cloud_range=geom_cfg.get("point_cloud_range"),
    )

    # keep batch_size = 1 for now to avoid custom collate_fn
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True,
    )
    return loader


def assign_heatmap_and_reg_targets(
    gt_boxes: torch.Tensor,   # [B, M, 7]
    gt_labels: torch.Tensor,  # [B, M]
    point_cloud_range,
    voxel_size,
    bev_size,
    downsample: int,
    num_classes: int,
    device: torch.device,
):
    """
    CenterPoint-like:
      - heatmap per class with Gaussian at box centers
      - residual-encoded regression targets at center cell
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size
    bev_w, bev_h = bev_size

    out_w = bev_w // downsample
    out_h = bev_h // downsample

    B, M, _ = gt_boxes.shape

    heatmaps = torch.zeros(
        (B, num_classes, out_h, out_w), device=device, dtype=torch.float32
    )
    reg_targets = torch.zeros(
        (B, 8, out_h, out_w), device=device, dtype=torch.float32
    )
    reg_mask = torch.zeros(
        (B, 1, out_h, out_w), device=device, dtype=torch.float32
    )

    for b in range(B):
        boxes = gt_boxes[b]   # [M,7]
        labels = gt_labels[b] # [M]

        for m in range(boxes.shape[0]):
            cls_id = int(labels[m].item())
            if cls_id < 0 or cls_id >= num_classes:
                continue

            x, y, z, dx, dy, dz, yaw = boxes[m]

            if dx <= 0 or dy <= 0 or dz <= 0:
                continue

            # full-res indices
            ix_full = (x - x_min) / vx
            iy_full = (y - y_min) / vy

            # output indices (float)
            ix = ix_full / downsample
            iy = iy_full / downsample

            ix_int = int(ix)
            iy_int = int(iy)

            if ix_int < 0 or ix_int >= out_w or iy_int < 0 or iy_int >= out_h:
                continue

            # box size in output feature pixels
            w_f = dx / (vx * downsample)
            h_f = dy / (vy * downsample)
            wh = torch.tensor([h_f, w_f], device=device)
            radius = gaussian_radius(wh, min_overlap=0.5)
            radius = max(0, int(radius.item()))
            radius = max(radius, 1)

            # draw Gaussian on heatmap for this class
            heatmaps[b, cls_id] = draw_gaussian(
                heatmaps[b, cls_id], center=(ix, iy), radius=radius
            )

            # regression target only at integer center cell
            # compute cell center in metric coordinates
            x_cell = x_min + (ix_int + 0.5) * vx * downsample
            y_cell = y_min + (iy_int + 0.5) * vy * downsample

            dx_res = (x - x_cell) / (vx * downsample)
            dy_res = (y - y_cell) / (vy * downsample)

            # log dims (avoid log 0)
            eps = 1e-2
            log_dx = torch.log(torch.clamp(dx, min=eps))
            log_dy = torch.log(torch.clamp(dy, min=eps))
            log_dz = torch.log(torch.clamp(dz, min=eps))

            yaw_sin = torch.sin(yaw)
            yaw_cos = torch.cos(yaw)

            reg_targets[b, 0, iy_int, ix_int] = dx_res
            reg_targets[b, 1, iy_int, ix_int] = dy_res
            reg_targets[b, 2, iy_int, ix_int] = z
            reg_targets[b, 3, iy_int, ix_int] = log_dx
            reg_targets[b, 4, iy_int, ix_int] = log_dy
            reg_targets[b, 5, iy_int, ix_int] = log_dz
            reg_targets[b, 6, iy_int, ix_int] = yaw_sin
            reg_targets[b, 7, iy_int, ix_int] = yaw_cos

            reg_mask[b, 0, iy_int, ix_int] = 1.0

    return heatmaps, reg_targets, reg_mask



def compute_losses(
    out: dict,
    heatmaps: torch.Tensor,
    reg_targets: torch.Tensor,
    reg_mask: torch.Tensor,
):
    cls_logits = out["cls_logits"]  # [B, C, H, W]
    box_reg = out["box_reg"]        # [B, 8, H, W]

    loss_cls = focal_loss_heatmap(cls_logits, heatmaps)

    # regression: L1 only on positive cells
    mask = reg_mask.expand_as(box_reg)  # [B,8,H,W]
    num_pos = reg_mask.sum().clamp(min=1.0)

    loss_reg = F.l1_loss(
        box_reg * mask, reg_targets * mask, reduction="sum"
    ) / num_pos

    loss = loss_cls + loss_reg
    return loss, loss_cls, loss_reg

def decode_predictions(
    out: dict,
    cfg: Config,
    score_thresh: float = 0.3,
    topk: int = 50,
):
    """
    Decode heatmap + residual outputs into 3D boxes per sample.

    Returns:
        preds: list length B; each element is a dict with:
          {
            "boxes": [K,7] (x,y,z,dx,dy,dz,yaw),
            "labels": [K],
            "scores": [K]
          }
    """
    geom_cfg = cfg["geometry"]
    data_cfg = cfg["data"]

    point_cloud_range = geom_cfg["point_cloud_range"]
    voxel_size = geom_cfg["voxel_size"]
    bev_size = geom_cfg["bev_size"]
    num_classes = len(data_cfg["classes"])

    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size
    bev_w, bev_h = bev_size

    downsample = 4  # must match BEV backbone

    cls_logits = out["cls_logits"]  # [B,C,H,W]
    box_reg = out["box_reg"]        # [B,8,H,W]

    B, C, H, W = cls_logits.shape
    assert C == num_classes

    heat = cls_logits.sigmoid()  # [B,C,H,W]

    preds = []

    for b in range(B):
        boxes_b = []
        labels_b = []
        scores_b = []

        for cls_id in range(num_classes):
            hm = heat[b, cls_id]  # [H,W]
            hm_flat = hm.view(-1)  # [H*W]

            vals, inds = torch.topk(hm_flat, k=min(topk, hm_flat.numel()))
            mask = vals > score_thresh
            vals = vals[mask]
            inds = inds[mask]

            if vals.numel() == 0:
                continue

            ys = (inds // W).long()
            xs = (inds % W).long()

            reg = box_reg[b, :, ys, xs]  # [8,K]

            dx_res = reg[0]
            dy_res = reg[1]
            z = reg[2]
            log_dx = reg[3]
            log_dy = reg[4]
            log_dz = reg[5]
            yaw_sin = reg[6]
            yaw_cos = reg[7]

            # cell centers in metric coords
            x_cell = x_min + (xs.float() + 0.5) * vx * downsample
            y_cell = y_min + (ys.float() + 0.5) * vy * downsample

            x = x_cell + dx_res * (vx * downsample)
            y = y_cell + dy_res * (vy * downsample)

            dx = torch.exp(log_dx)
            dy = torch.exp(log_dy)
            dz = torch.exp(log_dz)

            yaw = torch.atan2(yaw_sin, yaw_cos)

            boxes = torch.stack([x, y, z, dx, dy, dz, yaw], dim=-1)  # [K,7]
            labels = torch.full(
                (boxes.shape[0],), cls_id, dtype=torch.long, device=boxes.device
            )
            scores = vals

            boxes_b.append(boxes)
            labels_b.append(labels)
            scores_b.append(scores)

        if len(boxes_b) == 0:
            preds.append(
                {
                    "boxes": torch.zeros((0, 7)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                    "scores": torch.zeros((0,)),
                }
            )
        else:
            boxes_b = torch.cat(boxes_b, dim=0)
            labels_b = torch.cat(labels_b, dim=0)
            scores_b = torch.cat(scores_b, dim=0)

            # sort by score desc
            order = torch.argsort(scores_b, descending=True)
            boxes_b = boxes_b[order]
            labels_b = labels_b[order]
            scores_b = scores_b[order]

            preds.append(
                {
                    "boxes": boxes_b.cpu(),
                    "labels": labels_b.cpu(),
                    "scores": scores_b.cpu(),
                }
            )

    return preds

def boxes_bev_aabb(boxes: torch.Tensor):
    """
    Convert [N,7] boxes (x,y,z,dx,dy,dz,yaw) to axis-aligned BEV boxes:
      [N,4] = (x_min, y_min, x_max, y_max)
    ignoring yaw. This is a simplification.
    """
    x, y, _, dx, dy, _, _ = boxes.unbind(dim=-1)
    x_min = x - dx / 2.0
    x_max = x + dx / 2.0
    y_min = y - dy / 2.0
    y_max = y + dy / 2.0
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def iou_bev_aabb(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Axis-aligned BEV IoU between two sets of boxes.
    boxes1: [N,4], boxes2: [M,4]
    Returns IoU: [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))

    N = boxes1.shape[0]
    M = boxes2.shape[0]

    b1 = boxes1.unsqueeze(1).expand(N, M, 4)
    b2 = boxes2.unsqueeze(0).expand(N, M, 4)

    x_min = torch.maximum(b1[..., 0], b2[..., 0])
    y_min = torch.maximum(b1[..., 1], b2[..., 1])
    x_max = torch.minimum(b1[..., 2], b2[..., 2])
    y_max = torch.minimum(b1[..., 3], b2[..., 3])

    inter_w = torch.clamp(x_max - x_min, min=0.0)
    inter_h = torch.clamp(y_max - y_min, min=0.0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    area1 = area1.unsqueeze(1).expand_as(inter_area)
    area2 = area2.unsqueeze(0).expand_as(inter_area)

    union = area1 + area2 - inter_area + 1e-6
    iou = inter_area / union
    return iou

def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.3,
    topk: int = 50,
):
    """
    Simple per-class precision/recall at IoU=0.5 on the val set.
    NOT official nuScenes NDS/mAP, but useful for relative comparisons.
    """
    data_cfg = cfg["data"]
    num_classes = len(data_cfg["classes"])

    model.eval()
    stats = []
    # for each class: [tp, fp, total_gt]
    tp = torch.zeros(num_classes, dtype=torch.float32)
    fp = torch.zeros(num_classes, dtype=torch.float32)
    total_gt = torch.zeros(num_classes, dtype=torch.float32)

    with torch.no_grad():
        for batch in loader:
            lidar_points = batch["lidar_points"].to(device)  # [B,N,4]
            images = batch["images"].to(device)              # [B,6,3,H,W]

            raw_cams = batch["cams"]
            lidar2ego_b = batch["lidar2ego"].to(device)      # [B,4,4]
            ego2global_b = batch["ego2global"].to(device)    # [B,4,4]

            B = lidar_points.shape[0]  # should be 1

            cams_list = []
            for b in range(B):
                cam_meta_b = {}
                for cam_name, meta in raw_cams.items():
                    if meta is None:
                        cam_meta_b[cam_name] = None
                        continue
                    cam_meta_b[cam_name] = {
                        "sensor2ego": meta["sensor2ego"][b].cpu().numpy(),
                        "ego2global": meta["ego2global"][b].cpu().numpy(),
                        "intrinsics": meta["intrinsics"][b].cpu().numpy(),
                    }
                cams_list.append(cam_meta_b)

            gt_boxes = batch["gt_boxes"][0]   # [M,7], B=1
            gt_labels = batch["gt_labels"][0] # [M]

            out = model(
                lidar_points,
                images,
                cams_list,
                lidar2ego_b,
                ego2global_b,
            )

            preds = decode_predictions(
                out, cfg, score_thresh=score_thresh, topk=topk
            )[0]  # first (and only) batch element

            pred_boxes = preds["boxes"]
            pred_labels = preds["labels"]
            pred_scores = preds["scores"]
            
            # update total_gt per class
            for c in range(num_classes):
                total_gt[c] += (gt_labels == c).sum().item()

            if pred_boxes.shape[0] == 0:
                continue

            # axis-aligned BEV boxes for IoU
            gt_bev = boxes_bev_aabb(gt_boxes)
            pred_bev = boxes_bev_aabb(pred_boxes)

            ious = iou_bev_aabb(pred_bev, gt_bev)  # [num_pred, num_gt]
            gt_used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

            # assign preds to gts greedily in score order
            for i in range(pred_boxes.shape[0]):
                c = int(pred_labels[i].item())
                best_iou = 0.0
                best_j = -1

                for j in range(gt_boxes.shape[0]):
                    if gt_used[j]:
                        continue
                    if int(gt_labels[j].item()) != c:
                        continue
                    iou_ij = float(ious[i, j].item())
                    if iou_ij > best_iou:
                        best_iou = iou_ij
                        best_j = j

                if best_j >= 0 and best_iou >= iou_thresh:
                    tp[c] += 1.0
                    gt_used[best_j] = True
                else:
                    fp[c] += 1.0

    # compute precision/recall
    precision = torch.zeros_like(tp)
    recall = torch.zeros_like(tp)

    for c in range(num_classes):
        denom_p = tp[c] + fp[c]
        denom_r = total_gt[c]
        if denom_p > 0:
            precision[c] = tp[c] / denom_p
        if denom_r > 0:
            recall[c] = tp[c] / denom_r

    # macro averages
    valid_classes = total_gt > 0
    if valid_classes.any():
        mean_prec = precision[valid_classes].mean().item()
        mean_rec = recall[valid_classes].mean().item()
    else:
        mean_prec = 0.0
        mean_rec = 0.0

    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "total_gt_per_class": total_gt,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    epoch: int,
):
    geom_cfg = cfg["geometry"]
    data_cfg = cfg["data"]

    point_cloud_range = geom_cfg["point_cloud_range"]
    voxel_size = geom_cfg["voxel_size"]
    bev_size = geom_cfg["bev_size"]
    num_classes = len(data_cfg["classes"])

    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0

    # backbone downsample factor (we know we used 2x + 2x)
    downsample = 4

    for it, batch in enumerate(loader):
        lidar_points = batch["lidar_points"].to(device)  # [1, N, 4]
        images = batch["images"].to(device)              # [1, 6, 3, H, W]

        gt_boxes = batch["gt_boxes"].to(device)          # [1, M, 7]
        gt_labels = batch["gt_labels"].to(device)        # [1, M]

        # ensure shapes [B,M,7] / [B,M]
        if gt_boxes.dim() == 2:
            gt_boxes = gt_boxes.unsqueeze(0)
        if gt_labels.dim() == 1:
            gt_labels = gt_labels.unsqueeze(0)

        cams = batch["cams"]                         # list of dicts after DataLoader; batch_size=1
        raw_cams = batch["cams"]                     # dict(cam_name -> meta_dict_of_tensors)
        lidar2ego_b = batch["lidar2ego"].to(device)  # [B,4,4]
        ego2global_b = batch["ego2global"].to(device)

        B = lidar_points.shape[0]  # should be 1, but keep general

        cams_list = []
        for b in range(B):
            cam_meta_b = {}
            for cam_name, meta in raw_cams.items():
                if meta is None:
                    cam_meta_b[cam_name] = None
                    continue
                cam_meta_b[cam_name] = {
                    # each is tensor [B,4,4] or [B,3,3] -> pick slice b and convert to numpy
                    "sensor2ego": meta["sensor2ego"][b].cpu().numpy(),
                    "ego2global": meta["ego2global"][b].cpu().numpy(),
                    "intrinsics": meta["intrinsics"][b].cpu().numpy(),
                }
            cams_list.append(cam_meta_b)

        out = model(
            lidar_points,
            images,
            cams_list,       # <-- pass list[dict]
            lidar2ego_b,
            ego2global_b,
        )

        heatmaps, reg_targets, reg_mask = assign_heatmap_and_reg_targets(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            bev_size=bev_size,
            downsample=downsample,
            num_classes=num_classes,
            device=device,
        )

        loss, loss_cls, loss_reg = compute_losses(
            out, heatmaps, reg_targets, reg_mask
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_reg += loss_reg.item()

        if (it + 1) % 10 == 0:
            avg_loss = total_loss / (it + 1)
            avg_cls = total_cls / (it + 1)
            avg_reg = total_reg / (it + 1)
            print(
                f"Epoch {epoch} Iter {it+1}/{len(loader)} "
                f"loss={avg_loss:.4f} cls={avg_cls:.4f} reg={avg_reg:.4f}"
            )


def main():
    cfg = Config.from_yaml("config/nuscenes_mini.yaml")
    data_cfg = cfg["data"]
    geom_cfg = cfg["geometry"]

    point_cloud_range = geom_cfg["point_cloud_range"]
    voxel_size = geom_cfg["voxel_size"]
    bev_size = geom_cfg["bev_size"]
    num_classes = len(data_cfg["classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = SimpleLidarBEVDetector(
    #     point_cloud_range=point_cloud_range,
    #     voxel_size=voxel_size,
    #     bev_size=bev_size,
    #     num_classes=num_classes,
    #     bev_out_channels=64,
    # ).to(device)

    # model = LidarCameraBEVDetector(
    #     point_cloud_range=point_cloud_range,
    #     voxel_size=voxel_size,
    #     bev_size=bev_size,
    #     num_classes=num_classes,
    #     bev_out_channels=64,
    #     cam_out_channels=32,
    # ).to(device)
    model = FusionIPMDetector(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        bev_size=bev_size,
        num_classes=num_classes,
        bev_out_channels=64,
        cam_bev_channels=32,
    ).to(device)

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    # simple Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = cfg.get("train", {}).get("epochs", 2)

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, cfg, device, epoch)
        # run eval at the end of each epoch
        metrics = evaluate_dataset(
            model, val_loader, cfg, device,
            iou_thresh=0.3, score_thresh=0.2, topk=50
        )
        
        print(
            f"[VAL] Epoch {epoch}: "
            f"mPrec@0.3={metrics['mean_precision']:.3f}, "
            f"mRec@0.3={metrics['mean_recall']:.3f}"
        )

    # save a small checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/lidar_bev_mini.pth")
    print("Saved checkpoint to checkpoints/lidar_bev_mini.pth")


if __name__ == "__main__":
    main()
