"""
render_bev_bbox.py
──────────────────
Renders a copy of bev_map.png with detection bounding boxes overlaid.
Uses PIL to draw directly on the existing image — no matplotlib imshow
alignment issues.

Coordinate system (from scene_utils.py):
  BEV axes: xlim(-50,50), ylim(-50,50), aspect=equal
  xlabel: '← left | right →'  → plot_x = right
  ylabel: '← back | forward →' → plot_y = forward
  ego frame: x=forward, y=left
  Therefore: plot_x = -y_ego, plot_y = x_ego
  Image y increases downward → flip plot_y
"""

import os, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BEV_RANGE_M     = 50.0
BEV_LABEL_RANGE = 35.0

BBOX_COLORS = {
    'car':                  (255, 100, 100),
    'truck':                (255, 165,   0),
    'bus':                  (255,  69,   0),
    'trailer':              (255, 140,   0),
    'motorcycle':           (  0, 191, 255),
    'bicycle':              ( 30, 144, 255),
    'pedestrian':           (  0, 255, 127),
    'traffic_cone':         (255, 255,   0),
    'barrier':              (128, 128, 128),
    'construction_vehicle': (139,  69,  19),
}
DEFAULT_COLOR = (200, 200, 200)
RADAR_COLOR   = (  0, 212, 255)


def find_plot_bounds(img_arr):
    """
    Detect the matplotlib axes plot area within the saved PNG.
    Plot background '#0a0e14' = RGB(10,14,20) — very dark.
    Returns (left, top, right, bottom) in pixels.
    """
    h, w = img_arr.shape[:2]
    dark = (img_arr[:,:,0] < 40) & (img_arr[:,:,1] < 40) & (img_arr[:,:,2] < 40)
    row_dark = dark.mean(axis=1) > 0.55
    col_dark = dark.mean(axis=0) > 0.55
    rows = np.where(row_dark)[0]
    cols = np.where(col_dark)[0]
    if len(rows) < 10 or len(cols) < 10:
        return 65, 60, w - 25, h - 70
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def ego_to_pixel(x_ego, y_ego, plot_l, plot_t, plot_r, plot_b):
    """Ego meters → image pixel coordinates."""
    plot_x = -y_ego
    plot_y =  x_ego
    nx = (plot_x + BEV_RANGE_M) / (2 * BEV_RANGE_M)
    ny = 1.0 - (plot_y + BEV_RANGE_M) / (2 * BEV_RANGE_M)
    px = plot_l + nx * (plot_r - plot_l)
    py = plot_t + ny * (plot_b - plot_t)
    return int(px), int(py)


def render_bev_with_bbox(bev_path, det_path, out_path, condition='B',
                          highlight_critical=True, max_range=BEV_RANGE_M):
    with open(det_path) as f:
        detections = json.load(f)

    img = Image.open(bev_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    img_arr = np.array(img)

    plot_l, plot_t, plot_r, plot_b = find_plot_bounds(img_arr)

    def to_px(x_e, y_e):
        return ego_to_pixel(x_e, y_e, plot_l, plot_t, plot_r, plot_b)

    for d in detections:
        dist   = d['distance_m']
        if dist > max_range:
            continue

        cls    = d['class']
        x_e, y_e = d['x'], d['y']
        length = d.get('length_m', 4.0)
        width  = d.get('width_m',  2.0)
        yaw    = d.get('yaw_rad',  0.0)
        vel    = d.get('velocity_ms', 0.0)
        vis    = d.get('visibility', 'fully_visible')
        bearing = d.get('bearing_deg', 90.0)

        color = BBOX_COLORS.get(cls, DEFAULT_COLOR)
        edge_color = RADAR_COLOR if (condition == 'D' and
                                      d.get('radar_quality') == 'reliable') else color

        vis_alpha = {'fully_visible': 220, 'mostly_visible': 170,
                     'partially_visible': 120, 'mostly_occluded': 70}.get(vis, 120)

        # Rotated corners in ego frame
        ca, sa = np.cos(yaw), np.sin(yaw)
        corners_ego = np.array([[ length/2, width/2], [-length/2, width/2],
                                 [-length/2,-width/2], [ length/2,-width/2]])
        rot    = np.array([[ca, -sa], [sa, ca]])
        c_ego  = (rot @ corners_ego.T).T + np.array([x_e, y_e])
        c_px   = [to_px(c[0], c[1]) for c in c_ego]

        # Fill + outline
        draw.polygon(c_px, fill=(*color, max(15, vis_alpha // 7)))
        for i in range(4):
            draw.line([c_px[i], c_px[(i+1)%4]],
                      fill=(*edge_color, vis_alpha), width=2)

        # Critical TTC ring (dashed approximation with arc)
        if highlight_critical and abs(bearing) < 45 and vel > 0.5:
            ttc = dist / vel
            if ttc < 5.0:
                cx, cy = to_px(x_e, y_e)
                r_m  = max(length, width) * 0.9
                r_px = max(8, int(r_m * (plot_r - plot_l) / (2 * BEV_RANGE_M)))
                draw.ellipse([cx-r_px, cy-r_px, cx+r_px, cy+r_px],
                             outline=(255, 0, 0, 200), width=2)

        # Label
        if dist <= BEV_LABEL_RANGE:
            cx, cy = to_px(x_e, y_e)
            vel_str = f"{vel:.1f}" if vel > 0.2 else "0"
            label = f"{cls[:3]} {dist:.0f}m {vel_str}m/s"
            try:
                font = ImageFont.load_default()
                tb   = draw.textbbox((cx, cy), label, font=font, anchor='mm')
                draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2],
                               fill=(10, 14, 20, 190))
                draw.text((cx, cy), label, fill=(255,255,255,230),
                          font=font, anchor='mm')
            except Exception:
                draw.text((cx-20, cy-6), label, fill=(255,255,255,200))

    # Composite overlay onto original
    result = Image.alpha_composite(img, overlay).convert('RGB')
    result.save(out_path)


def render_scene(scene_dir, out_dir=None, condition='B'):
    bev  = os.path.join(scene_dir, 'bev_map.png')
    det  = os.path.join(scene_dir, 'detections.json')
    if not os.path.exists(bev) or not os.path.exists(det):
        raise FileNotFoundError(f"Missing files in {scene_dir}")
    out  = out_dir or scene_dir
    path = os.path.join(out, 'bev_map_bbox.png')
    render_bev_with_bbox(bev, det, path, condition=condition)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bev_path',  required=True)
    ap.add_argument('--det_path',  required=True)
    ap.add_argument('--out_path',  required=True)
    ap.add_argument('--condition', default='B', choices=['B', 'D'])
    args = ap.parse_args()
    render_bev_with_bbox(args.bev_path, args.det_path, args.out_path,
                          condition=args.condition)
    print(f"Saved: {args.out_path}")
