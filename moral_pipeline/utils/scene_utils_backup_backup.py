"""
utils/scene_utils.py
====================
Shared utility functions for the MoRAL pipeline.

detections.json schema (same for GT and BEVFusion):
----------------------------------------------------
{
  "class":          str,    # car/truck/pedestrian etc
  "x":              float,  # ego-relative x (meters, forward)
  "y":              float,  # ego-relative y (meters, left)
  "z":              float,  # ego-relative z (meters, up)
  "distance_m":     float,  # sqrt(x+y)
  "bearing_deg":    float,  # angle from ego forward axis
  "confidence":     float,  # 0.0-1.0 (GT=1.0, BEVFusion=real score)
  "width_m":        float,
  "length_m":       float,
  "height_m":       float,
  "yaw_rad":        float,  # heading angle — NOW IN BOTH GT AND BEVFUSION
  "source":         str,    # "ground_truth" | "bevfusion"

  # GT-only bonus fields (not in BEVFusion, ignored by Cosmos):
  "velocity_ms":    float,
  "vx":             float,
  "vy":             float,
  "num_lidar_pts":  int,
  "num_radar_pts":  int,
  "visibility":     str,
  "ann_token":      str,
}
"""

import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion

CLASS_MAP = {
    'vehicle.car':                        'car',
    'vehicle.truck':                      'truck',
    'vehicle.bus.bendy':                  'bus',
    'vehicle.bus.rigid':                  'bus',
    'vehicle.trailer':                    'trailer',
    'vehicle.motorcycle':                 'motorcycle',
    'vehicle.bicycle':                    'bicycle',
    'vehicle.construction':               'construction_vehicle',
    'human.pedestrian.adult':             'pedestrian',
    'human.pedestrian.child':             'pedestrian',
    'human.pedestrian.wheelchair':        'pedestrian',
    'human.pedestrian.stroller':          'pedestrian',
    'human.pedestrian.personal_mobility': 'pedestrian',
    'movable_object.barrier':             'barrier',
    'movable_object.trafficcone':         'traffic_cone',
    'movable_object.pushable_pullable':   'barrier',
    'movable_object.debris':              'barrier',
    'static_object.bicycle_rack':         'barrier',
}

CLASS_LIST = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

VISIBILITY_MAP = {
    '1': 'mostly_occluded',
    '2': 'partially_visible',
    '3': 'mostly_visible',
    '4': 'fully_visible',
}

def get_ego_pose(nusc, sample):
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    return nusc.get('ego_pose', lidar_sd['ego_pose_token'])

def get_ego_speed(nusc, sample):
    try:
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        if lidar_sd['next'] != '':
            next_sd   = nusc.get('sample_data', lidar_sd['next'])
            next_pose = nusc.get('ego_pose', next_sd['ego_pose_token'])
            dt = (next_pose['timestamp'] - ego_pose['timestamp']) / 1e6
            dx = next_pose['translation'][0] - ego_pose['translation'][0]
            dy = next_pose['translation'][1] - ego_pose['translation'][1]
            if dt > 0:
                return round(float(np.sqrt(dx**2 + dy**2) / dt), 2)
    except Exception:
        pass
    return None

def get_lidar_points(nusc, sample, dataroot):
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pcd_path = os.path.join(dataroot, lidar_sd['filename'])
    return np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)

def get_detections_from_gt(nusc, sample, ego_pose, min_lidar_pts=1):
    ego_x   = ego_pose['translation'][0]
    ego_y   = ego_pose['translation'][1]
    ego_z   = ego_pose['translation'][2]
    ego_q   = Quaternion(ego_pose['rotation'])
    ego_yaw = ego_q.yaw_pitch_roll[0]

    detections = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        cat = ann['category_name']
        if cat not in CLASS_MAP:
            continue
        if ann['num_lidar_pts'] < min_lidar_pts:
            continue

        wx, wy, wz = ann['translation']
        dx = wx - ego_x
        dy = wy - ego_y
        x  =  dx * np.cos(-ego_yaw) - dy * np.sin(-ego_yaw)
        y  =  dx * np.sin(-ego_yaw) + dy * np.cos(-ego_yaw)
        z  =  wz - ego_z

        distance = round(float(np.sqrt(x**2 + y**2)), 2)
        bearing  = round(float(np.degrees(np.arctan2(y, x))), 1)

        # yaw in ego frame — matches BEVFusion schema
        obj_yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        yaw_ego = round(float(obj_yaw - ego_yaw), 4)

        try:
            vel   = nusc.box_velocity(ann_token)
            vx    = float(vel[0]) if not np.isnan(vel[0]) else 0.0
            vy    = float(vel[1]) if not np.isnan(vel[1]) else 0.0
            speed = round(float(np.sqrt(vx**2 + vy**2)), 2)
        except Exception:
            vx, vy, speed = 0.0, 0.0, 0.0

        vis_token  = ann.get('visibility_token', '4')
        visibility = VISIBILITY_MAP.get(str(vis_token), 'unknown')

        detections.append({
            # Core fields — identical schema to BEVFusion
            'class':        CLASS_MAP[cat],
            'x':            round(float(x), 3),
            'y':            round(float(y), 3),
            'z':            round(float(z), 3),
            'distance_m':   distance,
            'bearing_deg':  bearing,
            'confidence':   1.0,
            'width_m':      round(ann['size'][0], 3),
            'length_m':     round(ann['size'][1], 3),
            'height_m':     round(ann['size'][2], 3),
            'yaw_rad':      yaw_ego,
            'source':       'ground_truth',
            # GT bonus fields
            'velocity_ms':  speed,
            'vx':           round(vx, 3),
            'vy':           round(vy, 3),
            'num_lidar_pts': ann['num_lidar_pts'],
            'num_radar_pts': ann['num_radar_pts'],
            'visibility':   visibility,
            'ann_token':    ann_token,
        })

    detections.sort(key=lambda d: d['distance_m'])
    return detections

def make_scene_text(detections, ego_speed=None, include_velocity=True, max_objects=15):
    if not detections:
        return "No objects detected in the scene."

    source = detections[0].get('source', 'ground_truth')
    lines  = [f"{len(detections)} objects detected around the vehicle."]

    for d in detections[:max_objects]:
        line = f"{d['class']} at {d['distance_m']}m bearing {d['bearing_deg']}deg"
        if source == 'bevfusion' and d.get('confidence', 1.0) < 1.0:
            line += f" confidence {int(d['confidence']*100)}%"
        if include_velocity and d.get('velocity_ms', 0) > 0.5:
            line += f" moving at {d['velocity_ms']}m/s"
        if d.get('visibility') in ['mostly_occluded', 'partially_visible']:
            line += f" ({d['visibility']})"
        lines.append(line + ".")

    if ego_speed is not None:
        lines.append(f"Ego vehicle travelling at {ego_speed:.1f}m/s ({ego_speed * 3.6:.1f}km/h).")

    return ' '.join(lines)

def save_bev_map(points, detections, out_path, scene_name="", source="GT"):
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    ax.scatter(points[:, 0], points[:, 1], s=0.4, c='#39d353', alpha=0.35, linewidths=0)

    ax.add_patch(patches.FancyBboxPatch((-1, -2.5), 2, 4.5,
        boxstyle="round,pad=0.1", linewidth=2, edgecolor='white', facecolor='#1f6feb', zorder=5))
    ax.annotate('EGO', (0, 0), color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=6)

    legend_items = {}
    for d in detections:
        cls   = d['class']
        label = CLASS_LIST.index(cls) if cls in CLASS_LIST else 0
        color = colors[label % 10]
        x, y  = d['x'], d['y']
        dx    = d.get('length_m', 4.0)
        dy    = d.get('width_m',  2.0)
        yaw   = d.get('yaw_rad',  0.0)

        ca, sa  = np.cos(yaw), np.sin(yaw)
        corners = np.array([[dx/2,dy/2],[-dx/2,dy/2],[-dx/2,-dy/2],[dx/2,-dy/2]])
        corners = (np.array([[ca,-sa],[sa,ca]]) @ corners.T).T + np.array([x,y])

        ax.add_patch(plt.Polygon(corners, linewidth=1.5, edgecolor=color, facecolor=(*color[:3], 0.2), zorder=4))

        conf = d.get('confidence', 1.0)
        label_text = f"{conf:.2f}" if conf < 1.0 else cls[:3]
        ax.text(x, y, label_text, color='white', fontsize=5, ha='center', va='center', zorder=5,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4))

        if cls not in legend_items:
            legend_items[cls] = patches.Patch(facecolor=color, edgecolor='white', linewidth=0.5, label=cls)

    for r in [10, 20, 30, 40, 50]:
        ax.add_patch(plt.Circle((0,0), r, color='#21262d', fill=False, linewidth=0.5, linestyle='--'))
        ax.text(r+0.5, 0.5, f'{r}m', color='#484f58', fontsize=7)

    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    ax.set_xlabel('X (meters)', color='#8b949e', fontsize=11)
    ax.set_ylabel('Y (meters)', color='#8b949e', fontsize=11)
    ax.set_title(f"MoRAL BEV [{source}] — {scene_name}", color='white', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    if legend_items:
        ax.legend(handles=list(legend_items.values()), loc='upper right',
                  facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=9, framealpha=0.9)
    ax.grid(True, color='#21262d', alpha=0.8, linewidth=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()

def copy_camera_images(nusc, sample, dataroot, out_dir, cameras=None):
    if cameras is None:
        cameras = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                   'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
    copied = {}
    for cam in cameras:
        if cam not in sample['data']:
            continue
        sd  = nusc.get('sample_data', sample['data'][cam])
        src = os.path.join(dataroot, sd['filename'])
        dst = os.path.join(out_dir, f'{cam}.jpg')
        shutil.copy2(src, dst)
        copied[cam] = dst
    return copied

def save_metadata(scene, sample, ego_pose, ego_speed, out_dir):
    meta = {
        'scene_token':     scene['token'],
        'scene_name':      scene['name'],
        'sample_token':    sample['token'],
        'timestamp':       sample['timestamp'],
        'ego_translation': ego_pose['translation'],
        'ego_rotation':    ego_pose['rotation'],
        'ego_speed_ms':    ego_speed,
        'description':     scene.get('description', ''),
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
