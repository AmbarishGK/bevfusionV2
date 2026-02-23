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
  "distance_m":     float,  # sqrt(x²+y²)
  "bearing_deg":    float,  # angle from ego forward axis
  "confidence":     float,  # 0.0-1.0 (GT=1.0, BEVFusion=real score)
  "width_m":        float,
  "length_m":       float,
  "height_m":       float,
  "yaw_rad":        float,  # heading angle — in both GT and BEVFusion
  "source":         str,    # "ground_truth" | "ground_truth_radar" | "bevfusion"

  # GT-only bonus fields (not in BEVFusion, ignored by Cosmos):
  "velocity_ms":    float,
  "vx":             float,
  "vy":             float,
  "num_lidar_pts":  int,
  "num_radar_pts":  int,
  "visibility":     str,
  "ann_token":      str,

  # Radar bonus fields (only when --include-radar):
  "radar_velocity_ms":       float | None,  # Doppler-measured speed
  "radar_vx":                float | None,  # Doppler vx in ego frame
  "radar_vy":                float | None,  # Doppler vy in ego frame
  "radar_velocity_confirmed": bool,         # True if radar point matched
  "radar_rcs":               float | None,  # Radar cross-section (dBsm)
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

# nuScenes radar sensors available
RADAR_SENSORS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT',
]

# nuScenes radar PCD field names (order matches file header)
# The PCD is mixed-type binary — NOT uniform float32.
# We parse the header dynamically; these names are for reference only.
RADAR_FIELD_NAMES = [
    'x', 'y', 'z', 'dyn_prop', 'id', 'rcs',
    'vx', 'vy', 'vx_comp', 'vy_comp',
    'is_quality_valid', 'ambig_state', 'x_rms', 'y_rms',
    'invalid_state', 'pdh0', 'vx_rms', 'vy_rms',
]


# ─────────────────────────────────────────────────────────────
#  Core ego/sensor helpers
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
#  Radar extraction — NEW
# ─────────────────────────────────────────────────────────────

def _load_radar_pcd(pcd_path):
    """
    Parse a nuScenes radar PCD file correctly.

    nuScenes radar PCDs are MIXED-TYPE binary (not uniform float32).
    The SIZE row contains [4,4,4,1,2,4,4,4,4,4,1,1,1,1,1,1,1,1] —
    different byte widths per field. Reading with np.fromfile(dtype=float32)
    gives wrong results. We parse the header and use struct.unpack_from.

    Returns np.ndarray (N, 6): [x, y, z, vx_comp, vy_comp, rcs]
    """
    import struct
    _TYPE_FMT = {
        'F': {'4': 'f', '8': 'd'},
        'I': {'1': 'b', '2': 'h', '4': 'i'},
        'U': {'1': 'B', '2': 'H', '4': 'I'},
    }
    fields, sizes, types, n_points = [], [], [], 0

    with open(pcd_path, 'rb') as f:
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('SIZE'):
                sizes = list(map(int, line.split()[1:]))
            elif line.startswith('TYPE'):
                types = line.split()[1:]
            elif line.startswith('POINTS'):
                n_points = int(line.split()[1])
            elif line.startswith('DATA'):
                break
        binary_data = f.read()

    if n_points == 0 or not fields:
        return np.zeros((0, 6), dtype=np.float32)

    fmt = '<' + ''.join(_TYPE_FMT[t][str(s)] for t, s in zip(types, sizes))
    point_size = struct.calcsize(fmt)

    if len(binary_data) < point_size * n_points:
        return np.zeros((0, 6), dtype=np.float32)

    fi = {name: idx for idx, name in enumerate(fields)}
    rows = []
    for i in range(n_points):
        pt = struct.unpack_from(fmt, binary_data, i * point_size)
        rows.append([
            pt[fi['x']], pt[fi['y']], pt[fi['z']],
            pt[fi['vx_comp']], pt[fi['vy_comp']], pt[fi['rcs']]
        ])

    return np.array(rows, dtype=np.float32)


def get_radar_points_ego(nusc, sample, dataroot, sensors=None):
    """
    Load radar points from all 5 radar sensors and transform into ego frame.

    Uses vx_comp / vy_comp — ego-motion-compensated Doppler velocity.
    These represent the TRUE velocity of the detected object, physically
    measured by the Doppler effect (not estimated from consecutive frames).

    Returns
    -------
    np.ndarray of shape (N, 6):
        columns: [x_ego, y_ego, z_ego, vx_comp_ego, vy_comp_ego, rcs]
    """
    if sensors is None:
        sensors = RADAR_SENSORS

    all_points = []

    for sensor in sensors:
        if sensor not in sample['data']:
            continue

        sd = nusc.get('sample_data', sample['data'][sensor])
        pcd_path = os.path.join(dataroot, sd['filename'])

        if not os.path.exists(pcd_path):
            continue

        pts = _load_radar_pcd(pcd_path)
        if pts.shape[0] == 0:
            continue

        # Position + velocity in sensor frame
        x_s  = pts[:, 0];  y_s  = pts[:, 1];  z_s  = pts[:, 2]
        vx_s = pts[:, 3];  vy_s = pts[:, 4];  rcs  = pts[:, 5]

        # Transform sensor frame → ego frame
        cs       = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        T_sensor = np.array(cs['translation'])
        R_sensor = Quaternion(cs['rotation']).rotation_matrix

        pts_s = np.vstack([x_s, y_s, z_s])                      # (3, N)
        pts_e = R_sensor @ pts_s                                  # rotate
        pts_e[0] += T_sensor[0]                                   # translate
        pts_e[1] += T_sensor[1]
        pts_e[2] += T_sensor[2]

        vel_s = np.vstack([vx_s, vy_s, np.zeros_like(vx_s)])    # (3, N)
        vel_e = R_sensor @ vel_s                                  # rotate only

        block = np.column_stack([pts_e[0], pts_e[1], pts_e[2],
                                 vel_e[0], vel_e[1], rcs])
        all_points.append(block)

    if not all_points:
        return np.zeros((0, 6), dtype=np.float32)

    return np.vstack(all_points).astype(np.float32)


def enrich_detections_with_radar(detections, radar_pts, match_radius_m=3.0):
    """
    For each GT/BEVFusion detection, find the nearest radar point within
    match_radius_m and attach its Doppler velocity.

    This is a simple nearest-neighbour match in 2D (x,y) — good enough
    for the nuScenes mini dataset where objects are well separated.

    Parameters
    ----------
    detections   : list of detection dicts (from get_detections_from_gt)
    radar_pts    : np.ndarray (N, 6) from get_radar_points_ego
    match_radius_m : float, max distance for radar↔object match (default 3m)

    Returns
    -------
    detections with radar fields added in-place.
    """
    enriched = []

    for d in detections:
        d = dict(d)  # copy so we don't mutate original
        obj_x = d['x']
        obj_y = d['y']

        radar_vel_ms  = None
        radar_vx      = None
        radar_vy      = None
        radar_rcs_val = None
        confirmed     = False

        if radar_pts.shape[0] > 0:
            # 2D distance from detection centre to each radar point
            dx = radar_pts[:, 0] - obj_x
            dy = radar_pts[:, 1] - obj_y
            dists = np.sqrt(dx**2 + dy**2)
            nearest_idx = int(np.argmin(dists))
            nearest_dist = float(dists[nearest_idx])

            if nearest_dist <= match_radius_m:
                vx_r = float(radar_pts[nearest_idx, 3])
                vy_r = float(radar_pts[nearest_idx, 4])
                spd  = float(np.sqrt(vx_r**2 + vy_r**2))

                radar_vel_ms  = round(spd, 2)
                radar_vx      = round(vx_r, 3)
                radar_vy      = round(vy_r, 3)
                radar_rcs_val = round(float(radar_pts[nearest_idx, 5]), 2)
                confirmed     = True

        d['radar_velocity_ms']        = radar_vel_ms
        d['radar_vx']                 = radar_vx
        d['radar_vy']                 = radar_vy
        d['radar_rcs']                = radar_rcs_val
        d['radar_velocity_confirmed'] = confirmed

        # Update source tag
        if d.get('source') == 'ground_truth':
            d['source'] = 'ground_truth_radar'
        elif d.get('source') == 'bevfusion':
            d['source'] = 'bevfusion_radar'

        enriched.append(d)

    return enriched


# ─────────────────────────────────────────────────────────────
#  GT detections
# ─────────────────────────────────────────────────────────────

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
            'class':         CLASS_MAP[cat],
            'x':             round(float(x), 3),
            'y':             round(float(y), 3),
            'z':             round(float(z), 3),
            'distance_m':    distance,
            'bearing_deg':   bearing,
            'confidence':    1.0,
            'width_m':       round(ann['size'][0], 3),
            'length_m':      round(ann['size'][1], 3),
            'height_m':      round(ann['size'][2], 3),
            'yaw_rad':       yaw_ego,
            'source':        'ground_truth',
            'velocity_ms':   speed,
            'vx':            round(vx, 3),
            'vy':            round(vy, 3),
            'num_lidar_pts': ann['num_lidar_pts'],
            'num_radar_pts': ann['num_radar_pts'],
            'visibility':    visibility,
            'ann_token':     ann_token,
        })

    detections.sort(key=lambda d: d['distance_m'])
    return detections


# ─────────────────────────────────────────────────────────────
#  Scene text generation
# ─────────────────────────────────────────────────────────────

def make_scene_text(detections, ego_speed=None, include_velocity=True, max_objects=15):
    """
    Convert detections list into natural language for Cosmos-Reason2.

    When radar data is present (radar_velocity_confirmed=True), the text says
    "radar-confirmed X m/s" instead of just "moving at X m/s", making it
    clear to the VLM that the velocity is physically measured, not estimated.
    """
    if not detections:
        return "No objects detected in the scene."

    source = detections[0].get('source', 'ground_truth')
    has_radar = 'radar' in source
    lines = [f"{len(detections)} objects detected around the vehicle."]

    for d in detections[:max_objects]:
        line = f"{d['class']} at {d['distance_m']}m bearing {d['bearing_deg']}deg"

        # Confidence — only for BEVFusion (GT confidence is always 1.0)
        if 'bevfusion' in source and d.get('confidence', 1.0) < 1.0:
            line += f" confidence {int(d['confidence']*100)}%"

        # Velocity — prefer radar-confirmed over GT estimate
        if include_velocity:
            if has_radar and d.get('radar_velocity_confirmed'):
                rv = d['radar_velocity_ms']
                if rv is not None and rv > 0.3:
                    line += f" radar-confirmed {rv}m/s"
            elif d.get('velocity_ms', 0) > 0.5:
                line += f" moving at {d['velocity_ms']}m/s"

        # Occlusion warning
        if d.get('visibility') in ['mostly_occluded', 'partially_visible']:
            line += f" ({d['visibility']})"

        lines.append(line + ".")

    if ego_speed is not None:
        lines.append(f"Ego vehicle travelling at {ego_speed:.1f}m/s ({ego_speed * 3.6:.1f}km/h).")

    return ' '.join(lines)


# ─────────────────────────────────────────────────────────────
#  BEV map visualisation
# ─────────────────────────────────────────────────────────────

def save_bev_map(points, detections, out_path, scene_name="", source="GT",
                 radar_pts=None):
    """
    Save a BEV visualisation.

    If radar_pts is provided (N,6 array in ego frame), radar returns
    are overlaid as cyan dots with velocity arrows — giving a visible
    difference from the non-radar BEV map.
    """
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # LiDAR points
    ax.scatter(points[:, 0], points[:, 1], s=0.4, c='#39d353',
               alpha=0.35, linewidths=0, label='LiDAR')

    # Radar returns + velocity arrows (cyan)
    if radar_pts is not None and radar_pts.shape[0] > 0:
        ax.scatter(radar_pts[:, 0], radar_pts[:, 1], s=6, c='cyan',
                   alpha=0.7, linewidths=0, zorder=3, label='Radar')
        # Draw Doppler velocity arrows (scaled for visibility)
        scale = 1.5
        for rp in radar_pts:
            rx, ry, _, rvx, rvy, _ = rp
            spd = np.sqrt(rvx**2 + rvy**2)
            if spd > 0.5:
                ax.annotate('', xy=(rx + rvx*scale, ry + rvy*scale),
                            xytext=(rx, ry),
                            arrowprops=dict(arrowstyle='->', color='cyan',
                                            lw=0.8, alpha=0.6))

    # Ego vehicle
    ax.add_patch(patches.FancyBboxPatch((-1, -2.5), 2, 4.5,
        boxstyle="round,pad=0.1", linewidth=2,
        edgecolor='white', facecolor='#1f6feb', zorder=5))
    ax.annotate('EGO', (0, 0), color='white', ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=6)

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
        corners = np.array([[dx/2, dy/2], [-dx/2, dy/2],
                             [-dx/2,-dy/2], [dx/2,-dy/2]])
        corners = (np.array([[ca,-sa],[sa, ca]]) @ corners.T).T + np.array([x, y])

        # Highlight radar-confirmed objects with a brighter edge
        edge_color = 'cyan' if d.get('radar_velocity_confirmed') else color
        edge_width = 2.0   if d.get('radar_velocity_confirmed') else 1.5
        ax.add_patch(plt.Polygon(corners, linewidth=edge_width,
                                 edgecolor=edge_color,
                                 facecolor=(*color[:3], 0.2), zorder=4))

        conf = d.get('confidence', 1.0)
        label_text = f"{conf:.2f}" if conf < 1.0 else cls[:3]
        ax.text(x, y, label_text, color='white', fontsize=5,
                ha='center', va='center', zorder=5,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4))

        if cls not in legend_items:
            legend_items[cls] = patches.Patch(
                facecolor=color, edgecolor='white',
                linewidth=0.5, label=cls)

    # Range rings
    for r in [10, 20, 30, 40, 50]:
        ax.add_patch(plt.Circle((0,0), r, color='#21262d',
                                fill=False, linewidth=0.5, linestyle='--'))
        ax.text(r+0.5, 0.5, f'{r}m', color='#484f58', fontsize=7)

    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    ax.set_xlabel('X (meters)', color='#8b949e', fontsize=11)
    ax.set_ylabel('Y (meters)', color='#8b949e', fontsize=11)
    ax.set_title(f"MoRAL BEV [{source}] — {scene_name}",
                 color='white', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    if legend_items:
        ax.legend(handles=list(legend_items.values()), loc='upper right',
                  facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='white', fontsize=9, framealpha=0.9)
    ax.grid(True, color='#21262d', alpha=0.8, linewidth=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()


# ─────────────────────────────────────────────────────────────
#  Camera images + metadata
# ─────────────────────────────────────────────────────────────

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


def save_metadata(scene, sample, ego_pose, ego_speed, out_dir,
                  radar_summary=None):
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
    if radar_summary is not None:
        meta['radar'] = radar_summary
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
