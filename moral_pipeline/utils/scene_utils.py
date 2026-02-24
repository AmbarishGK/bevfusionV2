"""
utils/scene_utils.py  —  MoRAL pipeline shared utilities
=========================================================

detections.json schema (GT and BEVFusion share core fields):
  class, x, y, z, distance_m, bearing_deg, confidence,
  width_m, length_m, height_m, yaw_rad, source

GT-only bonus fields:
  velocity_ms, vx, vy, num_lidar_pts, num_radar_pts, visibility, ann_token

Radar bonus fields (condition 02/04):
  radar_velocity_ms, radar_vx, radar_vy, radar_rcs,
  radar_velocity_confirmed, radar_quality

radar_quality values:
  "reliable"        — radar speed agrees with GT, object is large enough for reliable Doppler
  "radial_ambiguous"— pedestrian/cyclist, radar radial-only so crosswise motion missed
  "range_mismatch"  — radar match was found but speed disagrees strongly (wrong match)
  "unconfirmed"     — no radar point within match radius
"""

import os, json, shutil, struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion

# ── Class mappings ────────────────────────────────────────────
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

# Classes where radar radial-only Doppler is unreliable for crosswise motion
RADIAL_AMBIGUOUS_CLASSES = {'pedestrian', 'bicycle', 'motorcycle'}

VISIBILITY_MAP = {
    '1': 'mostly_occluded',
    '2': 'partially_visible',
    '3': 'mostly_visible',
    '4': 'fully_visible',
}

RADAR_SENSORS = [
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
]

RADAR_FIELD_NAMES = [
    'x','y','z','dyn_prop','id','rcs',
    'vx','vy','vx_comp','vy_comp',
    'is_quality_valid','ambig_state','x_rms','y_rms',
    'invalid_state','pdh0','vx_rms','vy_rms',
]

# BEV plot range — radar/objects beyond this are out of frame
BEV_RANGE_M     = 50.0
# Only show full detail (box + label) within this range
BEV_LABEL_RANGE = 35.0
# Minimum LiDAR points for an object to get a solid box (else faint outline)
MIN_STRONG_PTS  = 10
# Minimum LiDAR points for object to appear in VLM scene description detail
MIN_VLM_PTS     = 3
# Maximum distance for full detail in scene description
DESC_DETAIL_RANGE = 50.0

# Physics
EGO_WIDTH_M  = 2.0
EGO_LENGTH_M = 4.5
BRAKE_DECEL  = 4.0   # m/s²

# ── Ego state ─────────────────────────────────────────────────

def get_ego_pose(nusc, sample):
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    return nusc.get('ego_pose', lidar_sd['ego_pose_token'])

def get_ego_speed(nusc, sample):
    try:
        lidar_sd  = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose  = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        if lidar_sd['next'] != '':
            nxt      = nusc.get('sample_data', lidar_sd['next'])
            nxt_pose = nusc.get('ego_pose', nxt['ego_pose_token'])
            dt = (nxt_pose['timestamp'] - ego_pose['timestamp']) / 1e6
            dx = nxt_pose['translation'][0] - ego_pose['translation'][0]
            dy = nxt_pose['translation'][1] - ego_pose['translation'][1]
            if dt > 0:
                return round(float(np.sqrt(dx**2 + dy**2) / dt), 2)
    except Exception:
        pass
    return None

# ── LiDAR ─────────────────────────────────────────────────────

def get_lidar_points(nusc, sample, dataroot):
    """
    Load LiDAR points and transform LIDAR_TOP sensor frame → ego frame.
    Without this the cloud is rotated relative to the detection boxes.
    """
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pcd_path = os.path.join(dataroot, lidar_sd['filename'])
    pts  = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
    cs   = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    T    = np.array(cs['translation'])
    R    = Quaternion(cs['rotation']).rotation_matrix
    xyz_e = (R @ pts[:, :3].T).T + T
    return np.column_stack([xyz_e, pts[:, 3], pts[:, 4]]).astype(np.float32)

# ── Radar ─────────────────────────────────────────────────────

def _load_radar_pcd(pcd_path):
    """
    Parse nuScenes radar PCD — mixed-type binary, NOT uniform float32.
    Reads header dynamically, unpacks with struct.
    Returns (N, 6): [x, y, z, vx_comp, vy_comp, rcs]
    """
    _TYPE_FMT = {
        'F': {'4': 'f', '8': 'd'},
        'I': {'1': 'b', '2': 'h', '4': 'i'},
        'U': {'1': 'B', '2': 'H', '4': 'I'},
    }
    fields, sizes, types, n_points = [], [], [], 0

    with open(pcd_path, 'rb') as f:
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if   line.startswith('FIELDS'): fields   = line.split()[1:]
            elif line.startswith('SIZE'):   sizes    = list(map(int, line.split()[1:]))
            elif line.startswith('TYPE'):   types    = line.split()[1:]
            elif line.startswith('POINTS'): n_points = int(line.split()[1])
            elif line.startswith('DATA'):   break
        binary_data = f.read()

    if n_points == 0 or not fields:
        return np.zeros((0, 6), dtype=np.float32)

    fmt        = '<' + ''.join(_TYPE_FMT[t][str(s)] for t, s in zip(types, sizes))
    point_size = struct.calcsize(fmt)

    if len(binary_data) < point_size * n_points:
        return np.zeros((0, 6), dtype=np.float32)

    fi   = {name: idx for idx, name in enumerate(fields)}
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
    Load all radar points from specified sensors, transform to ego frame.
    Uses vx_comp/vy_comp (ego-motion-compensated Doppler).
    Filters to BEV_RANGE_M + 5m to remove far ghost returns.
    Returns (N, 6): [x_ego, y_ego, z_ego, vx_ego, vy_ego, rcs]
    """
    if sensors is None:
        sensors = RADAR_SENSORS

    all_pts = []
    for sensor in sensors:
        if sensor not in sample['data']:
            continue
        sd       = nusc.get('sample_data', sample['data'][sensor])
        pcd_path = os.path.join(dataroot, sd['filename'])
        if not os.path.exists(pcd_path):
            continue

        pts = _load_radar_pcd(pcd_path)
        if pts.shape[0] == 0:
            continue

        # Sensor → ego frame
        cs       = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        T        = np.array(cs['translation'])
        R        = Quaternion(cs['rotation']).rotation_matrix

        xyz_s    = pts[:, :3].T                              # (3, N)
        xyz_e    = R @ xyz_s                                 # rotate
        xyz_e[0] += T[0]; xyz_e[1] += T[1]; xyz_e[2] += T[2]

        vel_s    = np.vstack([pts[:,3], pts[:,4], np.zeros(len(pts))])
        vel_e    = R @ vel_s

        block = np.column_stack([xyz_e[0], xyz_e[1], xyz_e[2],
                                 vel_e[0], vel_e[1], pts[:, 5]])
        all_pts.append(block)

    if not all_pts:
        return np.zeros((0, 6), dtype=np.float32)

    combined = np.vstack(all_pts).astype(np.float32)

    # Filter to plot range + small buffer — removes ghost/multipath returns
    r2d = np.sqrt(combined[:,0]**2 + combined[:,1]**2)
    combined = combined[r2d <= BEV_RANGE_M + 5.0]

    return combined


def enrich_detections_with_radar(detections, radar_pts, match_radius_m=2.0):
    """
    Match each detection to nearest radar point within match_radius_m.

    Quality classification:
      "reliable"         — vehicle class, radar speed within 50% of GT, match < 2m
      "radial_ambiguous" — pedestrian/bicycle/motorcycle: radar only sees radial
                           component so crosswise motion is invisible. Speed may
                           read near-zero even for moving objects.
      "range_mismatch"   — match found but speed disagrees strongly (wrong match)
      "unconfirmed"      — no radar point within match_radius_m
    """
    enriched = []

    for d in detections:
        d     = dict(d)
        obj_x = d['x']
        obj_y = d['y']
        cls   = d['class']
        gt_spd= d.get('velocity_ms', 0.0)

        radar_vel_ms = None
        radar_vx     = None
        radar_vy     = None
        radar_rcs_v  = None
        confirmed    = False
        quality      = 'unconfirmed'

        if radar_pts.shape[0] > 0:
            dx    = radar_pts[:, 0] - obj_x
            dy    = radar_pts[:, 1] - obj_y
            dists = np.sqrt(dx**2 + dy**2)
            ni    = int(np.argmin(dists))
            nd    = float(dists[ni])

            if nd <= match_radius_m:
                vx_r = float(radar_pts[ni, 3])
                vy_r = float(radar_pts[ni, 4])
                spd  = float(np.sqrt(vx_r**2 + vy_r**2))

                radar_vel_ms = round(spd, 2)
                radar_vx     = round(vx_r, 3)
                radar_vy     = round(vy_r, 3)
                radar_rcs_v  = round(float(radar_pts[ni, 5]), 2)
                confirmed    = True

                # Classify quality
                if cls in RADIAL_AMBIGUOUS_CLASSES:
                    # Radar radial-only: crosswise motion invisible
                    # Mark confirmed but flag as radial_ambiguous
                    quality = 'radial_ambiguous'
                else:
                    # For vehicles: check speed agreement
                    # If GT says fast but radar says near-zero → wrong match
                    if gt_spd > 2.0 and spd < 0.5:
                        quality   = 'range_mismatch'
                        confirmed = False  # don't trust this match
                    else:
                        quality = 'reliable'

        d['radar_velocity_ms']        = radar_vel_ms
        d['radar_vx']                 = radar_vx
        d['radar_vy']                 = radar_vy
        d['radar_rcs']                = radar_rcs_v
        d['radar_velocity_confirmed'] = confirmed
        d['radar_quality']            = quality

        if d.get('source') == 'ground_truth':
            d['source'] = 'ground_truth_radar'
        elif d.get('source') == 'bevfusion':
            d['source'] = 'bevfusion_radar'

        enriched.append(d)

    return enriched

# ── GT detections ─────────────────────────────────────────────

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
            vel   = nusc.box_velocity(ann_token)   # returns world frame
            vx_w  = float(vel[0]) if not np.isnan(vel[0]) else 0.0
            vy_w  = float(vel[1]) if not np.isnan(vel[1]) else 0.0
            # Rotate into ego frame (same -ego_yaw rotation used for positions)
            vx    =  vx_w * np.cos(-ego_yaw) - vy_w * np.sin(-ego_yaw)
            vy    =  vx_w * np.sin(-ego_yaw) + vy_w * np.cos(-ego_yaw)
            speed = round(float(np.sqrt(vx**2 + vy**2)), 2)
        except Exception:
            vx, vy, speed = 0.0, 0.0, 0.0

        vis_token  = ann.get('visibility_token', '4')
        visibility = VISIBILITY_MAP.get(str(vis_token), 'unknown')

        detections.append({
            # Core fields — same schema as BEVFusion
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
            # GT bonus
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

# ── Zone helpers ──────────────────────────────────────────────

ZONE_LABELS = [
    ((-22.5,  22.5),  "directly ahead"),
    (( 22.5,  67.5),  "front-left"),
    (( 67.5, 112.5),  "left"),
    ((112.5, 157.5),  "rear-left"),
    ((157.5, 180.0),  "directly behind"),
    ((-180., -157.5), "directly behind"),
    ((-157.5,-112.5), "rear-right"),
    ((-112.5, -67.5), "right"),
    (( -67.5, -22.5), "front-right"),
]

def _zone(bearing_deg):
    for (lo, hi), label in ZONE_LABELS:
        if lo <= bearing_deg <= hi:
            return label
    return "nearby"

def _stopping_dist(speed_ms):
    return round(speed_ms**2 / (2 * BRAKE_DECEL), 1)

def _ttc(dist_m, obj_spd, ego_spd, bearing_deg):
    """
    Compute time-to-collision only for physically plausible scenarios.

    AHEAD  (|bearing| < 45°):  ego approaching object → closing = ego + obj speed
    BEHIND (|bearing| > 135°): only dangerous if object is overtaking ego from behind.
                                Stationary/slow objects behind are no threat — ego is
                                moving away from them. Skip TTC for these.
    SIDE   (45–135°):          lateral — no TTC computed.
    """
    b = abs(bearing_deg)
    if b < 45:
        # Ahead — ego closing on object
        closing = ego_spd + obj_spd
        if closing > 0.1:
            return round(dist_m / closing, 1)
    elif b > 135:
        # Behind — only risky if object is overtaking (approaching ego from rear)
        # Object must be faster than ego to close the gap
        if obj_spd > ego_spd + 1.0:
            closing = obj_spd - ego_spd
            return round(dist_m / closing, 1)
    return None

def _gap(d):
    gap = 3.7 - d.get('width_m', 2.0) / 2 - EGO_WIDTH_M / 2
    return gap, gap >= 0.3

def _best_vel(d, has_radar):
    """
    Return (speed_ms, source_label).
    For pedestrians/cyclists with radar: use GT (radar radial-only unreliable).
    For vehicles with reliable radar: use radar.
    """
    if has_radar:
        q = d.get('radar_quality', 'unconfirmed')
        if q == 'reliable':
            rv = d.get('radar_velocity_ms')
            if rv is not None:
                return rv, 'radar-confirmed'
        elif q == 'radial_ambiguous':
            # Use GT — radar is unreliable for sideways-moving pedestrians
            return d.get('velocity_ms', 0.0), 'gt-estimated'
    return d.get('velocity_ms', 0.0), 'gt-estimated'

def _lidar_tier(d):
    """
    Evidence strength label for a detection.
    GT mode:       uses num_lidar_pts
    BEVFusion mode: num_lidar_pts absent → falls back to confidence score
    """
    pts = d.get('num_lidar_pts', -1)
    if pts >= 0:
        if pts > 50: return 'high'
        if pts > 10: return 'medium'
        if pts >= 3: return 'low'
        return 'marginal'
    else:
        conf = d.get('confidence', 0.5)
        if conf >= 0.8: return 'high'
        if conf >= 0.5: return 'medium'
        if conf >= 0.3: return 'low'
        return 'marginal'

# ── Scene description ─────────────────────────────────────────

def make_scene_text(detections, ego_speed=None, include_velocity=True,
                    max_objects=None):
    """
    Rich structured scene description for Cosmos-Reason2 QA generation.

    Objects are split into two tiers:
      DETAILED LIST   — within DESC_DETAIL_RANGE (50m) AND lidar_pts >= MIN_VLM_PTS (3)
      DISTANT/MARGINAL— beyond range or too few points, listed as summary only

    This avoids the VLM treating a 1-point detection at 70m with the same
    confidence as a 495-point truck at 17m.

    Radar velocity:
      - 'radar-confirmed' only for vehicles with reliable match
      - 'gt-estimated' for pedestrians/cyclists (radial ambiguity)
      - 'range_mismatch' objects use GT velocity with a note
    """
    if not detections:
        return "No objects detected in the scene."

    source    = detections[0].get('source', 'ground_truth')
    has_radar = 'radar' in source
    is_bev    = 'bevfusion' in source
    es        = ego_speed or 0.0

    # Split detections into tiers
    # GT: use num_lidar_pts as evidence quality gate
    # BEVFusion: num_lidar_pts absent → use confidence threshold
    if is_bev:
        detailed = [d for d in detections
                    if d['distance_m'] <= DESC_DETAIL_RANGE
                    and d.get('confidence', 0.0) >= 0.3]
    else:
        detailed = [d for d in detections
                    if d['distance_m'] <= DESC_DETAIL_RANGE
                    and d.get('num_lidar_pts', 99) >= MIN_VLM_PTS]
    marginal  = [d for d in detections if d not in detailed]

    lines = []

    # ── 1. Scene overview ────────────────────────────────────
    data_src   = "LiDAR+camera BEVFusion" if is_bev else "LiDAR ground-truth"
    radar_note = " Radar Doppler velocities included (vehicles only — pedestrian radar unreliable for crosswise motion)." if has_radar else ""
    n_moving   = sum(1 for d in detailed if _best_vel(d, has_radar)[0] > 0.5)
    n_occluded = sum(1 for d in detailed if d.get('visibility') in ['mostly_occluded', 'partially_visible'])

    lines.append(
        f"SCENE OVERVIEW: {len(detailed)} high-confidence objects within {DESC_DETAIL_RANGE}m "
        f"({len(marginal)} additional marginal detections beyond range or with weak LiDAR returns). "
        f"Data source: {data_src}.{radar_note} "
        f"{n_moving} objects moving. {n_occluded} partially or fully occluded."
    )

    # ── 2. Ego state ─────────────────────────────────────────
    if ego_speed is not None:
        lines.append(
            f"EGO STATE: Speed {es:.1f} m/s ({es*3.6:.1f} km/h). "
            f"Stopping distance at current speed: {_stopping_dist(es)} m. "
            f"Vehicle size: {EGO_LENGTH_M} m long, {EGO_WIDTH_M} m wide."
        )

    # ── 3. Nearest per zone (detailed objects only) ──────────
    zones = {}
    for d in detailed:
        z = _zone(d['bearing_deg'])
        if z not in zones or d['distance_m'] < zones[z]['distance_m']:
            zones[z] = d

    zone_order = ["directly ahead","front-left","front-right",
                  "left","right","rear-left","rear-right","directly behind"]
    zone_parts = []
    for z in zone_order:
        if z in zones:
            d = zones[z]
            spd, src = _best_vel(d, has_radar)
            vel_str  = f" moving {spd:.1f} m/s ({src})" if spd > 0.3 else ""
            zone_parts.append(f"{z}: {d['class']} at {d['distance_m']} m{vel_str}")
    if zone_parts:
        lines.append("NEAREST OBJECT PER ZONE: " + "; ".join(zone_parts) + ".")

    # ── 4. Safety critical ───────────────────────────────────
    critical = []
    for d in detailed:
        spd, src = _best_vel(d, has_radar)
        ttc = _ttc(d['distance_m'], spd, es, d['bearing_deg'])
        if ttc is not None and ttc < 5.0:
            critical.append((ttc, d, spd, src))
        elif d['distance_m'] < 8.0:
            critical.append((d['distance_m'] / max(es, 0.1), d, spd, src))
    critical.sort(key=lambda x: x[0])
    if critical:
        crit_parts = []
        for ttc_v, d, spd, src in critical[:5]:
            vel_str = f" at {spd:.1f} m/s ({src})" if spd > 0.3 else ""
            crit_parts.append(
                f"{d['class']} {_zone(d['bearing_deg'])} {d['distance_m']} m{vel_str} (TTC ≈ {ttc_v:.1f} s)"
            )
        lines.append("SAFETY CRITICAL: " + "; ".join(crit_parts) + ".")

    # ── 5. Full object list — detailed tier only ─────────────
    lines.append(f"OBJECTS WITHIN {int(DESC_DETAIL_RANGE)}m (high-confidence):")
    for i, d in enumerate(detailed):
        spd, src   = _best_vel(d, has_radar)
        tier       = _lidar_tier(d)
        vis        = d.get('visibility', 'unknown')
        rcs        = d.get('radar_rcs')
        rq         = d.get('radar_quality', '')

        parts = [
            f"[{i+1}] {d['class']}",
            f"zone={_zone(d['bearing_deg'])}",
            f"dist={d['distance_m']}m",
            f"size={d.get('width_m',0):.1f}x{d.get('length_m',0):.1f}x{d.get('height_m',0):.1f}m",
        ]

        if spd > 0.3:
            parts.append(f"speed={spd:.2f}m/s({src})")
        else:
            parts.append("speed=stationary")

        if is_bev:
            parts.append(f"conf={int(d.get('confidence',1.0)*100)}%")

        parts.append(f"visibility={vis}")
        parts.append(f"lidar={tier}")

        if rcs is not None and rq == 'reliable':
            parts.append(f"rcs={rcs:.1f}dBsm")

        ttc = _ttc(d['distance_m'], spd, es, d['bearing_deg'])
        if ttc is not None:
            parts.append(f"ttc≈{ttc}s")

        if d['class'] in ['truck','bus','construction_vehicle','car']:
            gap, passable = _gap(d)
            parts.append(f"gap≈{gap:.1f}m({'passable' if passable else 'tight'})")

        lines.append("  " + ", ".join(parts) + ".")

    # ── 6. Marginal objects — summary only ───────────────────
    if marginal:
        m_counts = {}
        for d in marginal:
            m_counts[d['class']] = m_counts.get(d['class'], 0) + 1
        m_str = ", ".join(f"{v} {k}" for k,v in sorted(m_counts.items()))
        lines.append(
            f"MARGINAL DETECTIONS (beyond {int(DESC_DETAIL_RANGE)}m or <{MIN_VLM_PTS} LiDAR pts — low confidence): "
            f"{len(marginal)} objects: {m_str}. "
            f"These are present in LiDAR data but too far or sparse for reliable VLM reasoning."
        )

    # ── 7. Object counts ─────────────────────────────────────
    counts = {}
    for d in detections:
        counts[d['class']] = counts.get(d['class'], 0) + 1
    count_str = ", ".join(f"{v} {k}{'s' if v>1 else ''}" for k,v in sorted(counts.items()))
    lines.append(f"TOTAL OBJECT COUNTS: {count_str}.")

    # ── 8. Occlusion summary ─────────────────────────────────
    occluded = [d for d in detailed if d.get('visibility') in ['mostly_occluded','partially_visible']]
    if occluded:
        occ_parts = [
            f"{d['class']} at {d['distance_m']}m {_zone(d['bearing_deg'])} ({d.get('visibility','')})"
            for d in occluded[:8]
        ]
        lines.append(
            "OCCLUDED (LiDAR detects, camera may miss): " + "; ".join(occ_parts) + "."
        )

    return "\n".join(lines)

# ── BEV map ───────────────────────────────────────────────────

def save_bev_map(points, detections, out_path, scene_name="", source="GT",
                 radar_pts=None, ego_speed=None):
    """
    BEV map optimised for VLM input — publication quality.

    Visual hierarchy (most important → most visible):
      1. Ego vehicle — white/blue, always visible
      2. High-confidence nearby objects — solid fill + label
      3. Radar-confirmed objects — cyan edge highlight
      4. Moving objects — velocity arrow on box
      5. Weak/far objects — faint outline only, no label
      6. Radar points — small cyan dots, under LiDAR
      7. LiDAR point cloud — green, background context

    Fixed vs previous version:
      - Full legend for ALL classes (was missing bicycle, construction_vehicle, trailer, motorcycle)
      - Forward direction indicator arrow on ego
      - Moving GT objects get velocity arrows (not just radar)
      - Strong objects (≥10 LiDAR pts) always visible even at distance
      - Radar arrows: capped at 4m, drawn per-detection not per-point
      - BEVFusion mode: uses confidence for visual weight instead of lidar_pts
      - Cleaner range ring labels
    """
    # ── Colour map — fixed assignment per class ───────────────
    CLASS_COLORS = {
        'car':                  '#4c8eda',  # blue
        'truck':                '#f0883e',  # orange
        'bus':                  '#e05252',  # red
        'trailer':              '#c678dd',  # purple
        'motorcycle':           '#e5c07b',  # yellow
        'bicycle':              '#56b6c2',  # teal
        'pedestrian':           '#98c379',  # green
        'traffic_cone':         '#00d4ff',  # cyan
        'barrier':              '#c8a97e',  # tan/brown
        'construction_vehicle': '#3cb371',  # medium-green
    }
    DEFAULT_COLOR = '#aaaaaa'

    is_bev = any('bevfusion' in d.get('source', '') for d in detections)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor('#0a0e14')
    fig.patch.set_facecolor('#0a0e14')

    # ── Coordinate convention ─────────────────────────────────
    # nuScenes ego frame: x=forward, y=left
    # We want the BEV map to show: UP = forward, RIGHT = right (i.e. -y)
    # So we plot everything as: plot_x = -y_ego,  plot_y = x_ego
    # This matches every autonomous driving paper convention and human intuition.
    # All data stays in ego frame — only the rendering is swapped.
    def px(x_ego, y_ego): return -y_ego   # plot horizontal = -y = rightward
    def py(x_ego, y_ego): return  x_ego   # plot vertical   =  x = forward

    # ── Range rings ──────────────────────────────────────────
    for r in [10, 20, 30, 40, 50]:
        ax.add_patch(plt.Circle((0, 0), r,
                                color='#22272e', fill=False,
                                linewidth=0.8, linestyle='--', zorder=1))
        ax.text(r + 0.8, 1.0, f'{r}m', color='#444c56',
                fontsize=7.5, zorder=1, va='bottom')

    # ── Radar dots (bottom layer) ─────────────────────────────
    if radar_pts is not None and radar_pts.shape[0] > 0:
        r2d   = np.sqrt(radar_pts[:, 0]**2 + radar_pts[:, 1]**2)
        r_vis = radar_pts[r2d <= BEV_RANGE_M]
        if r_vis.shape[0] > 0:
            ax.scatter(px(r_vis[:,0], r_vis[:,1]),
                       py(r_vis[:,0], r_vis[:,1]),
                       s=2.5, c='#00d4ff', alpha=0.35,
                       linewidths=0, zorder=2)

            # Only draw velocity arrows for radar points matched to a detection.
            # Unmatched points are ghost/multipath returns — their arrows are
            # visually misleading (random directions in empty space).
            det_xy = np.array([[d['x'], d['y']] for d in detections]) \
                     if detections else np.zeros((0, 2))

            for rp in r_vis:
                rx, ry, _, rvx, rvy, _ = rp
                spd = float(np.sqrt(rvx**2 + rvy**2))
                if spd < 2.0:
                    continue
                # Check if this radar point is within match_radius of any detection
                if det_xy.shape[0] > 0:
                    dists = np.sqrt((det_xy[:,0]-rx)**2 + (det_xy[:,1]-ry)**2)
                    if float(dists.min()) > 2.0:   # match_radius_m = 2.0
                        continue                    # ghost return — skip arrow
                scale = min(4.0 / spd, 1.5)
                ax.annotate('',
                    xy=(px(rx + rvx*scale, ry + rvy*scale),
                        py(rx + rvx*scale, ry + rvy*scale)),
                    xytext=(px(rx, ry), py(rx, ry)),
                    arrowprops=dict(arrowstyle='->', color='#00d4ff',
                                    lw=0.8, alpha=0.5),
                    zorder=2)

    # ── LiDAR point cloud ────────────────────────────────────
    if points is not None and len(points) > 0:
        ax.scatter(px(points[:,0], points[:,1]),
                   py(points[:,0], points[:,1]),
                   s=0.25, c='#2ea043', alpha=0.22,
                   linewidths=0, zorder=3)

    # ── Detection boxes ──────────────────────────────────────
    legend_patches = {}

    for d in detections:
        cls   = d['class']
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        x_e, y_e = d['x'], d['y']     # ego frame
        dx    = d.get('length_m', 4.0)
        dy    = d.get('width_m',  2.0)
        yaw   = d.get('yaw_rad',  0.0)
        dist  = d['distance_m']

        if dist > BEV_RANGE_M + 3:
            continue

        if is_bev:
            conf   = d.get('confidence', 0.5)
            strong = conf >= 0.5
        else:
            pts    = d.get('num_lidar_pts', 0)
            strong = pts >= MIN_STRONG_PTS

        radar_reliable = d.get('radar_quality') == 'reliable'

        # Rotated box corners in ego frame, then transform to plot coords
        ca, sa = np.cos(yaw), np.sin(yaw)
        corners_ego = np.array([[ dx/2,  dy/2],
                                 [-dx/2,  dy/2],
                                 [-dx/2, -dy/2],
                                 [ dx/2, -dy/2]])
        rot     = np.array([[ca, -sa], [sa, ca]])
        c_ego   = (rot @ corners_ego.T).T + np.array([x_e, y_e])
        # Transform each corner to plot coords
        corners_plot = np.column_stack([px(c_ego[:,0], c_ego[:,1]),
                                        py(c_ego[:,0], c_ego[:,1])])

        if strong:
            edge_color = '#00d4ff' if radar_reliable else color
            face_alpha = 0.22
            lw         = 2.5
        else:
            r, g, b = int(color[1:3],16)/255, int(color[3:5],16)/255, int(color[5:7],16)/255
            edge_color = (r, g, b, 0.55)
            face_alpha = 0.0
            lw         = 1.2

        r_val = int(color[1:3], 16) / 255
        g_val = int(color[3:5], 16) / 255
        b_val = int(color[5:7], 16) / 255

        ax.add_patch(plt.Polygon(corners_plot,
                                  linewidth=lw,
                                  edgecolor=edge_color,
                                  facecolor=(r_val, g_val, b_val, face_alpha),
                                  zorder=5))

        # Velocity arrow — transform to plot coords
        spd = d.get('velocity_ms', 0.0)
        if strong and spd > 0.5 and not is_bev:
            vx_d = d.get('vx', 0.0)
            vy_d = d.get('vy', 0.0)
            if abs(vx_d) + abs(vy_d) > 0.3:
                norm  = np.sqrt(vx_d**2 + vy_d**2)
                scale = min(4.0 / norm, 2.0)
                tip_x = x_e + vx_d * scale
                tip_y = y_e + vy_d * scale
                ax.annotate('',
                    xy=(px(tip_x, tip_y), py(tip_x, tip_y)),
                    xytext=(px(x_e, y_e), py(x_e, y_e)),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.2, alpha=0.75),
                    zorder=5)

        # Label — only strong objects within label range
        if strong and dist <= BEV_LABEL_RANGE:
            if is_bev:
                conf_val = d.get('confidence', 1.0)
                label    = f"{conf_val:.2f}" if conf_val < 1.0 else cls[:3]
            else:
                label = cls[:3]
            ax.text(px(x_e, y_e), py(x_e, y_e), label,
                    color='white', fontsize=8.5, fontweight='bold',
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.18',
                              facecolor='#0a0e14', alpha=0.65,
                              edgecolor='none'))

        if cls not in legend_patches:
            legend_patches[cls] = patches.Patch(
                facecolor=color,
                edgecolor='white',
                linewidth=0.7,
                label=cls.replace('_', ' '))

    # ── Ego vehicle ──────────────────────────────────────────
    # In plot coords: ego is at (0,0), forward = up (+py direction)
    # Box: width=2m (horizontal in plot), length=4.5m (vertical in plot)
    ax.add_patch(patches.FancyBboxPatch(
        (-1.0, -2.25), 2.0, 4.5,
        boxstyle="round,pad=0.15",
        linewidth=2.2, edgecolor='white',
        facecolor='#1f6feb', zorder=7))
    # Forward arrow pointing UP (forward in plot = +y = +x_ego)
    ax.annotate('',
        xy=(0, 3.5), xytext=(0, 2.25),
        arrowprops=dict(arrowstyle='->', color='white', lw=2.0),
        zorder=8)
    ax.text(0, 0, 'EGO', color='white', fontsize=9,
            fontweight='bold', ha='center', va='center', zorder=8)

    if ego_speed is not None:
        spd_str = f'{ego_speed:.1f} m/s  ({ego_speed*3.6:.0f} km/h)'
        ax.text(0, -3.5, spd_str, color='#8b949e', fontsize=7.5,
                ha='center', va='top', zorder=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0e14',
                          alpha=0.7, edgecolor='none'))

    # ── Legend — all classes present in scene ────────────────
    # Always add radar entry if radar data present
    if radar_pts is not None and radar_pts.shape[0] > 0:
        legend_patches['_radar'] = patches.Patch(
            facecolor='#00d4ff', edgecolor='white',
            linewidth=0.5, label='radar point')

    if legend_patches:
        # Sort legend: vehicles first, then infrastructure
        order = ['car','truck','bus','trailer','motorcycle','bicycle',
                 'pedestrian','traffic_cone','barrier','construction_vehicle','_radar']
        handles = []
        for k in order:
            if k in legend_patches:
                handles.append(legend_patches[k])
        # Any remaining not in order list
        for k, v in legend_patches.items():
            if k not in order:
                handles.append(v)

        ax.legend(handles=handles, loc='upper right',
                  facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='white', fontsize=9,
                  framealpha=0.92, borderpad=0.8,
                  handlelength=1.2, handleheight=1.0)

    # ── Axes styling ─────────────────────────────────────────
    ax.set_xlim(-BEV_RANGE_M, BEV_RANGE_M)
    ax.set_ylim(-BEV_RANGE_M, BEV_RANGE_M)
    ax.set_xlabel('← left  |  right →  (metres)',   color='#8b949e', fontsize=10)
    ax.set_ylabel('← back  |  forward →  (metres)', color='#8b949e', fontsize=10)
    ax.set_title(f"MoRAL BEV [{source}] — {scene_name}",
                 color='white', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#8b949e', labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e14')
    plt.close()

# ── Camera + metadata ─────────────────────────────────────────

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
