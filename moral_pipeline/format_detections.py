"""
format_detections.py
────────────────────
Formats detections.json into structured text for the img+det prompt condition.
Handles both conditionB (no radar fields) and conditionD (radar fields present).

Usage:
    from format_detections import format_detections_text
    text = format_detections_text(detections, condition='B', ego_speed=4.36)
"""

import math

# Objects beyond this range omitted from prompt (matches BEV_RANGE_M)
MAX_RANGE_M = 50.0
# TTC warning threshold
TTC_WARN_S  = 5.0


def _bearing_label(bearing_deg):
    """Convert bearing_deg to human-readable direction label."""
    b = bearing_deg % 360
    # bearing_deg in scene_utils is atan2(y,x) in ego frame
    # 0° = right, 90° = behind-left, -90° = front-right
    # Remap to compass-style relative to forward (ego +x = forward)
    # bearing_deg here is from ego frame: 0=front, positive=left, negative=right
    # Actually from detections: bearing_deg = degrees(atan2(y, x))
    # ego frame: x=forward, y=left
    # so: 0° = forward, 90° = left, ±180° = behind, -90° = right
    if -22.5 <= bearing_deg <= 22.5:
        return "ahead"
    elif 22.5 < bearing_deg <= 67.5:
        return "front-left"
    elif 67.5 < bearing_deg <= 112.5:
        return "left"
    elif 112.5 < bearing_deg <= 157.5:
        return "rear-left"
    elif bearing_deg > 157.5 or bearing_deg < -157.5:
        return "behind"
    elif -157.5 <= bearing_deg < -112.5:
        return "rear-right"
    elif -112.5 <= bearing_deg < -67.5:
        return "right"
    elif -67.5 <= bearing_deg < -22.5:
        return "front-right"
    return "nearby"


def _ttc(distance_m, velocity_ms, ego_speed=0.0):
    """
    Compute TTC in seconds.
    For objects ahead (bearing ~ 0), relative velocity = obj_vel + ego_vel (closing).
    Returns None if not applicable (moving away or stationary).
    """
    if velocity_ms is None or distance_m <= 0:
        return None
    # Simplified: use object velocity magnitude + ego speed as closing speed
    closing = velocity_ms + (ego_speed or 0.0)
    if closing < 0.5:
        return None
    return round(distance_m / closing, 1)


def format_detections_text(detections, condition='B', ego_speed=None, max_range=MAX_RANGE_M):
    """
    Returns structured detection string for injection into VLM prompt.

    conditionB: uses velocity_ms from LiDAR-derived GT
    conditionD: prefers radar_velocity_ms when radar_quality == 'reliable'
    """
    # Filter and sort by distance
    dets = [d for d in detections if d.get('distance_m', 999) <= max_range]
    dets.sort(key=lambda d: d['distance_m'])

    if not dets:
        return "No objects detected within sensor range."

    lines = ["Detected objects (sorted by distance from ego vehicle):"]

    for d in dets:
        cls      = d['class']
        dist     = d['distance_m']
        bearing  = d.get('bearing_deg', 0.0)
        vis      = d.get('visibility', 'unknown')
        direction = _bearing_label(bearing)

        # Velocity: prefer confirmed radar for conditionD
        if condition == 'D':
            radar_q = d.get('radar_quality', 'unconfirmed')
            if radar_q == 'reliable' and d.get('radar_velocity_ms') is not None:
                vel   = d['radar_velocity_ms']
                vsrc  = 'radar-confirmed'
            else:
                vel   = d.get('velocity_ms', 0.0)
                vsrc  = 'lidar-est'
        else:
            vel  = d.get('velocity_ms', 0.0)
            vsrc = 'lidar-est'

        # TTC for objects ahead
        ttc = None
        if abs(bearing) < 45:  # roughly ahead
            ttc = _ttc(dist, vel, ego_speed)

        # Compose line
        vel_str = f"{vel:.1f} m/s ({vsrc})"
        if vel < 0.2:
            vel_str = "stationary"
        
        line = f"  - {cls.upper()} | {direction} | {dist:.1f}m | {vel_str} | {vis}"
        
        if ttc is not None:
            warn = " ⚠ CRITICAL" if ttc < TTC_WARN_S else ""
            line += f" | TTC={ttc}s{warn}"

        # Radar quality note for conditionD
        if condition == 'D':
            rq = d.get('radar_quality', 'unconfirmed')
            if rq == 'reliable':
                line += " | radar:reliable"
            elif rq == 'radial_ambiguous':
                line += " | radar:radial-only"

        lines.append(line)

    # Summary stats
    n_moving   = sum(1 for d in dets if d.get('velocity_ms', 0) > 0.5)
    n_critical = sum(1 for d in dets
                     if abs(d.get('bearing_deg', 90)) < 45
                     and _ttc(d['distance_m'], d.get('velocity_ms', 0), ego_speed or 0) is not None
                     and (_ttc(d['distance_m'], d.get('velocity_ms', 0), ego_speed or 0) or 999) < TTC_WARN_S)

    lines.append(f"\nSummary: {len(dets)} objects within {max_range:.0f}m | "
                 f"{n_moving} moving | {n_critical} with TTC < {TTC_WARN_S}s")

    if condition == 'D':
        n_radar = sum(1 for d in dets if d.get('radar_quality') == 'reliable')
        lines.append(f"Radar: {n_radar} objects with confirmed Doppler velocity")

    return '\n'.join(lines)


def inject_detections_into_record(record, detections, condition='B', ego_speed=None):
    """
    Takes a JSONL record and returns a modified copy with detection text
    appended to the user text content.
    Used by evaluate_zeroshot.py to build img+det input condition.
    """
    import copy
    rec = copy.deepcopy(record)
    det_text = format_detections_text(detections, condition=condition, ego_speed=ego_speed)

    # Find the text part in user message content
    for part in rec['messages'][0]['content']:
        if part['type'] == 'text':
            part['text'] = det_text + '\n\n' + part['text']
            break

    return rec


if __name__ == '__main__':
    import json, sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/01_gt_annotations/scene-0001/detections.json'
    cond = sys.argv[2] if len(sys.argv) > 2 else 'B'
    with open(path) as f:
        dets = json.load(f)
    print(format_detections_text(dets, condition=cond, ego_speed=4.36))
