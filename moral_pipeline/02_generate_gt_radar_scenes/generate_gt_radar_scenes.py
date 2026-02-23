#!/usr/bin/env python3
"""
moral_pipeline/02_generate_gt_radar_scenes/generate_gt_radar_scenes.py
=======================================================================
Step 1b of MoRAL pipeline — Condition D (GT + Radar).

Extends 01_generate_gt_scenes by adding Doppler-measured radar velocities
from nuScenes' 5 radar sensors (RADAR_FRONT, RADAR_FRONT_LEFT, etc.).

Key difference from 01_gt_annotations:
  - Each detection gains radar_velocity_ms, radar_vx, radar_vy, radar_rcs
  - radar_velocity_confirmed=True when a radar point matches within 3m
  - BEV map shows cyan radar returns + Doppler velocity arrows
  - Scene description says "radar-confirmed X m/s" instead of "moving at X m/s"
  - source field becomes "ground_truth_radar"

This is a genuine novelty claim: radar Doppler velocity is PHYSICALLY MEASURED,
not estimated from consecutive frames like GT velocity annotations.

USAGE
-----
# All 10 mini scenes
python generate_gt_radar_scenes.py \\
    --dataroot ../data/nuscenes \\
    --out-dir ../outputs/02_gt_with_radar

# Quick 3-scene test
python generate_gt_radar_scenes.py \\
    --dataroot ../data/nuscenes \\
    --out-dir ../outputs/02_gt_with_radar \\
    --max-scenes 3

# Only front radar (faster, less coverage)
python generate_gt_radar_scenes.py \\
    --dataroot ../data/nuscenes \\
    --out-dir ../outputs/02_gt_with_radar \\
    --radar-sensors RADAR_FRONT

OUTPUT PER SCENE
----------------
outputs/02_gt_with_radar/
  scene-0061/
    bev_map.png            ← BEV with cyan radar dots + Doppler arrows
    CAM_FRONT.jpg          ← same cameras as condition 01
    ...
    detections.json        ← GT detections + radar_velocity_* fields
    scene_description.txt  ← "radar-confirmed X m/s" for moving objects
    metadata.json          ← includes radar_summary stats
    radar_points.npy       ← raw radar ego-frame points (N, 6) for debugging
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.scene_utils import (
    get_ego_pose, get_ego_speed, get_lidar_points,
    get_detections_from_gt, get_radar_points_ego, enrich_detections_with_radar,
    make_scene_text, save_bev_map, copy_camera_images, save_metadata,
    RADAR_SENSORS,
)


def process_sample(nusc, scene, sample, dataroot, out_dir,
                   cameras, radar_sensors, match_radius_m, sample_idx=0):

    if sample_idx == 0:
        scene_out = os.path.join(out_dir, scene['name'])
    else:
        scene_out = os.path.join(out_dir, f"{scene['name']}_frame{sample_idx:03d}")
    os.makedirs(scene_out, exist_ok=True)

    # ── Ego state ──────────────────────────────────────────
    ego_pose  = get_ego_pose(nusc, sample)
    ego_speed = get_ego_speed(nusc, sample)

    # ── LiDAR ──────────────────────────────────────────────
    points = get_lidar_points(nusc, sample, dataroot)

    # ── Radar ──────────────────────────────────────────────
    radar_pts = get_radar_points_ego(nusc, sample, dataroot, sensors=radar_sensors)

    # ── GT detections ──────────────────────────────────────
    detections = get_detections_from_gt(nusc, sample, ego_pose)

    # ── Enrich with radar ──────────────────────────────────
    detections = enrich_detections_with_radar(
        detections, radar_pts, match_radius_m=match_radius_m)

    # ── Scene text ─────────────────────────────────────────
    scene_text = make_scene_text(detections, ego_speed=ego_speed, include_velocity=True)

    # ── Radar summary stats (for metadata) ─────────────────
    n_confirmed = sum(1 for d in detections if d.get('radar_velocity_confirmed'))
    radar_summary = {
        'total_radar_points': int(radar_pts.shape[0]),
        'detections_with_radar': n_confirmed,
        'detections_total': len(detections),
        'match_radius_m': match_radius_m,
        'sensors_used': radar_sensors,
    }

    # ── Save outputs ───────────────────────────────────────
    # 1. detections.json (enriched)
    with open(os.path.join(scene_out, 'detections.json'), 'w') as f:
        json.dump(detections, f, indent=2)

    # 2. scene_description.txt
    with open(os.path.join(scene_out, 'scene_description.txt'), 'w') as f:
        f.write(scene_text)

    # 3. bev_map.png — with radar overlay
    save_bev_map(
        points, detections,
        os.path.join(scene_out, 'bev_map.png'),
        scene_name=scene['name'],
        source='GT+Radar',
        radar_pts=radar_pts if radar_pts.shape[0] > 0 else None,
    )

    # 4. Camera images
    cam_list = cameras if cameras else None
    copy_camera_images(nusc, sample, dataroot, scene_out, cameras=cam_list)

    # 5. metadata.json
    save_metadata(scene, sample, ego_pose, ego_speed, scene_out,
                  radar_summary=radar_summary)

    # 6. radar_points.npy — raw ego-frame radar for debugging / future use
    np.save(os.path.join(scene_out, 'radar_points.npy'), radar_pts)

    return scene_out, len(detections), n_confirmed, int(radar_pts.shape[0])


def process_scene(nusc, scene, dataroot, out_dir, cameras,
                  radar_sensors, match_radius_m, all_frames=False):
    results = []

    if not all_frames:
        sample = nusc.get('sample', scene['first_sample_token'])
        r = process_sample(nusc, scene, sample, dataroot, out_dir,
                           cameras, radar_sensors, match_radius_m, 0)
        results.append(r)
    else:
        token = scene['first_sample_token']
        idx = 0
        while token != '':
            sample = nusc.get('sample', token)
            r = process_sample(nusc, scene, sample, dataroot, out_dir,
                               cameras, radar_sensors, match_radius_m, idx)
            results.append(r)
            token = sample['next']
            idx += 1

    return results


def main():
    ap = argparse.ArgumentParser(
        description='MoRAL Step 1b: Generate GT + Radar scenes (Condition D)'
    )
    ap.add_argument('--dataroot',      required=True,
                    help='Path to nuScenes dataset root')
    ap.add_argument('--out-dir',       default='../outputs/02_gt_with_radar',
                    help='Output directory')
    ap.add_argument('--version',       default='v1.0-mini',
                    choices=['v1.0-mini', 'v1.0-trainval'])
    ap.add_argument('--max-scenes',    type=int, default=-1,
                    help='Max scenes (-1 = all)')
    ap.add_argument('--cameras',       nargs='+', default=None,
                    choices=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                             'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'])
    ap.add_argument('--radar-sensors', nargs='+', default=None,
                    choices=RADAR_SENSORS,
                    help='Which radar sensors to use (default: all 5)')
    ap.add_argument('--match-radius',  type=float, default=3.0,
                    help='Max distance (m) to match radar point to detection (default: 3.0)')
    ap.add_argument('--all-frames',    action='store_true',
                    help='Process all frames (default: first frame only)')
    args = ap.parse_args()

    radar_sensors = args.radar_sensors or RADAR_SENSORS

    print(f"\nLoading nuScenes {args.version} from {args.dataroot}...")
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    except Exception as e:
        print(f"ERROR loading nuScenes: {e}")
        sys.exit(1)

    scenes = nusc.scene
    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Processing {len(scenes)} scene(s)")
    print(f"Output:         {args.out_dir}")
    print(f"Cameras:        {args.cameras or 'all 6'}")
    print(f"Radar sensors:  {radar_sensors}")
    print(f"Match radius:   {args.match_radius}m")
    print(f"Frames:         {'all' if args.all_frames else 'first only'}\n")

    success, failed = 0, 0
    total_frames, total_radar_confirmed = 0, 0

    for scene in tqdm(scenes, desc="Scenes"):
        try:
            results = process_scene(
                nusc, scene,
                dataroot=args.dataroot,
                out_dir=args.out_dir,
                cameras=args.cameras,
                radar_sensors=radar_sensors,
                match_radius_m=args.match_radius,
                all_frames=args.all_frames,
            )
            success += 1
            total_frames += len(results)

            # results: list of (scene_out, n_det, n_confirmed, n_radar_pts)
            avg_det  = sum(r[1] for r in results) / len(results)
            avg_conf = sum(r[2] for r in results) / len(results)
            avg_rpts = sum(r[3] for r in results) / len(results)
            total_radar_confirmed += sum(r[2] for r in results)

            tqdm.write(
                f"  ✓ {scene['name']} — {len(results)} frame(s), "
                f"avg {avg_det:.0f} detections, "
                f"{avg_conf:.0f} radar-confirmed, "
                f"{avg_rpts:.0f} radar points"
            )
        except Exception as e:
            tqdm.write(f"  ✗ {scene['name']}: {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"  Complete")
    print(f"  Scenes:  {success} succeeded, {failed} failed")
    print(f"  Frames:  {total_frames} total")
    print(f"  Radar-confirmed detections: {total_radar_confirmed} total")
    print(f"  Output:  {args.out_dir}")
    print(f"\n  Each folder contains:")
    print(f"    bev_map.png            ← BEV with cyan radar + velocity arrows")
    print(f"    detections.json        ← GT + radar_velocity_* fields")
    print(f"    scene_description.txt  ← radar-confirmed velocities")
    print(f"    radar_points.npy       ← raw ego-frame radar (N,6)")
    print(f"    metadata.json          ← includes radar_summary")
    print(f"{'='*55}\n")

    config = {
        'dataroot':      args.dataroot,
        'version':       args.version,
        'max_scenes':    args.max_scenes,
        'cameras':       args.cameras,
        'radar_sensors': radar_sensors,
        'match_radius_m': args.match_radius,
        'all_frames':    args.all_frames,
        'scenes_processed': success,
        'frames_generated': total_frames,
        'radar_confirmed_total': total_radar_confirmed,
    }
    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved to {args.out_dir}/run_config.json")


if __name__ == '__main__':
    main()
