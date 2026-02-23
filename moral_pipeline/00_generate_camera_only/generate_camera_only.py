#!/usr/bin/env python3
"""
moral_pipeline/00_generate_camera_only/generate_camera_only.py
==============================================================
Step 0 of MoRAL pipeline — Condition A (Camera Only baseline).

The simplest condition: Cosmos gets ONLY the 6 camera images.
No BEV map. No detections. No scene description.

This is the hardest condition for Cosmos — it must answer spatial
questions using only what it can see in the images. Distance estimation,
object detection behind the vehicle, and occluded object reasoning
are all impossible or unreliable without LiDAR/BEV data.

Comparing score(A) vs score(C) is your core research finding.

USAGE
-----
python generate_camera_only.py --dataroot ../data/nuscenes --out-dir ../outputs/00_camera_only
python generate_camera_only.py --dataroot ../data/nuscenes --out-dir ../outputs/00_camera_only --max-scenes 3
"""

import os
import sys
import json
import shutil
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.scene_utils import get_ego_pose, get_ego_speed, save_metadata

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def process_scene(nusc, scene, dataroot, out_dir, cameras):
    scene_out = os.path.join(out_dir, scene['name'])
    os.makedirs(scene_out, exist_ok=True)

    sample   = nusc.get('sample', scene['first_sample_token'])
    ego_pose = get_ego_pose(nusc, sample)
    ego_speed = get_ego_speed(nusc, sample)

    # Copy camera images — that's ALL for this condition
    copied = 0
    for cam in cameras:
        if cam not in sample['data']:
            continue
        sd  = nusc.get('sample_data', sample['data'][cam])
        src = os.path.join(dataroot, sd['filename'])
        dst = os.path.join(scene_out, f'{cam}.jpg')
        shutil.copy2(src, dst)
        copied += 1

    # Minimal metadata — no detections, no BEV, no scene text
    meta = {
        'scene_token':    scene['token'],
        'scene_name':     scene['name'],
        'sample_token':   sample['token'],
        'timestamp':      sample['timestamp'],
        'ego_speed_ms':   ego_speed,
        'condition':      'camera_only',
        'cameras':        cameras,
        'note':           'No BEV map, no detections, no scene description. Camera images only.',
    }
    with open(os.path.join(scene_out, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return copied


def main():
    ap = argparse.ArgumentParser(
        description='MoRAL Step 0: Camera-only baseline (Condition A)'
    )
    ap.add_argument('--dataroot',   required=True)
    ap.add_argument('--out-dir',    default='../outputs/00_camera_only')
    ap.add_argument('--version',    default='v1.0-mini',
                    choices=['v1.0-mini', 'v1.0-trainval'])
    ap.add_argument('--max-scenes', type=int, default=-1)
    ap.add_argument('--cameras',    nargs='+', default=CAMERAS,
                    choices=CAMERAS)
    args = ap.parse_args()

    print(f"\nLoading nuScenes {args.version} from {args.dataroot}...")
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    except Exception as e:
        print(f"ERROR: {e}"); sys.exit(1)

    scenes = nusc.scene
    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Processing {len(scenes)} scene(s) → {args.out_dir}\n")

    success, failed = 0, 0
    for scene in tqdm(scenes, desc="Scenes"):
        try:
            n = process_scene(nusc, scene, args.dataroot, args.out_dir, args.cameras)
            tqdm.write(f"  ✓ {scene['name']} — {n} camera images")
            success += 1
        except Exception as e:
            tqdm.write(f"  ✗ {scene['name']}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Complete: {success} succeeded, {failed} failed")
    print(f"  Output:   {args.out_dir}")
    print(f"\n  Each folder contains ONLY:")
    print(f"    CAM_FRONT.jpg, CAM_BACK.jpg, etc. (6 images)")
    print(f"    metadata.json")
    print(f"  NO bev_map.png, NO detections.json, NO scene_description.txt")
    print(f"{'='*50}\n")

    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as f:
        json.dump({'condition': 'A_camera_only', 'scenes': success,
                   'cameras': args.cameras, 'version': args.version}, f, indent=2)


if __name__ == '__main__':
    main()
