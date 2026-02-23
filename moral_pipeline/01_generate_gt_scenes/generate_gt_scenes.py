#!/usr/bin/env python3
"""
01_generate_gt_scenes/generate_gt_scenes.py
============================================
Step 1 of MoRAL pipeline.
Generates BEV maps, detections, and scene descriptions from nuScenes GT.
No GPU required. Runs on your laptop.

USAGE
-----
# All 10 mini scenes
python generate_gt_scenes.py \
    --dataroot ../data/nuscenes \
    --out-dir ../outputs/gt_scenes

# First 3 scenes only (quick test)
python generate_gt_scenes.py \
    --dataroot ../data/nuscenes \
    --out-dir ../outputs/gt_scenes \
    --max-scenes 3

# Front camera only (smaller output)
python generate_gt_scenes.py \
    --dataroot ../data/nuscenes \
    --out-dir ../outputs/gt_scenes \
    --cameras CAM_FRONT

# All frames in each scene (not just first)
python generate_gt_scenes.py \
    --dataroot ../data/nuscenes \
    --out-dir ../outputs/gt_scenes \
    --all-frames

OUTPUT PER SCENE
----------------
outputs/gt_scenes/
  scene-0061/
    bev_map.png            ← BEV visualization (LiDAR + GT boxes)
    CAM_FRONT.jpg          ← Front camera image
    CAM_FRONT_LEFT.jpg     ← (if --cameras not specified)
    CAM_FRONT_RIGHT.jpg
    CAM_BACK.jpg
    CAM_BACK_LEFT.jpg
    CAM_BACK_RIGHT.jpg
    detections.json        ← Structured detections for Cosmos
    scene_description.txt  ← Natural language for Cosmos
    metadata.json          ← Scene info for reproducibility
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

# Add parent dir to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.scene_utils import (
    get_ego_pose, get_ego_speed, get_lidar_points,
    get_detections_from_gt, make_scene_text,
    save_bev_map, copy_camera_images, save_metadata
)


def process_sample(nusc, scene, sample, dataroot, out_dir, cameras, sample_idx=0):
    """Process a single nuScenes sample and save all output files."""

    # Create output directory
    if sample_idx == 0:
        scene_out = os.path.join(out_dir, scene['name'])
    else:
        scene_out = os.path.join(out_dir, f"{scene['name']}_frame{sample_idx:03d}")
    os.makedirs(scene_out, exist_ok=True)

    # ── Get ego state ──────────────────────────────────────
    ego_pose  = get_ego_pose(nusc, sample)
    ego_speed = get_ego_speed(nusc, sample)

    # ── Get LiDAR point cloud ──────────────────────────────
    points    = get_lidar_points(nusc, sample, dataroot)

    # ── Get GT detections ──────────────────────────────────
    detections = get_detections_from_gt(nusc, sample, ego_pose)

    # ── Generate scene description ─────────────────────────
    scene_text = make_scene_text(
        detections,
        ego_speed=ego_speed,
        include_velocity=True
    )

    # ── Save all outputs ───────────────────────────────────
    # 1. detections.json
    with open(os.path.join(scene_out, 'detections.json'), 'w') as f:
        json.dump(detections, f, indent=2)

    # 2. scene_description.txt
    with open(os.path.join(scene_out, 'scene_description.txt'), 'w') as f:
        f.write(scene_text)

    # 3. bev_map.png
    save_bev_map(
        points, detections,
        os.path.join(scene_out, 'bev_map.png'),
        scene_name=scene['name'],
        source='GT'
    )

    # 4. Camera images
    cam_list = cameras if cameras else None
    copy_camera_images(nusc, sample, dataroot, scene_out, cameras=cam_list)

    # 5. metadata.json
    save_metadata(scene, sample, ego_pose, ego_speed, scene_out)

    return scene_out, len(detections)


def process_scene(nusc, scene, dataroot, out_dir, cameras, all_frames=False):
    """Process one full scene — either first frame only or all frames."""
    results = []

    if not all_frames:
        # Just first sample
        sample = nusc.get('sample', scene['first_sample_token'])
        scene_out, n_det = process_sample(
            nusc, scene, sample, dataroot, out_dir, cameras, sample_idx=0)
        results.append((scene_out, n_det))
    else:
        # All samples in scene
        sample_token = scene['first_sample_token']
        idx = 0
        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            scene_out, n_det = process_sample(
                nusc, scene, sample, dataroot, out_dir, cameras, sample_idx=idx)
            results.append((scene_out, n_det))
            sample_token = sample['next']
            idx += 1

    return results


def main():
    ap = argparse.ArgumentParser(
        description='MoRAL Step 1: Generate GT scenes from nuScenes'
    )
    ap.add_argument('--dataroot',   required=True,
                    help='Path to nuScenes dataset root')
    ap.add_argument('--out-dir',    default='../outputs/gt_scenes',
                    help='Output directory (default: ../outputs/gt_scenes)')
    ap.add_argument('--version',    default='v1.0-mini',
                    choices=['v1.0-mini', 'v1.0-trainval'],
                    help='nuScenes version (default: v1.0-mini)')
    ap.add_argument('--max-scenes', type=int, default=-1,
                    help='Max scenes to process (-1 = all, default: -1)')
    ap.add_argument('--cameras',    nargs='+', default=None,
                    choices=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                    help='Which cameras to copy (default: all 6)')
    ap.add_argument('--all-frames', action='store_true',
                    help='Process all frames in each scene (default: first frame only)')
    ap.add_argument('--min-lidar-pts', type=int, default=1,
                    help='Min LiDAR points for object to be included (default: 1)')
    args = ap.parse_args()

    # ── Load nuScenes ──────────────────────────────────────
    print(f"\nLoading nuScenes {args.version} from {args.dataroot}...")
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(
            version=args.version,
            dataroot=args.dataroot,
            verbose=False
        )
    except Exception as e:
        print(f"ERROR loading nuScenes: {e}")
        sys.exit(1)

    scenes = nusc.scene
    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Processing {len(scenes)} scene(s)")
    print(f"Output:    {args.out_dir}")
    print(f"Cameras:   {args.cameras or 'all 6'}")
    print(f"Frames:    {'all' if args.all_frames else 'first only'}\n")

    # ── Process scenes ─────────────────────────────────────
    success, failed = 0, 0
    total_frames    = 0

    for scene in tqdm(scenes, desc="Scenes"):
        try:
            results = process_scene(
                nusc, scene,
                dataroot=args.dataroot,
                out_dir=args.out_dir,
                cameras=args.cameras,
                all_frames=args.all_frames
            )
            success     += 1
            total_frames += len(results)
            avg_det = sum(r[1] for r in results) / len(results)
            tqdm.write(f"  ✓ {scene['name']} — {len(results)} frame(s), "
                       f"avg {avg_det:.1f} detections")

        except Exception as e:
            tqdm.write(f"  ✗ {scene['name']}: {e}")
            failed += 1
            continue

    # ── Summary ────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Complete")
    print(f"  Scenes:  {success} succeeded, {failed} failed")
    print(f"  Frames:  {total_frames} total")
    print(f"  Output:  {args.out_dir}")
    print(f"\n  Each folder contains:")
    print(f"    bev_map.png            ← visual BEV for Cosmos")
    print(f"    CAM_FRONT.jpg          ← front camera for Cosmos")
    print(f"    detections.json        ← structured detections")
    print(f"    scene_description.txt  ← natural language context")
    print(f"    metadata.json          ← reproducibility info")
    print(f"{'='*50}\n")

    # ── Save run config for reproducibility ────────────────
    config = {
        'dataroot':     args.dataroot,
        'version':      args.version,
        'max_scenes':   args.max_scenes,
        'cameras':      args.cameras,
        'all_frames':   args.all_frames,
        'scenes_processed': success,
        'frames_generated': total_frames,
    }
    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved to {args.out_dir}/run_config.json")


if __name__ == '__main__':
    main()
