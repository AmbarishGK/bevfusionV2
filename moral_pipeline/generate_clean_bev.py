"""
generate_clean_bev.py
─────────────────────
Generates "clean" BEV maps — LiDAR + radar point cloud only, NO GT boxes drawn.

Purpose:
  Feed these to a zero-shot or fine-tuned VLM. The model must detect objects
  from point cloud density alone. Compare model output vs detections.json to
  find GT annotation gaps (objects model finds that GT missed).

Output per scene:
  outputs/03_clean_bev/scene-XXXX/
    bev_lidar_only.png        ← LiDAR point cloud, no boxes
    bev_lidar_radar.png       ← LiDAR + radar dots, no boxes
    metadata.json             ← copied from conditionB

Usage:
  # Test 5 scenes (auto-detects nuScenes path):
  python generate_clean_bev.py --max_scenes 5

  # Explicit paths:
  python generate_clean_bev.py \\
      --nusc_root ../data \\
      --gt_root   outputs/01_gt_annotations \\
      --out_dir   outputs/03_clean_bev \\
      --max_scenes 5

  # Single scene:
  python generate_clean_bev.py --scene scene-0001
"""

import os
import sys
import json
import argparse
import shutil

import numpy as np

# ── Path setup ────────────────────────────────────────────────
# scene_utils.py lives in the same directory as this script
# (you should 'cp utils/scene_utils.py .' before running)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from scene_utils import save_bev_map, get_lidar_points, get_radar_points_ego, get_ego_pose
except ModuleNotFoundError:
    print("ERROR: scene_utils.py not found in current directory.")
    print("Fix:  cp utils/scene_utils.py .")
    sys.exit(1)


# ── nuScenes path auto-detection ──────────────────────────────
CANDIDATE_ROOTS = [
    '../data',
    '../../data',
    '/data/nuscenes',
    '/data',
    os.path.expanduser('~/data/nuscenes'),
]

def find_nusc_root(hint=None):
    """Find nuScenes root by looking for v1.0-trainval/ directory."""
    candidates = ([hint] if hint else []) + CANDIDATE_ROOTS
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, 'v1.0-trainval')):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"nuScenes v1.0-trainval not found. Tried: {candidates}\n"
        f"Pass --nusc_root explicitly, e.g. --nusc_root ../data"
    )


def get_nuscenes(nusc_root):
    from nuscenes.nuscenes import NuScenes
    print(f"Loading nuScenes from {nusc_root} ...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root, verbose=False)
    print(f"  {len(nusc.scene)} scenes loaded")
    return nusc


def get_scene_token(nusc, scene_name):
    for s in nusc.scene:
        if s['name'] == scene_name:
            return s['token']
    raise ValueError(f"Scene '{scene_name}' not found in nuScenes")


def get_first_sample(nusc, scene_token):
    scene = nusc.get('scene', scene_token)
    return nusc.get('sample', scene['first_sample_token'])


def render_clean_bev(nusc, scene_name, nusc_root, out_dir):
    """
    Render bev_lidar_only.png and bev_lidar_radar.png for one scene.
    Returns output scene directory path.
    """
    scene_out = os.path.join(out_dir, scene_name)
    os.makedirs(scene_out, exist_ok=True)

    scene_token = get_scene_token(nusc, scene_name)
    sample      = get_first_sample(nusc, scene_token)

    lidar_pts = get_lidar_points(nusc, sample, nusc_root)
    radar_pts = get_radar_points_ego(nusc, sample, nusc_root)

    n_lidar = len(lidar_pts) if lidar_pts is not None else 0
    n_radar = radar_pts.shape[0] if (radar_pts is not None and
                                      hasattr(radar_pts, 'shape')) else 0

    # ── LiDAR only ─────────────────────────────────────────────
    save_bev_map(
        points     = lidar_pts,
        detections = [],            # ← no boxes
        out_path   = os.path.join(scene_out, 'bev_lidar_only.png'),
        scene_name = scene_name,
        source     = 'LIDAR_ONLY',
        radar_pts  = None,
    )

    # ── LiDAR + radar ──────────────────────────────────────────
    save_bev_map(
        points     = lidar_pts,
        detections = [],            # ← still no boxes
        out_path   = os.path.join(scene_out, 'bev_lidar_radar.png'),
        scene_name = scene_name,
        source     = 'LIDAR+RADAR',
        radar_pts  = radar_pts,
    )

    print(f"  {scene_name}: {n_lidar} LiDAR pts | {n_radar} radar pts → saved")
    return scene_out


def get_all_scene_names(gt_root):
    return sorted([
        d for d in os.listdir(gt_root)
        if os.path.isdir(os.path.join(gt_root, d)) and d.startswith('scene-')
    ])


def main():
    ap = argparse.ArgumentParser(
        description='Generate clean BEV maps (LiDAR/radar only, no GT boxes)'
    )
    ap.add_argument('--nusc_root',  default=None,
                    help='nuScenes dataroot (auto-detected if not set)')
    ap.add_argument('--gt_root',    default='outputs/01_gt_annotations',
                    help='conditionB output dir — scene list + metadata source')
    ap.add_argument('--out_dir',    default='outputs/03_clean_bev')
    ap.add_argument('--max_scenes', type=int, default=None)
    ap.add_argument('--scene',      default=None,
                    help='Single scene name, e.g. scene-0001')
    args = ap.parse_args()

    # Auto-detect nuScenes root
    nusc_root = find_nusc_root(args.nusc_root)
    print(f"nuScenes root: {nusc_root}")

    os.makedirs(args.out_dir, exist_ok=True)
    nusc = get_nuscenes(nusc_root)

    # Build scene list
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = get_all_scene_names(args.gt_root)
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]

    print(f"\nGenerating clean BEV maps for {len(scenes)} scenes → {args.out_dir}")
    print("  (LiDAR + radar dots only — no GT annotation boxes)\n")

    ok, failed = 0, []
    for i, scene_name in enumerate(scenes, 1):
        print(f"[{i:4d}/{len(scenes)}] ", end='', flush=True)
        try:
            scene_out = render_clean_bev(nusc, scene_name, nusc_root, args.out_dir)
            # Copy metadata for reference
            meta_src = os.path.join(args.gt_root, scene_name, 'metadata.json')
            if os.path.exists(meta_src):
                shutil.copy(meta_src, os.path.join(scene_out, 'metadata.json'))
            ok += 1
        except Exception as e:
            print(f"ERROR — {e}")
            failed.append((scene_name, str(e)))

    print(f"\n{'─'*60}")
    print(f"  Done: {ok}/{len(scenes)} scenes")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for s, e in failed[:10]:
            print(f"    {s}: {e}")
    print(f"  Output: {args.out_dir}/")


if __name__ == '__main__':
    main()
