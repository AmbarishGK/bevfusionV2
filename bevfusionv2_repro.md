# bevfusionV2 – Repro Checklist

This file is for **future you** if you lose this chat and want to get back to the same state.
It assumes you still have (or can restore) this repo from Git.

---

## 0. Make sure this work is committed

From the repo root:

```bash
git status
# If there are changes:
git add .
git commit -m "Working bevfusionV2 data pipeline (nuscenes mini, multisweep, lidar-frame GT)"
```

If you keep this repo on GitHub/GitLab/etc, push it:

```bash
git remote -v           # check remote
git push origin main    # or your branch name
```

If you do this, you **do not need this chat** to recover code – Git is the source of truth.

---

## 1. Recreate the environment with uv

From a fresh machine:

```bash
git clone <your-repo-url> bevfusionV2
cd bevfusionV2

# Create the Python 3.10 venv (if you don't already have 3.10 system-wide)
uv python pin 3.10 || true

# Create virtualenv + install deps from lockfile
uv sync --locked || uv sync
```

Check:

```bash
source .venv/bin/activate
python --version    # should be 3.10.x
uv pip list | grep torch
# torch==2.9.0, torchvision==0.24.0, torchaudio==2.9.0
```

If you accidentally initialize a new project with uv later and it sets `requires-python = ">=3.12"`,
edit `pyproject.toml` to:

```toml
[project]
requires-python = ">=3.10,<3.11"
```

and re-run:

```bash
rm -rf .venv uv.lock
uv python pin 3.10
uv sync
```

---

## 2. NuScenes mini layout

The repo expects nuScenes mini to be here:

```text
data/
  nuscenes/
    v1.0-mini/
    samples/
      CAM_*/
      LIDAR_TOP/
      RADAR_*/
    sweeps/
      CAM_*/
      LIDAR_TOP/
      RADAR_*/
    maps/
```

To restore:

1. Download nuScenes **mini** from the official website.
2. Extract everything under `data/nuscenes` so you get the structure above.

---

## 3. Regenerate processed infos (if needed)

If you change `tools/create_data.py` or delete the processed data, run:

```bash
source .venv/bin/activate

python tools/create_data.py nuscenes   --root data/nuscenes   --version v1.0-mini   --out-dir data/nuscenes_mini_bevfusion   --max-sweeps 10
```

You should see something like:

```text
Saved 323 train infos to data/nuscenes_mini_bevfusion/infos_train.pkl
Saved 81 val infos to data/nuscenes_mini_bevfusion/infos_val.pkl
```

---

## 4. What files matter (data side)

At this point, the important files (which should be in Git) are:

- `pyproject.toml` / `uv.lock`  
  – env and dependency versions

- `config/nuscenes_mini.yaml`  
  – dataset paths, geometry, classes, augment.image_mean/std, etc.

- `config/loader.py`  
  – tiny `Config` YAML loader

- `tools/create_data.py`  
  – nuScenes → `infos_train.pkl` / `infos_val.pkl` generator  
  – stores lidar path, cams, transforms, anns, sweeps, etc.

- `datasets/nuscenes_bevfusion.py`  
  – core dataset:
    - multi-sweep LiDAR aggregation (into keyframe lidar frame)
    - BEV-range filtering
    - 6-camera image loading + normalization
    - 10-class detection labels in lidar frame (`gt_boxes`, `gt_labels`)

- `main.py`  
  – simple smoke test that:
    - loads `config/nuscenes_mini.yaml`
    - builds `NuScenesBEVFusionDataset`
    - wraps in `DataLoader`
    - prints shapes

As long as these files are in Git, you can restore the project exactly.

---

## 5. Sanity check (to confirm everything works)

From repo root:

```bash
source .venv/bin/activate
python main.py
```

You should see something like:

```text
Keys: dict_keys(['sample_token', 'scene_token', 'timestamp',
                 'lidar_points', 'images', 'cams', 'anns',
                 'gt_boxes', 'gt_labels', 'lidar2ego', 'ego2global'])
lidar_points: torch.Size([1, N, 4])      # N ~ 30k, depends on filtering
images: torch.Size([1, 6, 3, 1600, 900])
gt_boxes (lidar): torch.Size([1, M, 7])  # M ~ 20–40
gt_labels: torch.Size([1, M])
```

If this runs, your data pipeline is back to the state we had before starting the model.

---

## 6. If you start a new ChatGPT session and want help from this point

If you ever lose this conversation but still have the repo, you can paste something like this to ChatGPT:

> I have a repo called `bevfusionV2` with a nuScenes mini pipeline already working.  
> The important files are:
> - `config/nuscenes_mini.yaml`
> - `config/loader.py`
> - `tools/create_data.py`
> - `datasets/nuscenes_bevfusion.py`
> - `main.py`
>  
> Data is under `data/nuscenes` and processed infos are in `data/nuscenes_mini_bevfusion`.  
>  
> `NuScenesBEVFusionDataset` already:
> - does multi-sweep lidar aggregation into keyframe lidar frame,
> - filters points and GT boxes to `point_cloud_range`,
> - maps nuScenes categories to 10 BEVFusion classes,
> - returns `lidar_points`, `images`, `gt_boxes`, `gt_labels`.  
>  
> Please help me design and implement a minimal LiDAR-only BEV backbone and detection head that fits this dataset.

That’s all you need for the assistant to reconstruct context and continue from the same conceptual point, even without this chat history.

---

## 7. TL;DR

1. **Commit and push** this repo – Git is your backup.  
2. Use `uv sync` + `python tools/create_data.py ...` to rebuild env and infos.  
3. Use `python main.py` to smoke-test the data pipeline.  
4. If starting a new chat, paste the block in section 6 so the assistant knows what you already have.
