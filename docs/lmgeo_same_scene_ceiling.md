# LMGeo Same-Scene Ceiling Eval

Date: 2026-07-29

This is an eval-only diagnostic.  For each object/window, one visible same-scene
frame is held out as the query and the remaining visible frames for that object
become keyframes.  The existing object-pose metric still estimates the Sim(3)
alignment from references only and reports ADD/ADD-S on query views only, so the
held-out query does not leak into alignment.

## Configs

- Data: `configs/data/lmgeo_same_scene_ceiling.yaml`
- A40 eval profile: `configs/train/train_lmgeo_eval_only_a40_40gb_same_scene_ceiling.yaml`
- Local RTX PRO 6000 profile: `configs/train/train_lmgeo_eval_only_rtxpro6000_70gb_same_scene_ceiling.yaml`

Key settings:

- `train.eval_only: true`
- `train/test.image_num_range: [25, 25]`
- `max_img_per_gpu: 25`
- `lmgeo_ceiling.max_reference: 24`
- `lmgeo_ceiling.query_frame_strategy: random`
- RGB references and queries are unmasked.
- Depth/valid masks remain object-masked for object-frame loss/Chamfer panels.

Current split sizes at 560x420 with `filter_preprocessed_query_depth=false`:

- `real_same_scene_ceiling`: 303 samples, reference range 2-11
- `pbr_same_scene_ceiling`: 1600 samples, reference range 6-24

Real LM-O test windows do not usually contain 24 usable same-object frames after
visibility filtering; held-out PBR does.

## Local Run

```bash
cd /home/dtrofimov/repositories/Pi3

RUN=lmgeo_rtx6000_70gb_same_scene_ceiling_eval_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 \
PI3_CUDA_MEMORY_LIMIT_GIB=70 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
HYDRA_FULL_ERROR=1 \
nohup /home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python -u scripts/train_pi3.py \
  train=train_lmgeo_eval_only_rtxpro6000_70gb_same_scene_ceiling \
  data=lmgeo_same_scene_ceiling \
  name=$RUN \
  model.ckpt=ckpts/Pi3/model.safetensors \
  log.output_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/$RUN \
  log.ckpt_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/$RUN/ckpts \
  hydra.run.dir=/media/internal/nvme/dtrofimov/spott3r/outputs/$RUN \
  log.use_tensorboard=true \
  log.save_best=false \
  log.save_checkpoints=false \
  > logs/${RUN}.log 2>&1 &
```

TensorBoard:

```bash
tensorboard --logdir /media/internal/nvme/dtrofimov/spott3r/outputs
```

## A40 Slurm Run

Use one A40 for the cleanest TensorBoard scalar metrics.  The raw prediction
parquet can gather rows in multi-rank jobs, but the online plugin metric scalars
are easiest to trust in a single-rank eval.

```bash
cd /homes/dtrofimov/repositories/Pi3

PI3_CUDA_MEMORY_LIMIT_GIB=40 \
PI3_FILTER_PREPROCESSED_QUERY_DEPTH=false \
PI3_TRAIN_GPUS=1 \
PI3_TRAIN_CONFIG=train_lmgeo_eval_only_a40_40gb_same_scene_ceiling \
PI3_DATA_CONFIG=lmgeo_same_scene_ceiling \
PI3_RUN_NAME=lmgeo_a40_same_scene_ceiling_eval_$(date +%Y%m%d_%H%M%S) \
PI3_CKPT=/homes/dtrofimov/repositories/Pi3/ckpts/model.safetensors \
PI3_WALLTIME=04:00:00 \
PI3_CKPT_INTERVAL=999 \
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train
```

The eval-only trainer branch exits after one validation pass and does not save
checkpoints.

## Smoke Checks Run

Commands passed on the local RTX PRO 6000 with `PI3_CUDA_MEMORY_LIMIT_GIB=40`:

- Worst-case PBR 24 refs + 1 query, one sample, visuals/metrics on:
  `lmgeo_same_scene_ceiling_pbr25_smoke_20260729_155007`
- Broad real + PBR config path, one sample per split, visuals/metrics on:
  `lmgeo_same_scene_ceiling_two_split_smoke_20260729_155135`
- PBR timing, 10 samples, workers=4, visuals off:
  `lmgeo_same_scene_ceiling_pbr_timing_20260729_155214`

Observed memory:

- Worst-case 24+1 eval: 5817 MiB allocated / 6086 MiB reserved
- Two-split smoke: up to 5454 MiB allocated / 5698 MiB reserved
- 10-sample PBR timing: up to 6137 MiB allocated / 6654 MiB reserved

Timing:

- 10-sample PBR timing averaged about 1.03 s/sample after setup on the local
  RTX PRO 6000 Blackwell.
- Full local eval is roughly 30-45 minutes.
- One A40 should be planned as roughly 45-90 minutes; use `PI3_WALLTIME=04:00:00`
  for queue safety.

Exported example TensorBoard images:

- `examples/lmgeo_same_scene_ceiling/`
- `examples/lmgeo_same_scene_ceiling/two_split_smoke/`

Validation:

```bash
python scripts/train_pi3.py --cfg job \
  train=train_lmgeo_eval_only_a40_40gb_same_scene_ceiling \
  data=lmgeo_same_scene_ceiling

python -m unittest discover -s tests -v
```

The CPU test suite passed: 66 tests OK.  `pyarrow` is installed in the local
environment, so `predictions.parquet` writing is available.
