# MegaPose-GSO local 70 GiB training

Use this profile on the local RTX PRO 6000 Blackwell with a 70 GiB PyTorch
allocator cap:

```text
train=train_megapose_gso_finetune_rtxpro6000_blackwell_70gb_336x252_dynamic
data=megapose_gso_anchor_scene_pairs_multival
```

It trains at fixed 336x252 resolution with dynamic N=2-16 references and
K=1-10 queries. Training and all validation loaders use strict `anchor_pair`
sampling with no repeated frames. The complete local index supports every
total view count from 3 through 26 for all 755 training objects. All 189
held-out objects support the N=5/K=1, N=16/K=1, N=5/K=5, and N=5/K=10
validation protocols.

## Measured budgets

The measurements below used the complete preprocessed dataset at
`/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed`, BF16, visibility
mask conditioning, and `PI3_CUDA_MEMORY_LIMIT_GIB=70`.

| Probe | Actual images/rank | Allocated | Reserved | Result |
| --- | ---: | ---: | ---: | --- |
| packed minimum, 52 x 3 views | 156 | 66,574 MiB | 66,984 MiB | passed 3 optimizer steps |
| logical maximum, 6 x 26 views | 156 | 66,575 MiB | 67,002 MiB | passed 3 optimizer steps |
| capacity edge, 56 x 3 views | 168 | 71,001 MiB | 71,424 MiB | passed, only 256 MiB reserved headroom |

The profile therefore uses `train.max_img_per_gpu=156`. A budget of 168 is a
measured capacity edge, not a safe long-run setting.

Validation uses a budget of 768, producing 768, 765, 760, and 765 actual
images/rank for N=5/K=1, N=16/K=1, N=5/K=5, and N=5/K=10 respectively. A
one-batch-per-loader smoke with camera/correspondence metrics, visuals, two
workers, prefetch factor two, and non-persistent workers measured:

- 21,342 MiB maximum allocated GPU memory;
- 26,322 MiB maximum reserved GPU memory;
- 46,648 MiB peak host cgroup memory, including a 3,796 MiB pre-run baseline;
- 42,852 MiB host-memory growth attributable to the run.

Two validation workers are therefore safe on the local 125 GiB usable RAM.
Workers remain non-persistent so the four loaders do not retain eight workers
and their prefetched batches simultaneously. Metrics run every validation
epoch; visual diagnostics run every five epochs because visualization of the
large K=5/K=10 batches was much slower than their forward and metric passes.

## Training command

From the repository root:

```bash
gso_run_name=megapose_gso_blackwell70_336_dynamic_$(date +%Y%m%d_%H%M%S)
gso_run_dir=/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/$gso_run_name

PI3_CUDA_MEMORY_LIMIT_GIB=70 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_megapose_gso_finetune_rtxpro6000_blackwell_70gb_336x252_dynamic \
  data=megapose_gso_anchor_scene_pairs_multival \
  gso.data_root=/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed \
  model.ckpt=ckpts/Pi3/model.safetensors \
  name="$gso_run_name" \
  log.output_dir="$gso_run_dir" \
  log.tensorboard_dir="$gso_run_dir/tensorboard" \
  log.ckpt_dir="$gso_run_dir/ckpts" \
  hydra.run.dir="$gso_run_dir"
```

Restore the same `gso_run_name` and add `train.resume=auto` to continue an
interrupted run. The allocator cap is environmental and must be set on every
launch; the YAML profile cannot impose it itself.

## Repeated-instance-aware render/scene-pair baseline

The ready baseline requested after the clean render bank was added is:

```text
train=train_megapose_gso_baseline_rtxpro6000_blackwell_70gb_336x252
data=megapose_gso_baseline_render_scene_pair_lmo_val
```

Training uses homogeneous 50/50 batches from two protocols:

- `GSORenderToScene`: random clean black-background renders are references;
  the query is a full-RGB scene track;
- `GSOScenePairMaskedReferences`: references are object-only RGB from one
  scene track; queries are full RGB from another scene.

Both use dynamic N=2-16 and K=1-10 with no repeated frames. Repeated-object
scenes are retained. A query supplies its target-instance visibility mask to
the model only when that scene contains more than one track/instance of the
same object ID. All other queries, clean renders, and scene references supply
`[mask=0, known=0]`; scene references are already object-only in RGB. Query RGB
is never masked by this condition. The object mask is retained independently
as GT metadata and restricts depth/point supervision to the target object.

The model uses the established two-channel 14x14 mask patch projection
(402,432 parameters at decoder dimension 1024), alpha `1.0`, main decoder/head
LR `5e-6`, and mask-projection LR `5e-5`. Dataset construction rejects a
full-RGB repeated-instance source without an explicit disambiguation treatment,
and dataloader construction rejects a config that supplies necessary masks
while disabling the model branch.

The named validations are:

| Loader | Data | Protocol | Samples | Views/sample |
| --- | --- | --- | ---: | ---: |
| `gso_heldout_render_n5_k1` | strict held-out GSO objects and scenes | render-to-scene, uniform sphere coverage | 19,093 tracks | 5+1 |
| `gso_heldout_render_n16_k1` | strict held-out GSO objects and scenes | render-to-scene, uniform sphere coverage | 19,093 tracks | 16+1 |
| `lmo_bop19_render_n5_k1` | complete official LM-O BOP19 targets | render-to-scene | 1,444 target instances | 5+1 |

The GSO lengths enumerate every eligible object/scene track once, rather than
one random track per object. Consequently, full validation is intentionally
substantial: at the 768-image budget it has 150, 425, and 12 batches. Set each
`val_datasets.<name>.runtime.iters_per_test=1` only for a smoke; production
leaves all three at zero, meaning no iteration cap.

GSO and LM-O have separate namespace-routed CAD metrics and query-pose overlay
visualizers. GSO reports aggregate ADD at 0.1 diameter plus the shared
ADD(-S), rotation, and translation summaries without per-object scalar spam.
LM-O applies ADD-S only to configured symmetric objects 10 and 11. Both pose
alignments use reference frames only; query GT never participates in Sim(3).

The conditioned full-model smoke on 2026-08-07 completed four optimizer steps
and one packed batch from every validation loader in 2m35s, including visuals.
Training packed 144-154 images/rank and peaked at 65,837 MiB allocated / 66,320
MiB reserved under the 70 GiB cap. The existing two-worker validation staging
measurement remains about 14.8 GiB host RSS, well below the machine's 125 GiB
usable RAM. The artifact is outside Git at
`/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_repeated_mask_70gb_smoke_20260807_160450`.

The first held-out N=5/K=1 validation batch contained two ambiguous queries out
of 128: `visibility_mask_query_known_area=0.015625`, while reference known area
was exactly zero. A separate two-step forced train smoke on repeated object 0,
scene 2155 showed query known area `1.0`; after the first backward/update, the
second forward reported a nonzero mask-token norm and projection-weight norm.
Its training portion is at
`/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_repeated_mask_forced_smoke_20260807_160821`;
it was intentionally stopped when redundant full validation began. The active
catalogues contain 3,149 repeated object-scene pairs in train and 186 in the
strict held-out validation split.

Launch 30 epochs locally from the repository root:

```bash
RUN=megapose_gso_baseline_336_$(date +%Y%m%d_%H%M%S)
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 \
PI3_CUDA_MEMORY_LIMIT_GIB=70 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
HYDRA_FULL_ERROR=1 \
nohup /home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python -u scripts/train_pi3.py \
  train=train_megapose_gso_baseline_rtxpro6000_blackwell_70gb_336x252 \
  data=megapose_gso_baseline_render_scene_pair_lmo_val \
  train.num_epoch=80 \
  name="$RUN" \
  gso.data_root=/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed \
  gso.assets_root=/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets \
  gso.references_root=/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/renders \
  lmo.data_root=/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o \
  model.ckpt=/home/dtrofimov/repositories/Pi3/ckpts/Pi3/model.safetensors \
  log.output_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN" \
  log.tensorboard_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"/tensorboard \
  log.ckpt_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"/ckpts \
  hydra.run.dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN" \
  log.use_tensorboard=true \
  log.ckpt_interval=5 \
  log.max_checkpoints=3 \
  > logs/"$RUN".log 2>&1 &
```
