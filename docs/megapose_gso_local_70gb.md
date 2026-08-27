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
| `gso_heldout_render_n5_k1` | strict held-out GSO objects and scenes | render-to-scene, uniform sphere coverage | 19,460 tracks | 5+1 |
| `gso_heldout_render_n16_k1` | strict held-out GSO objects and scenes | render-to-scene, uniform sphere coverage | 19,460 tracks | 16+1 |
| `lmo_bop19_render_n5_k1` | complete official LM-O BOP19 targets | render-to-scene | 1,444 target instances | 5+1 |

> **Warning: this is not a calibrated keyframe-count ablation.** The old GSO
> N=5 and N=16 loaders use the legacy full-frame render-to-scene pipeline. They
> do not match the renders' effective normalized focal length/crop to the scene
> query. The query can therefore be out of distribution in image scale and
> perspective relative to every reference. A reference-only Sim(3) cannot
> correct an intrinsics mismatch. Comparing these loaders only tests whether
> adding more *focal-inconsistent* render references helps the same OOD query;
> it must not be cited as evidence that more calibrated keyframes do or do not
> help. The current geometry-constrained focal-consistent pipeline has only
> been evaluated at N=5 and needs new N-specific plan catalogues for a valid
> reference-count sweep.

The GSO lengths enumerate every eligible object/scene track once, rather than
one random track per object. Consequently, full validation is intentionally
substantial: at the 768-image budget it has 153, 433, and 12 batches. Set each
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

Launch the historical baseline locally from the repository root (the override
extends its inherited 30 epochs to 80):

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

## Transfer-diagnostic training profile

The follow-up profile keeps the same measured 70 GiB image budgets and frozen
DINOv2 encoder, but changes the trainable Pi3 decoder/head LR from `5e-6` to
`1e-5`. The 402,432-parameter mask projection remains a separate optimizer
group at `5e-5`. Its 80 epochs, 500 steps/epoch, 1,500-step warm-up, and
validation every three epochs are part of the train config rather than launch
overrides.

Training queries use full RGB. Repeated-instance scenes always receive the
target-instance conditioning mask; every other query view receives it with
probability `0.5`. One photometric recipe is shared by all references in a
sample and an independent recipe by all queries. This augmentation is active
only in training.

`data=megapose_gso_transfer_diagnostics` exposes eight deterministic loaders:

| Loader | Purpose | Samples |
| --- | --- | ---: |
| `gso_heldout_render_n5_k1` | GSO repeated-only query conditioning | 19,460 |
| `gso_heldout_render_n16_k1` | same GSO cohort with 16 references | 19,460 |
| `gso_heldout_render_n5_k1_all_query_conditioned` | oracle mask on every GSO query | 19,460 |
| `gso_heldout_render_n5_k1_visibility_gt_0_5` | GSO `visib_fract > 0.5` exactly | 17,922 |
| `lmo_pbr_new_val_render_n5_k1` | LM-O synthetic transfer, no query condition | 1,600 |
| `lmo_pbr_new_val_render_n5_k1_all_query_conditioned` | LM-O synthetic oracle mask | 1,600 |
| `lmo_bop19_render_n5_k1` | LM-O real BOP19 transfer, no query condition | 1,444 |
| `lmo_bop19_render_n5_k1_all_query_conditioned` | LM-O real oracle mask | 1,444 |

The focal-consistency warning above also applies to the N=5 versus N=16 GSO
diagnostic pair in this profile.

The paired loaders use the same deterministic references and query records.
All LM-O diagnostic loaders deliberately have eager preprocessed-depth
filtering disabled: the native and target aspect ratios are both 4:3,
center-crop visibility is still checked, and `BaseDataset` rejects/refetches a
sample if its processed depth becomes empty. This avoids a long network-volume
scan during every job startup.

The full-model smoke on 2026-08-10 completed four bf16 optimizer steps and one
batch from all eight validation loaders with two workers. It exercised both
GSO and LM-O CAD metrics, made the initially zero mask projection nonzero, and
peaked at 12.7 GiB for the intentionally small six-image smoke batch. The
production 156-image budget itself was already measured by the baseline smoke
at about 66.3 GiB reserved; RGB photometric augmentation does not add model
activations.

Launch the production run from the repository root:

```bash
RUN=megapose_gso_transfer_diag_336_$(date +%Y%m%d_%H%M%S)
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 \
PI3_CUDA_MEMORY_LIMIT_GIB=70 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
HYDRA_FULL_ERROR=1 \
nohup /home/dtrofimov/miniconda3/envs/pi3-lmgeo/bin/python -u scripts/train_pi3.py \
  train=train_megapose_gso_transfer_rtxpro6000_blackwell_70gb_336x252 \
  data=megapose_gso_transfer_diagnostics \
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

Validation is intentionally substantial and runs every three epochs. For a
one-batch preflight, append `test.iters_per_test=1`; do not use that override
for production metrics.
