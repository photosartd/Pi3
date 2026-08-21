# MegaPose-GSO A40 training

MegaPose-GSO production uses strict `anchor_pair` sampling for both training
and validation. References come from one physical `(scene_id, gt_id)` track and
queries come from a second track of the same object in a different scene. No
frame repeats. `megapose_gso_cross_scene` deliberately remains available only
for sparse/local checks whose partial shards cannot supply long anchor tracks;
do not use it for production results.

The dataset must already contain:

```text
pi3_index/megapose_gso.sqlite
pi3_index/megapose_gso.splits.json
```

Changing training resolution does not require preprocessing again. RGB, depth,
masks, poses, and intrinsics are resized/adjusted by the runtime loader.

### Geometry-constrained bundle portability

The `megapose_gso_geometry_n5_k1_masked` profile additionally requires the
materialized train/validation geometry SQLite files and their N=5 plans. The
reference bank is packed: `renders/pi3_index/references.sqlite` points to the
70 relative `renders/shard-*.tar` files, so copying the complete `renders`
directory preserves random access without unpacking it.

Plan catalogues built before 2026-08-21 contain the workstation's absolute
`geometry_index_path` in SQLite metadata. This is now treated as legacy
provenance rather than a runtime location: the dataset config explicitly pairs
each plan with its Hydra-selected geometry index, and the reader can also
relocate a copied legacy plan to a same-named sibling. Existing transferred
indexes do **not** need regeneration. Newly generated plan and surface-bitset
metadata is relative to its owning SQLite file.

Before `sbatch`, geometry submissions automatically run:

```bash
python scripts/validate_megapose_gso_geometry_runtime.py \
  --data-root "$PI3_DATA_ROOT" \
  --assets-root "$PI3_GSO_ASSETS_ROOT" \
  --references-root "$PI3_GSO_REFERENCES_ROOT"
```

It checks all scene/reference TAR sizes, SQLite completion and plan/geometry
fingerprints, the model catalogue, and every resolved model symlink. Missing
`.surface_bits` files are only a warning: they are required to regenerate or
reanalyze plans, not to train from already-materialized SQLite plans.

## 336x252 profiles

Use one of these complete train/data pairs:

| Purpose | Train config | Data config |
| --- | --- | --- |
| Cheapest generalization diagnostic | `train_megapose_gso_finetune_a40_40gb_336x252_n5_k1` | `megapose_gso_anchor_scene_pairs` |
| Dynamic production/context sweep | `train_megapose_gso_finetune_a40_40gb_336x252_dynamic` | `megapose_gso_anchor_scene_pairs_multival` |
| Query full-depth transfer ablation | `train_megapose_gso_transfer_a40_40gb_336x252` | `megapose_gso_transfer_diagnostics_query_full_depth` |

The fixed profile always trains on five anchor views and one query. It has one
matching validation loader, `gso_val`, also with N=5/K=1.

The dynamic profile trains with 2-16 references, 1-10 queries, and 3-26 total
views. Its four held-out-object/held-out-scene validation loaders are:

| Loader | References | Queries | Total views | Semantics |
| --- | ---: | ---: | ---: | --- |
| `gso_val` | 5 | 1 | 6 | baseline and best-checkpoint selector |
| `gso_val_ref16` | 16 | 1 | 17 | more anchor context |
| `gso_val_k5` | 5 | 5 | 10 | five query frames from one query track |
| `gso_val_k10` | 5 | 10 | 15 | ten query frames from one query track |

All four use `anchor_pair` and `allow_repeat: false`. The K=5/K=10 queries are
therefore genuine within-scene context, not unrelated independent scenes.

The older fixed-560x420 pair remains available as
`train_megapose_gso_finetune_a40_46gb` plus
`megapose_gso_anchor_scene_pairs`; its validation is now also strict
anchor-pair N=5/K=1.

## Measured capacity and selected budgets

Capacity was measured on 2026-08-06 on the local RTX PRO 6000 Blackwell using
the real ten-shard MegaPose tensors, BF16, visibility-mask conditioning, and a
40 GiB PyTorch allocator cap. Independent-scene sampling was used only because
the partial index has zero held-out objects capable of anchor-pair N=5/K=1;
the tensor shapes and model path are otherwise identical.

| Probe | Shape | Allocated | Reserved | Result |
| --- | --- | ---: | ---: | --- |
| fixed N=5/K=1 capacity edge | `13 x 6 = 78` images | 37,801 MiB | 38,036 MiB | passed 3 steps, too close for production default |
| dynamic maximum edge | `3 x 26 = 78` images | 37,802 MiB | 38,022 MiB | passed 3 steps |
| dynamic packed-min edge | `26 x 3 = 78` images | 37,801 MiB | 37,976 MiB | passed 3 steps |
| selected training budget | `12 x 6 = 72` images | 35,588 MiB | 35,824 MiB | passed; about 4.2 GiB allocator headroom |

Both 336x252 profiles therefore use `train.max_img_per_gpu: 72`. The fixed
profile packs 12 N=5/K=1 samples per GPU. The dynamic sampler packs
`floor(72 / total_views)` samples, including 24 three-view samples and two
26-view samples per GPU.

Validation uses a budget of 256 for every protocol, realizing 252, 255, 250,
and 255 images/rank for the table above. An eval-only smoke with the configured
camera/correspondence metrics and visuals peaked at 10,107 MiB allocated and
12,124 MiB reserved. These are workstation-constrained results, not substitutes
for the mandatory A40 cluster smoke.

## Host RAM and workers

The historical launcher default requested 100 GB per training GPU: 400 GB for
a four-GPU job. Smoke requested 160 GB. Slurm `--mem` is total memory for the
job allocation, not a per-GPU allowance; requesting `110G` on a four-A40 node
gives all four DDP ranks a shared 110 GB job limit.

Each new profile uses two validation workers per rank. Four validation loaders
would create 32 persistent workers on four ranks and retain prefetched batches.
The loader now supports profile-controlled worker persistence, and these
profiles set `test.persistent_workers: false` while retaining two workers and a
prefetch factor of two.

On the local one-rank, four-loader validation smoke:

| Worker policy | Host-RAM growth over pre-run cgroup baseline |
| --- | ---: |
| persistent workers | 25,100 MiB |
| non-persistent workers | 11,929 MiB |

The conservative four-rank projection is about 48 GiB, before cluster-specific
runtime variation. The launcher therefore requests **110 GB total** by default
for these 336x252 GSO profiles and 80 GB for their two-GPU smoke. This leaves
more than 2x the projected measured use. Override with `PI3_TRAIN_MEM` or
`PI3_SMOKE_MEM` if CITEc accounting or an on-node measurement requires it. The
older 560x420 and LMGeo profiles retain their historical memory defaults.

## Compose checks

These commands require no GPU and do not instantiate the external data:

```bash
python scripts/train_pi3.py --cfg job \
  train=train_megapose_gso_finetune_a40_40gb_336x252_n5_k1 \
  data=megapose_gso_anchor_scene_pairs

python scripts/train_pi3.py --cfg job \
  train=train_megapose_gso_finetune_a40_40gb_336x252_dynamic \
  data=megapose_gso_anchor_scene_pairs_multival
```

The ten-shard local index cannot instantiate production anchor validation: its
maximum held-out track has four eligible views. For a local shape/loader check,
explicitly use `data=megapose_gso_cross_scene`. Do not transfer that data config
to the production command.

## CITEc submission

Set common locations once:

```bash
export PI3_REPO_DIR=/homes/dtrofimov/repositories/Pi3
export PI3_CONDA_ROOT=/homes/dtrofimov/miniconda3
export PI3_CONDA_ENV=pi3-lmgeo
export PI3_DATA_FAMILY=megapose_gso
export PI3_DATA_ROOT=/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/MegaPose-GSO-fixed
export PI3_CKPT=/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors
export PI3_RUNS_ROOT=/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3
export PI3_TENSORBOARD_ROOT=/homes/dtrofimov/logs/spott3r
export PI3_SLURM_JOB_PREFIX=pi3-gso-336
export PI3_SMOKE_GPUS=2
export PI3_TRAIN_GPUS=4
export PI3_TRAIN_CPUS=32
export PI3_WALLTIME=48:00:00
export PI3_CKPT_INTERVAL=1
export PI3_MAX_CHECKPOINTS=5
```

For the fixed N=5/K=1 experiment:

```bash
export PI3_TRAIN_CONFIG=train_megapose_gso_finetune_a40_40gb_336x252_n5_k1
export PI3_DATA_CONFIG=megapose_gso_anchor_scene_pairs

export PI3_RUN_NAME=megapose_gso_336_n5_k1_smoke_$(date +%Y%m%d_%H%M%S)
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke --fresh

export PI3_RUN_NAME=megapose_gso_336_n5_k1_$(date +%Y%m%d_%H%M%S)
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train --fresh
```

For dynamic training and all four validation protocols:

```bash
export PI3_TRAIN_CONFIG=train_megapose_gso_finetune_a40_40gb_336x252_dynamic
export PI3_DATA_CONFIG=megapose_gso_anchor_scene_pairs_multival

export PI3_RUN_NAME=megapose_gso_336_dynamic_smoke_$(date +%Y%m%d_%H%M%S)
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke --fresh

export PI3_RUN_NAME=megapose_gso_336_dynamic_$(date +%Y%m%d_%H%M%S)
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train --fresh
```

### Query full-depth transfer ablation

This isolated profile inherits the repeated-safe 50% query-mask policy,
train-only role-consistent photometric augmentation, eight transfer diagnostic
loaders, 80 epochs, 1,500-step warm-up, and validation every three epochs. Its
only data-target change is:

- all references/keyframes retain `depth: object_only`;
- GSO training queries use `depth: full`;
- validation retains object-only depth for direct comparison with the baseline.

Because the current point loss averages all valid pixels, this is deliberately
a strong ablation rather than a low-weight context auxiliary. In real samples,
full query depth covered nearly the complete 336x252 image and therefore
outnumbered target-object reference pixels substantially.

Local full-model checks under a 40 GiB allocator cap completed four backward
steps at both sampler extremes. The N=16/K=10 edge (`2 x 26 = 52` images) used
28,428 MiB peak reserved; the packed N=2/K=1 edge (`24 x 3 = 72` images) used
35,816 MiB, leaving about 4.2 GiB allocator headroom. These Blackwell checks do
not replace the required A40 cluster smoke.

From the repository on CITEc, submit the two-GPU smoke first:

```bash
PI3_CUDA_MEMORY_LIMIT_GIB=40 \
PI3_SMOKE_GPUS=2 \
PI3_TRAIN_CONFIG=train_megapose_gso_transfer_a40_40gb_336x252 \
PI3_DATA_CONFIG=megapose_gso_transfer_diagnostics_query_full_depth \
PI3_RUN_NAME=megapose_gso_query_full_depth_a40_smoke_$(date +%Y%m%d_%H%M%S) \
PI3_CKPT=/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors \
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke --fresh
```

After that succeeds, submit production on four A40s:

```bash
PI3_CUDA_MEMORY_LIMIT_GIB=40 \
PI3_TRAIN_GPUS=4 \
PI3_TRAIN_CONFIG=train_megapose_gso_transfer_a40_40gb_336x252 \
PI3_DATA_CONFIG=megapose_gso_transfer_diagnostics_query_full_depth \
PI3_RUN_NAME=megapose_gso_query_full_depth_a40_$(date +%Y%m%d_%H%M%S) \
PI3_CKPT=/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors \
PI3_WALLTIME=4-00:00:00 \
PI3_CKPT_INTERVAL=5 \
PI3_MAX_CHECKPOINTS=5 \
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train --fresh
```

The launcher now recognizes `megapose_gso_transfer_*` data profiles as
mesh/render-backed GSO data and validates/routes the shared assets before the
job starts. Override `PI3_GSO_ASSETS_ROOT` and `PI3_GSO_REFERENCES_ROOT` only if
their cluster locations differ from the documented `/vol/coro/...` defaults.

The smoke launcher reads the selected Hydra profile rather than hard-coding the
old 560x420/28-image shape. The fixed profile exercises its `12 x 6` edge. The
dynamic profile exercises `2 x 26` and `24 x 3`, plus one batch from every
configured validation loader at its production budget. A missing eligible
anchor protocol on the full server index is a data-capacity error; never fix it
by silently restoring independent scenes or repetition.

To continue an interrupted run, restore the same `PI3_RUN_NAME` and use:

```bash
scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh train --continue
```
