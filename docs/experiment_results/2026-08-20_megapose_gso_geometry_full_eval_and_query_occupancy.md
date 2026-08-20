# 2026-08-20 MegaPose-GSO standalone held-out eval and query-occupancy audit

Status: complete. Two new isolated eval-only Hydra profiles and one sampling
policy feature, all additive to the existing GSO/LM-O pipelines.

## Motivation

The [2026-08-18 full-Pi3 ray fine-tune](2026-08-18_megapose_gso_geometry_pi3_finetune_ray_run.md)
reported held-out GSO metrics from training-time validation: one deterministic
N=5/K=1 sample per of 189 held-out objects, `scale_estimation=camera_centers`,
no per-query occupancy diagnostics. The [2026-08-18 LM-O transfer](2026-08-18_lmgeo_geometry_matched_gso_transfer.md)
later added `scale_estimation=reference_depth` and exact post-crop
query-occupancy analysis, and found apparent query size to be the strongest
measured LM-O failure axis. This entry:

1. re-runs the completed checkpoint on its own held-out GSO validation split,
   standalone, with the same enriched diagnostics used for LM-O, so the two
   datasets are compared on equal footing (same scale estimator, same
   occupancy bins);
2. adds an `enumerate_query_groups` sampling-policy feature so held-out GSO
   validation is not limited to one query per object, and reruns the
   render-to-scene regime with it enabled.

Checkpoint under test throughout: `checkpoint_29` from
`megapose_gso_geometry_pi3_finetune_ray_n5k1_336_warmup1000_photoaug_20260817_102019`
(15,000 updates, full-size Pi3 + ray adapter).

## Part 1: standalone eval, matched to the LM-O protocol

### Implementation

Two new eval-only Hydra profiles, both additive (nothing existing was
modified):

- [`configs/train/train_megapose_gso_geometry_pi3_eval_rtxpro6000_70gb_336x252.yaml`](../../configs/train/train_megapose_gso_geometry_pi3_eval_rtxpro6000_70gb_336x252.yaml) —
  eval-only restoration of `checkpoint_29`, mirrors the existing LM-O eval
  profile: `ObjectPoseMetric`/`CameraAlignmentMetric` with
  `scale_estimation=reference_depth`, `query_occupancy_analysis=true`, same
  bin edges `[0, 0.005, 0.01, 0.02, 0.04, 0.08, 1.000001]`. Visuals disabled
  (metrics only). `correspondence`/`chamfer` metrics disabled to match the
  LM-O eval's minimal, fast metric set.
- [`configs/general/megapose_gso_geometry_eval.yaml`](../../configs/general/megapose_gso_geometry_eval.yaml) —
  disables checkpoint writes, the GSO-named twin of `general=lmgeo_geometry_eval`.

Both were smoke-tested (`test.iters_per_test=1`) before the full run.

```bash
RUN=megapose_gso_geometry_val_full_query_occupancy_20260819
CUDA_VISIBLE_DEVICES=0 PI3_CUDA_MEMORY_LIMIT_GIB=70 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
python -u scripts/train_pi3.py \
  train=train_megapose_gso_geometry_pi3_eval_rtxpro6000_70gb_336x252 \
  data=megapose_gso_geometry_n5_k1_masked \
  general=megapose_gso_geometry_eval \
  name="$RUN" \
  log.output_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN" \
  log.tensorboard_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"/tensorboard \
  log.ckpt_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"/ckpts \
  hydra.run.dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"
```

Ran in 1m47s, ~25 GiB peak VRAM, 0 underconstrained Sim(3) fits, no
NaNs/errors. Loss values are bit-identical to the training run's final
validation loss (rotation is scale-invariant so it matches exactly regardless
of scale estimator; translation/camera-center shift a few percent under
`reference_depth`, the same size effect as the LM-O reference-depth ablation).

### Main metrics (all `reference_depth` scale — apples-to-apples across rows)

189 queries per GSO regime; 1,364 for LM-O (from the 2026-08-18 doc).

| Metric | GSO scene→scene | GSO render→scene | LM-O render→scene |
|---|---:|---:|---:|
| Reference rotation, median | 5.23° | 3.57° | 2.68° |
| Reference center residual, median | 11.0 cm | 2.6 cm | 1.8 cm |
| Query rotation, median / mean | 15.45° / 42.07° | 11.01° / 31.89° | 13.14° / 33.34° |
| Query translation \(T_{C\leftarrow O}\), median / mean | 15.3 / 23.3 cm | 9.7 / 14.6 cm | 8.1 / 12.7 cm |
| Query camera-center, median / mean | 32.9 / 59.2 cm | 21.6 / 44.2 cm | 20.7 / 39.3 cm |
| Normalized pose, median / mean / p90 | 0.736 / 1.085 / 2.271d | 0.461 / 0.681 / 1.575d | 0.450 / 0.848 / 2.193d |
| Tangential / radial center ratio | 2.42x | 2.37x | 2.54x |

**Recall under thresholds**

| Threshold | GSO scene→scene | GSO render→scene | LM-O render→scene |
|---|---:|---:|---:|
| Pose < 0.5d | 33.86% | 53.44% | 53.59% |
| Pose < 1d | 64.02% | 79.37% | 74.12% |
| Pose < 2d | 86.77% | 96.30% | 88.64% |
| Official ADD(-S) < 0.1d | 2.12% | 3.70% | 5.87% |

GSO has no symmetric object IDs, so ADD(-S) equals strict ADD there.

**Reading:** the checkpoint generalizes to clean-reference GSO render queries
about as well as it does to matched LM-O; GSO scene-to-scene is
substantially worse, and part of that gap is that GSO scene references are
themselves noisy (11.0 cm residual vs. LM-O's 1.8 cm) — not purely a
query-side generalization failure the way LM-O's clean-reference protocol is.
The tangential/radial center-error ratio is close to 2.4x in all three rows:
the dominant failure mode is consistently *where around the object* the query
camera is placed, not distance-to-object or rotation alone.

### Query-occupancy bins (N=189/regime)

| Occupancy | GSO scene N | GSO scene pose med | GSO render N | GSO render pose med |
|---|---:|---:|---:|---:|
| <0.5% | 22 | 1.970d | 32 | 1.355d |
| 0.5–1% | 25 | 1.530d | 35 | 0.617d |
| 1–2% | 35 | 0.685d | 58 | 0.433d |
| 2–4% | 47 | 0.644d | 43 | 0.280d |
| 4–8% | 37 | 0.535d | 19 | 0.209d |
| ≥8% | 23 | 0.259d | **2** | 0.506d |

Spearman correlation (log occupancy vs. error, all queries):

| Error | GSO scene | GSO render | LM-O (reference) |
|---|---:|---:|---:|
| Camera center | −0.599 | −0.511 | −0.715 |
| Rotation | −0.548 | −0.400 | −0.586 |
| Translation | −0.528 | −0.500 | −0.604 |

Direction and rough magnitude match LM-O. Caveat at this sample size: GSO val
draws exactly one query per held-out object (189 objects, 189 queries), so
unlike LM-O (1,364 queries over 8 repeated objects) there was no repetition to
run a per-object-controlled correlation, and the render ≥8% bin has only 2
samples. Part 2 fixes both.

## Part 2: enumerated render-to-scene validation (bigger N)

### Why N was capped at 189

`GeometryConstrainedScenePairPolicy`/`GeometryConstrainedRenderToScenePolicy`
inherit `natural_length()` from the base sampling-policy class:

```python
def natural_length(self, sources):
    return len(self.eligible_object_ids(sources))
```

i.e. dataset length = number of eligible objects, one deterministic
`query_selection="first"` sample per object — by design, for cheap
per-epoch validation during a 15,000-update run. The actual held-out pool is
much larger: the GSO val geometry index has a median of 87 crop-feasible
scene groups per held-out object (16,520 groups total across 189 objects).

### Implementation

Added an `enumerate_query_groups` flag (plus a `max_query_groups_per_object`
cap) to `_GeometryConstrainedReferencePolicy` in
[`datasets/object_sampling.py`](../../datasets/object_sampling.py) — the
shared base class for both scene-pair and render-to-scene geometry-constrained
policies, so the feature works for either regime without duplicated code.
This also completes a pre-existing stub: `GeometryConstrainedRenderToScenePolicy`
already accepted an `enumerate_query_groups` kwarg that unconditionally raised
`NotImplementedError`; that interception was removed so the kwarg now flows to
the (newly functional) base implementation. Naming matches the older
`RenderToScenePolicy.enumerate_query_groups` precedent used by the baseline
(non-geometry-constrained) render-scene-pair profile.

When enabled, `_enumerated_query_targets()` walks every eligible object's
scene groups in natural order, keeps the first geometrically-feasible query
record per group (capped at `max_query_groups_per_object`), and caches the
result once per dataset instance (`natural_length`/`build_plan` then index
into this list instead of always taking the first hit). `_choose_query_and_plan`
was refactored to return unresolved plan candidates so both the
streaming (training-time, early-exit) and enumerated (validation-time,
exhaustive) code paths resolve `plan_selection` in one shared place.

New, isolated data config —
[`configs/data/megapose_gso_geometry_n5_k1_masked_render_enumerated_eval.yaml`](../../configs/data/megapose_gso_geometry_n5_k1_masked_render_enumerated_eval.yaml) —
composes on top of `megapose_gso_geometry_n5_k1_masked` and overrides only the
render val entry's sampling policy (`enumerate_query_groups: true`,
`max_query_groups_per_object: 12`, chosen to land close to LM-O's 1,364-query
scale without enumerating all ~16,520 groups). Scene-to-scene validation is
untouched by this file.

Reused the existing eval-only train profile from Part 1 (no new train config
needed):

```bash
RUN=megapose_gso_geometry_val_render_enumerated_full_20260819
CUDA_VISIBLE_DEVICES=0 PI3_CUDA_MEMORY_LIMIT_GIB=70 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
python -u scripts/train_pi3.py \
  train=train_megapose_gso_geometry_pi3_eval_rtxpro6000_70gb_336x252 \
  data=megapose_gso_geometry_n5_k1_masked_render_enumerated_eval \
  general=megapose_gso_geometry_eval \
  name="$RUN" \
  log.output_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN" \
  log.tensorboard_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"/tensorboard \
  log.ckpt_dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"/ckpts \
  hydra.run.dir=/media/internal/nvme/dtrofimov/spott3r/outputs/"$RUN"
```

Ran in ~5m07s total (≈3.5 min one-time enumeration/index build + ≈1m34s GPU
inference over 16 batches), no errors/NaNs/underconstrained fits. Scene-to-scene
validation loss in this same run (0.1676) is bit-identical to every prior run,
confirming zero effect on the untouched regime.

### Result: 1,988 render→scene queries, all 189 objects represented

Per-object query count: min 2, median 12 (the cap), mean 10.5, max 12.

| Metric | Old render (N=189) | New render (N=1,988) |
|---|---:|---:|
| Reference rotation, median / mean | 3.57° / 11.78° | 3.51° / 10.56° |
| Reference center residual, median / mean | 2.6 / 6.5 cm | 2.6 / 5.9 cm |
| Query rotation, median / mean | 11.01° / 31.89° | 12.22° / 34.24° |
| Query translation, median / mean | 9.7 / 14.6 cm | 10.6 / 16.8 cm |
| Query camera-center, median / mean | 21.6 / 44.2 cm | 24.3 / 47.0 cm |
| Normalized pose, median / mean / p90 | 0.461 / 0.681 / 1.575d | 0.514 / 0.788 / 1.784d |
| Tangential / radial center ratio | 2.37x | 2.35x |
| Recall < 0.5d / 1d / 2d | 53.44 / 79.37 / 96.30% | 48.74 / 75.55 / 92.15% |
| Official ADD(-S) < 0.1d | 3.70% | 4.53% |

The reference-side numbers barely move (as expected — references still come
from the same clean render bank; only the query side is enumerated). Every
query-side metric gets mildly worse (+1° rotation, +1 cm translation, −5 pp
recall <0.5d): the deterministic single-query-per-object sample was a mild
positive-selection artifact, not a wrong number. The qualitative conclusion is
unchanged (tangential ≈ 2.35–2.4x radial, consistent across scene, render, and
LM-O).

### Occupancy bins, now well-populated in every bin

| Occupancy | N | Pose median | Pose p90 | Recall <0.5d | Rotation median | Translation median | Camera-center median |
|---|---:|---:|---:|---:|---:|---:|---:|
| <0.5% | 358 | 1.232d | 2.946d | 12.0% | 46.24° | 0.254 m | 0.564 m |
| 0.5–1% | 409 | 0.691d | 1.853d | 32.3% | 17.56° | 0.143 m | 0.356 m |
| 1–2% | 546 | 0.417d | 1.252d | 55.7% | 9.78° | 0.090 m | 0.211 m |
| 2–4% | 467 | 0.297d | 0.954d | 69.8% | 7.96° | 0.066 m | 0.151 m |
| 4–8% | 184 | 0.221d | 0.731d | 78.8% | 5.91° | 0.048 m | 0.086 m |
| ≥8% | 24 | 0.174d | 0.888d | 79.2% | 5.07° | 0.037 m | 0.060 m |

(The old ≥8% bin had 2 samples; this run gives 24 — an actual measurement
instead of a coin flip.)

### Correlation, with the control GSO previously couldn't run

| Error | All queries (n=1,988) | visib≥0.7 (n=1,208) | Within-object rank (n=1,988) |
|---|---:|---:|---:|
| Normalized pose | −0.545 | −0.427 | −0.575 |
| Camera center | −0.506 | −0.411 | −0.569 |
| Rotation | −0.427 | −0.231 | −0.485 |
| Translation | −0.527 | −0.417 | −0.563 |

With only 1 query/object, the earlier GSO run could not compute a
within-object-controlled correlation the way the 1,364-query, 8-object LM-O
run could. With up to 12 queries/object here, it can — and the
within-object correlation is as strong or stronger than the raw aggregate.
This confirms query occupancy is a real per-object effect on GSO too, not an
artifact of small objects also being harder objects/scenes.

## Validation

- 211/211 CPU tests pass (210 pre-existing + 1 new:
  `tests.test_geometry_constrained_sampling.GeometryConstrainedSamplingTest.test_enumerate_query_groups_covers_every_cross_scene_group_once`,
  which extends the existing synthetic fixture with a third cross-scene query
  group and checks enumerated length, the `max_query_groups_per_object` cap,
  and that every enumerated group is actually visited once).
- `hydra --cfg job` dry-run confirmed the enumerated data config only touches
  `val_datasets.gso_geometry_val_render_n5_k1.dataset.sampling_policy`; the
  scene policy config is unchanged.
- 1-batch smokes passed for both new eval profiles before the full runs.
- No errors, NaNs, or underconstrained Sim(3) fits in any of the three GPU
  runs in this entry.

## Artifacts

| Run | Path |
| --- | --- |
| Standalone matched eval (Part 1) | `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_geometry_val_full_query_occupancy_20260819` |
| Enumerated render-to-scene eval (Part 2) | `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_geometry_val_render_enumerated_full_20260819` |

Both contain `tensorboard/` and
`query_occupancy/<val_name>/step_00000030/` (`query_rows.csv`,
`summary.json`, `summary_visibility_ge_0_7.json`, `correlations.json`,
`object_bin_counts.csv`, `occupancy_metrics.png`).

## What this does and does not establish

It establishes that the LM-O query-occupancy failure mode (small apparent
query size predicts worse rotation/translation/camera-center, even
controlling for visibility and object identity) reproduces on GSO's own
held-out split, in both the scene-to-scene and render-to-scene regimes, and
that the enumerated render-to-scene sample is large enough to run the
same within-object control LM-O used. It does not change the underlying
checkpoint or training data, does not establish a causal fix, and the
`enumerate_query_groups`/`max_query_groups_per_object=12` cap is a scale
choice for tractable eval time, not an exhaustive census of the held-out
pool (16,520 groups exist; 1,988 were sampled). Scene-to-scene enumeration
was implemented but not run in this entry — the flag is available on
`GeometryConstrainedScenePairPolicy` too, should a larger scene-regime sample
be wanted later.
