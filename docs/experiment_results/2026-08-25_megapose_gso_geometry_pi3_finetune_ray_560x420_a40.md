# 2026-08-25 MegaPose-GSO full-Pi3 ray fine-tune at 560x420

Status: complete. The run reached update 15,000 after a storage-related
interruption at update 5,972 and a restart from the update-5,000 checkpoint.
Standalone one-GPU validation is also complete for the best-loss checkpoint
at update 12,500 (`checkpoint_24`) and the final checkpoint at update 15,000
(`checkpoint_29`). This note compares them with the completed
[336x252 run](2026-08-18_megapose_gso_geometry_pi3_finetune_ray_run.md).

## Setup and artifacts

| Field | Value |
| --- | --- |
| Run | `megapose_gso_geometry_pi3_finetune_ray_560x420_a40_20260821_133116` |
| Train config | `train_megapose_gso_geometry_pi3_finetune_ray_a40_40gb_560x420` |
| Data config | `megapose_gso_geometry_n5_k1_masked` |
| Model | released full-size Pi3 checkpoint; frozen DINOv2-L encoder |
| Conditioning | zero-initialized ray projection, trained at `1e-5` peak LR |
| Historical Pi3 peak LR | `5e-6` |
| Views | N=5 references, K=1 query |
| Resolution | 560x420: 40x30 = 1,200 patches/view |
| Distributed batch | 4 samples / 24 images per rank, 2 A40 ranks; 8 samples/update globally |
| Schedule | 30 x 500 = 15,000 updates; 500-update OneCycle warm-up |
| TensorBoard | `/vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/megapose_gso_geometry_pi3_finetune_ray_560x420_a40_20260821_133116` |

The directory currently contains two event files and no checkpoint or Hydra
snapshot. The first event covers updates 1-5,972; the resumed event starts at
5,001 and ends at 15,000. All numbers below merge them by keeping the later
event for overlapping steps. This matters at update 5,500: the rerun from the
update-5,000 checkpoint followed a slightly different stochastic trajectory.

The logical, deduplicated event spans total about 25 hours of active training
and validation. The discarded first attempt at updates 5,001-5,972 cost about
1 hour 24 minutes. The training mixture was again almost exact 50/50: 7,540
render-reference updates and 7,460 scene-reference updates.

TensorBoard contains 561 scalar tags and 14 image tags. It can be viewed with:

```bash
tensorboard \
  --logdir /vol/coro/dtrofimov/data/projects/gfm-6dof/runs/Pi3/megapose_gso_geometry_pi3_finetune_ray_560x420_a40_20260821_133116 \
  --port 6006
```

## Standalone full validation (authoritative)

The training-time validation was distributed incorrectly for metric reporting,
as explained below. To remove that ambiguity, both useful checkpoints were
evaluated again on one GPU. Each evaluation contains exactly 189 unique
scene-reference queries and 189 unique render-reference queries, with no
underconstrained reference alignment. The protocol is identical to the earlier
standalone 336x252 audit except for resolution:

- N=5 references and K=1 query at 560x420;
- the fixed held-out validation plans, not the expanded 1,988-query diagnostic;
- reference-depth-derived metric scale and reference-camera-derived rotation
  and translation (`scale_estimation=reference_depth`);
- query-occupancy row export enabled and visualizers disabled;
- 132-image validation budget, giving nine batches per regime on one GPU.

Artifacts:

| Checkpoint | Update | Output / TensorBoard root |
| --- | ---: | --- |
| Best scene loss | 12,500 | `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_geometry_560x420_full_eval_checkpoint24_20260825` |
| Final | 15,000 | `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_geometry_560x420_full_eval_checkpoint29_20260825` |

Both used 17.44 GiB peak allocated-plus-reserved memory and took approximately
2 minutes 20 seconds after checkpoint loading. The full Accelerate checkpoint
directories are under:

```text
/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/Pi3/megapose_gso_geometry_pi3_finetune_ray_560x420_a40_20260821_133116/ckpts
```

### Full-set comparison with 336x252

The final checkpoint is the primary result below. Unlike the preliminary
training-time table later in this note, every column contains the same 189
query identities.

#### Scene references to scene query

| Metric | 336x252 final | 560x420 checkpoint 29 | Change |
| --- | ---: | ---: | ---: |
| Validation loss | 0.16759 | **0.14862** | -11.3% |
| Normalized pose median / mean / p90 | 0.7359 / 1.0853 / **2.2709d** | **0.6224 / 1.0501** / 2.4418d | -15.4% / -3.2% / +7.5% |
| Query rotation median / mean | 15.45 / 42.07 deg | **9.65 / 38.44 deg** | -37.6% / -8.6% |
| Query translation median / mean | 0.1531 / 0.2329 m | **0.1249 / 0.2253 m** | -18.4% / -3.3% |
| Query camera-center median / mean | 0.3286 / 0.5919 m | **0.2068 / 0.5595 m** | -37.1% / -5.5% |
| Reference rotation median | 5.23 deg | **4.10 deg** | -21.6% |
| Reference camera-center median | 0.1110 m | **0.0898 m** | -19.1% |
| Recall below 0.5d / 1d / 2d | 33.86 / 64.02 / **86.77%** | **40.21 / 65.08** / 86.24% | +6.35 / +1.06 / -0.53 pp |
| Strict ADD below 0.1d | **2.12%** | 0.53% | -1.59 pp |
| ADD-S-for-all below 0.1d | **11.64%** | 8.99% | -2.65 pp |

The typical estimate is substantially better, including rotation, translation,
camera center, and reference alignment. The p90 and strict ADD do not improve:
high resolution moves the center of the distribution but does not eliminate
the difficult scene-reference tail.

#### Render references to scene query

| Metric | 336x252 final | 560x420 checkpoint 29 | Change |
| --- | ---: | ---: | ---: |
| Validation loss | 0.20634 | **0.16548** | -19.8% |
| Normalized pose median / mean / p90 | 0.4614 / 0.6805 / 1.5747d | **0.3898 / 0.6217 / 1.3733d** | -15.5% / -8.6% / -12.8% |
| Query rotation median / mean | 11.01 / 31.89 deg | **9.20 / 25.29 deg** | -16.5% / -20.7% |
| Query translation median / mean | 0.0973 / 0.1464 m | **0.0865 / 0.1347 m** | -11.1% / -8.0% |
| Query camera-center median / mean | 0.2164 / 0.4422 m | **0.1635 / 0.3715 m** | -24.4% / -16.0% |
| Reference rotation median | 3.57 deg | **2.99 deg** | -16.2% |
| Reference camera-center median | 0.0260 m | **0.0223 m** | -14.4% |
| Recall below 0.5d / 1d / 2d | 53.44 / 79.37 / 96.30% | **58.73 / 83.07** / 96.30% | +5.29 / +3.70 / 0.00 pp |
| Strict ADD below 0.1d | 3.70% | **7.94%** | +4.23 pp |
| ADD-S-for-all below 0.1d | 22.22% | **30.69%** | +8.47 pp |

Render-to-scene improves on almost every axis, including the rotation tail,
translation, camera center, normalized-pose p90, and ADD(-S). This is stronger
and more uniform evidence than the scene-reference result.

### Best-loss checkpoint versus final checkpoint

Checkpoint 24 has the best scene validation loss (0.14697 versus 0.14862),
slightly better scene mean/p90 pose (1.0475/2.3292d versus 1.0501/2.4418d),
and better scene ADD-S-for-all (10.05% versus 8.99%). Checkpoint 29 improves
the scene medians for rotation (9.77 to 9.65 degrees) and translation (0.136
to 0.125 m) plus the 0.5d/1d recalls.

Checkpoint 29 is broadly better for render references: normalized pose improves
from 0.409/0.660/1.678d to 0.390/0.622/1.373d, rotation from 9.68/25.69 to
9.20/25.29 degrees, and ADD-S-for-all from 28.04% to 30.69%. Therefore the final
checkpoint is the better general default; checkpoint 24 is useful when the
scene-reference validation loss and hard tail are the priority.

### Warning about the historical N=5 versus N=16 GSO result

This experiment evaluates only N=5/K=1. An older GSO pipeline did compare N=5
with N=16 render references over the same 19,460 held-out queries, but it did
not match effective normalized focal length or object-preserving crops between
the render bank and each scene query. Those queries could be out of distribution
in image scale and perspective, and reference-only Sim(3) alignment cannot
remove an intrinsics mismatch. Its result therefore means only that adding more
focal-inconsistent references did not solve the OOD query; it is not evidence
about the value of additional calibrated keyframes.

A valid reference-count ablation for this run requires N-specific
geometry-plan catalogues, the same query identities, focal-consistent crops,
the 10-degree positive-reference and surface-coverage constraints, and
`reference_depth` scale estimation. It should cover render-to-scene and
scene-to-scene separately. Until that is run, the effect of N in the current
calibrated MegaPose-GSO setting is unknown.

### Query occupancy / visible-patch analysis

`query_visible_patches` is visible object area divided by the 14x14 patch area.
It is a patch-equivalent area, not a count of tokens that merely intersect the
mask. The 560x420 preprocessing preserves essentially the same crop occupancy
as 336x252, while raising this budget by a median factor of 2.78. Therefore the
binning below uses visible image fraction to compare the same object-size
regimes.

#### Scene-reference checkpoint 29

| Query occupancy | N | Median visible patches | Pose med. | Rot. med. | Trans. med. | Center med. | Recall <0.5d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <0.5% | 21 | 3.09 | 1.893d | 68.08 deg | 0.388 m | 1.013 m | 9.5% |
| 0.5-1% | 26 | 9.26 | 1.740d | 37.25 deg | 0.353 m | 0.881 m | 3.8% |
| 1-2% | 35 | 18.17 | 0.591d | 11.67 deg | 0.122 m | 0.278 m | 37.1% |
| 2-4% | 47 | 32.59 | 0.524d | 7.08 deg | 0.105 m | 0.146 m | 44.7% |
| 4-8% | 37 | 66.77 | 0.466d | 5.18 deg | 0.105 m | 0.112 m | 59.5% |
| >=8% | 23 | 162.10 | 0.300d | 3.51 deg | 0.071 m | 0.068 m | 73.9% |

#### Render-reference checkpoint 29

| Query occupancy | N | Median visible patches | Pose med. | Rot. med. | Trans. med. | Center med. | Recall <0.5d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <0.5% | 32 | 3.59 | 1.257d | 17.97 deg | 0.266 m | 0.494 m | 12.5% |
| 0.5-1% | 37 | 8.08 | 0.500d | 9.66 deg | 0.095 m | 0.203 m | 51.4% |
| 1-2% | 56 | 17.45 | 0.388d | 9.30 deg | 0.085 m | 0.155 m | 67.9% |
| 2-4% | 43 | 32.96 | 0.242d | 3.73 deg | 0.055 m | 0.090 m | 76.7% |
| 4-8% | 19 | 66.10 | 0.155d | 4.73 deg | 0.037 m | 0.075 m | 84.2% |
| >=8% | 2 | 106.74 | 0.320d | 43.90 deg | 0.063 m | 0.450 m | 50.0% |

The final render `>=8%` bin has only two samples and is not statistically
interpretable. All other bins expose the main failure axis clearly. For scene
references, the transition below 1% occupancy (roughly fewer than ten visible
patch-equivalents even at 560x420) is catastrophic. Render references tolerate
the 0.5-1% regime much better, but also fail strongly below 0.5%.

Spearman correlation between occupancy and error remains strong at 560x420:
scene pose/rotation/translation/center correlations are -0.524/-0.540/-0.512/
-0.540; render correlations are -0.526/-0.376/-0.513/-0.473. The extra pixels
lower absolute errors but do not remove object size as a failure predictor.

An exact per-query 336x252-to-560x420 pairing confirms that the rotation gain
is distributed across object sizes. Scene median rotation improves from
100.34 to 68.08 degrees below 0.5% occupancy, 14.07 to 11.67 degrees at 1-2%,
11.33 to 7.08 degrees at 2-4%, and 6.36 to 3.51 degrees at >=8%. Rotation gets
better for 63.0% of individual scene queries and 66.7% of render queries.

This supports the extra-patch-evidence explanation, but it is not a causal
resolution ablation: global sample batch and training exposure also changed.
The clean isolation is a 2x2 evaluation of the old checkpoint at 560x420 and
the new checkpoint at 336x252, in addition to their native resolutions.

### Why scene query rotation fell more than render rotation

The full-set comparison confirms that both fall: scene median rotation drops
15.45 to 9.65 degrees, while render drops 11.01 to 9.20 degrees and its mean
drops 31.89 to 25.29 degrees. Render already had a roughly 10-degree median,
so it had less typical-case headroom; its larger benefit is in the tail. Scene
references are partial, cluttered observations, and the denser representation
also improves their alignment (reference rotation 5.23 to 4.10 degrees), which
helps the subsequent query-to-reference registration. That reference gain is
only about one degree, however, so most of the scene query improvement is a
better relative query registration rather than merely a better Sim(3) frame.

Lower median rotation does not imply solved 6-DoF pose. The scene queries below
1% occupancy still have 37-68 degree median rotation, 0.35-0.39 m translation,
and 0.88-1.01 m camera-center error. These failures keep p90 pose and strict
ADD poor even though the typical query is much better.

## Training-time TensorBoard caveat: validation was sharded

The 336x252 run evaluated all 189 deterministic held-out samples on one GPU.
In this two-GPU run, each validation metric was accumulated locally and only
the main process wrote TensorBoard; its `query_count` is 95 per regime. Thus
the 560x420 event reports rank 0's even-index validation shard, not a reduction
over all 189 unique samples. PyTorch pads the odd-sized dataset to 190 entries
for two ranks, hence 95 samples/rank.

This makes loss (averaged by the trainer) and broad trends useful, but makes
small recall changes noisy: one sample is 1.05 percentage points. It is not an
exact full-set resolution ablation. A definitive comparison requires a
standalone one-GPU eval of the 560x420 checkpoint over all 189 samples.

The following historical analysis motivated the standalone validation above.
As a partial control, the earlier standalone 336x252 raw rows can be restricted
to the same even-index 95-object shard. Rotation is scale-invariant, so this is
an especially clean check:

| Rotation metric | 336x252 same 95-object shard | 560x420 rank-0 shard | Change |
| --- | ---: | ---: | ---: |
| Scene query median / mean | 18.20 / 43.86 deg | **8.45 / 39.98 deg** | -53.6% / -8.9% |
| Render query median / mean | **10.61** / 38.82 deg | 11.02 / **27.56 deg** | +3.8% / -29.0% |

The scene median-rotation gain and render rotation-tail gain therefore are not
explained by comparing the new half-set with the old full-set. This assumes the
unchanged deterministic validation index order used by both runs.

## Optimization curves

“First” is the first completed epoch at update 500. “Best” is selected per
metric, so best entries do not identify one jointly best checkpoint.

| Metric | First | Best (update) | Final |
| --- | ---: | ---: | ---: |
| Train loss | 0.50932 | **0.17236** (13,500) | 0.18337 |
| Train local points | 0.04902 | **0.01128** (14,000) | 0.01136 |
| Train translation loss | 0.03593 | **0.01239** (14,500) | 0.01330 |
| Train rotation loss | 1.00998 | **0.36918** (13,500) | 0.38970 |
| Scene validation loss | 0.30441 | **0.14792** (12,500) | 0.14952 |
| Render validation loss | 0.45382 | **0.16455** (11,000) | 0.16885 |

Final train, scene-validation, and render-validation losses fell by 64.0%,
50.9%, and 62.8% from the first evaluation. Relative to the 336x252 final,
the new final scene and render losses are lower by 10.8% and 18.2%. The local
point loss is 3.8% higher, while train translation and rotation losses are
6.0% and 9.1% lower. Train-loss comparisons are less controlled because the
global sample batch changed from 26 to 8.

The curve is effectively at its horizon by 12,500-14,500. Scene validation
loss reaches its minimum at 12,500 and finishes only 1.1% higher. Render loss
reaches its minimum earlier at 11,000 and finishes 2.6% higher. The last few
epochs improve some medians but worsen the hard tail, so there is no evidence
that simply extending the exhausted OneCycle schedule would remove the
remaining failures.

## Preliminary training-time pose comparison (95-query shard)

These are direct final TensorBoard values. The 336x252 column has 189 queries;
the 560x420 column has the rank-0 95-query shard described above.

### Scene references to scene query

| Metric | 336x252 final | 560x420 final | Observed change |
| --- | ---: | ---: | ---: |
| Validation loss | 0.1676 | **0.1495** | -10.8% |
| Normalized pose, median | 0.783d | **0.741d** | -5.5% |
| Normalized pose, mean / p90 | **1.134 / 2.445d** | 1.155 / 2.580d | +1.9% / +5.5% |
| Query rotation, median / mean | 15.45 / 42.07 deg | **8.45 / 39.98 deg** | -45.3% / -5.0% |
| Query translation, median / mean | 0.170 / **0.245 m** | **0.153** / 0.249 m | -10.1% / +1.7% |
| Query camera center, median / mean | 0.342 / 0.599 m | **0.250 / 0.591 m** | -26.8% / -1.4% |
| Reference rotation, median | 5.23 deg | **4.07 deg** | -22.1% |
| Reference center, median | 0.105 m | **0.086 m** | -18.9% |
| Recall below 0.5d / 1d / 2d | 33.33 / **62.43** / 85.19% | **38.95** / 56.84 / **86.32%** | +5.61 / -5.59 / +1.13 pp |
| Strict ADD below 0.1d | **2.12%** | 0.00% | -2.12 pp |
| ADD-S-for-all below 0.1d | **13.76%** | 11.58% | -2.18 pp |
| Correspondence L2 median | **0.0587** | 0.0635 | +8.3% |

The typical scene-query estimate improves strongly, especially its rotation
and camera-center median. The mean, p90, 1d recall, and strict ADD do not
improve. Higher resolution therefore did not remove the difficult
scene-reference tail. The best scene median pose was 0.712d at update 11,000;
the best median rotation was 8.28 degrees at 14,500; the final translation
median of 0.153 m was the run's best.

### Render references to scene query

| Metric | 336x252 final | 560x420 final | Observed change |
| --- | ---: | ---: | ---: |
| Validation loss | 0.2063 | **0.1688** | -18.2% |
| Normalized pose, median / mean / p90 | 0.439 / 0.810 / 1.785d | **0.403 / 0.621 / 1.578d** | -8.2% / -23.3% / -11.6% |
| Query rotation, median / mean | **11.01** / 31.89 deg | 11.02 / **27.56 deg** | +0.1% / -13.6% |
| Query translation, median / mean | 0.099 / 0.175 m | **0.090 / 0.135 m** | -8.9% / -23.0% |
| Query camera center, median / mean | 0.213 / 0.438 m | **0.175 / 0.371 m** | -17.7% / -15.3% |
| Reference rotation, median | 3.57 deg | **3.17 deg** | -11.1% |
| Reference center, median | 0.026 m | **0.023 m** | -12.9% |
| Recall below 0.5d / 1d / 2d | 52.91 / 76.72 / 92.59% | **57.89 / 80.00 / 96.84%** | +4.98 / +3.28 / +4.25 pp |
| Strict ADD below 0.1d | 4.76% | **11.58%** | +6.82 pp |
| ADD-S-for-all below 0.1d | 21.16% | **28.42%** | +7.26 pp |
| Correspondence L2 median | 0.0961 | **0.0860** | -10.5% |

Render-to-scene improves much more consistently. Median rotation is unchanged,
but its mean improves substantially: resolution reduces the rotation failure
tail rather than the already-good typical render-reference case. Translation,
camera center, normalized-pose tail, all broad recalls, strict ADD, and the
correspondence diagnostic improve together. Best median pose was 0.391d at
14,500, best translation was 0.089 m at 12,000, and strict ADD reached and
retained 11.58% from update 12,500.

## What the extra spatial resolution appears to buy

The resolution changes from 432 to 1,200 patches/view, a 2.78x increase. At
the same post-crop visible object fraction, the approximate object-patch budget
changes as follows:

| Query occupancy | 336x252 object patches | 560x420 object patches |
| --- | ---: | ---: |
| 0.5% | 2.2 | 6 |
| 1% | 4.3 | 12 |
| 2% | 8.6 | 24 |
| 4% | 17.3 | 48 |
| 8% | 34.6 | 96 |

This was the initial hypothesis from the training event. The training profile
did not enable `query_occupancy_analysis`, so those TensorBoard files alone do
not contain occupancy bins or raw rows. The completed standalone evaluation
above now tests the hypothesis on all 189 queries and confirms that occupancy
remains a dominant failure axis at 560x420.

The run's new camera-error decomposition is consistent with better typical
viewpoint placement. At the final step:

| Query-center diagnostic | Scene refs | Render refs |
| --- | ---: | ---: |
| Center median / mean | 0.250 / 0.591 m | 0.175 / 0.371 m |
| Tangential median | 0.158 m | 0.129 m |
| Radial median | 0.121 m | 0.052 m |
| Camera-direction median / mean | 5.69 / 22.38 deg | 5.38 / 15.38 deg |
| Radius-error median / mean | 0.107 / 0.158 m | 0.050 / 0.101 m |

The median-to-mean gaps remain large. Scene camera direction is good for the
typical sample, yet its mean is still 22 degrees and the radius error remains
about 11 cm at the median. The residual problem is still a failure tail in
query placement around and away from the object, especially with scene
references—not a uniform inability to regress a pose.

## Ray conditioning did not explain the pose gain

All intrinsics fits were valid and both Sim(3) metrics reported zero
underconstrained samples on the logged shard. Final query-ray angular medians
were 0.402 degrees (scene) and 0.556 degrees (render), versus 0.316 and 0.529
degrees at 336x252. The scene value is 27% worse and the render value 5% worse.
Fitted query focal errors are about 3.3-4.8%; principal-point errors are about
5-9 pixels at 560x420.

Thus the pose improvements do not come from a more accurate ray adapter. The
adapter is already sufficiently accurate in both runs; the likely benefit is
the denser frozen-DINO/decoder evidence for object geometry and cross-view
registration. Scene-reference correspondence actually worsens slightly while
render-reference correspondence improves, matching the split behavior above.

## Data exposure and interpretation

The 336x252 run used 26 samples/update on one GPU: 390,000 sampled sequences
and 2.34 million images. This two-GPU run used 8 samples/update globally:
120,000 sequences and 720,000 images. It therefore saw only 31% as many sampled
sequences, although every image had 2.78x as many patches. The comparison is
not a pure resolution ablation: resolution, world size, global sample batch,
and data exposure changed together, while LR and update count stayed fixed.

Even with much less sample exposure, 560x420 gives a real improvement for
clean render references and a strong typical-case scene-rotation improvement.
It does not solve the scene-reference tail or strict scene ADD. The standalone
full evaluation has now removed the distributed shard ambiguity and confirms
the low-occupancy failure mode. The next controlled diagnostic is the 2x2
cross-resolution checkpoint evaluation described above; extending the already
exhausted OneCycle schedule is lower priority.
