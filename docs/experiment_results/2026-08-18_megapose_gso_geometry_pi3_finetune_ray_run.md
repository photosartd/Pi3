# 2026-08-18 MegaPose-GSO geometry full-Pi3 ray fine-tune

Status: complete. The run finished all 15,000 updates without an exception,
OOM, NaN, or manual interruption. It is the production continuation of the
[2026-08-17 preflight](2026-08-17_megapose_gso_geometry_pi3_finetune_ray_preflight.md).

## Setup

| Field | Value |
| --- | --- |
| Run | `megapose_gso_geometry_pi3_finetune_ray_n5k1_336_warmup1000_photoaug_20260817_102019` |
| Train config | `train_megapose_gso_geometry_pi3_finetune_ray_rtxpro6000_70gb_336x252` |
| Data config | `megapose_gso_geometry_n5_k1_masked` |
| Model | released full-size Pi3 checkpoint, frozen DINOv2-L encoder |
| New module | 402,432-parameter zero-initialized ray projection |
| Trainable parameters | 588,397,656 |
| Views | fixed N=5 references and K=1 query |
| Training mixture | 50% scene-to-scene, 50% render-to-scene |
| Input treatment | object-only reference and query RGB/depth; calibrated 4:3 crops |
| Train-only augmentation | role-consistent color/gamma/JPEG/blur plus crop-center and per-view focal augmentation |
| Resolution | 336x252 |
| Batch | 26 samples / 156 images per update |
| Schedule | 30 x 500 = 15,000 updates; OneCycle |
| Peak LR | `5e-6` historical Pi3 weights, `1e-5` ray projection |
| Losses | standard Pi3 point/camera losses; correspondence weight `0` |

The actual Hydra snapshot has `pct_start=1/30`, hence a 500-update warm-up.
The run name still says `warmup1000` because it was not renamed after the
configuration was changed from the preflight's 1,000-update value.

The run consumed 390,000 sampled six-view sequences, or 2.34 million model
images. TensorBoard recorded 7,540 render and 7,460 scene batches, a 50.27/49.73
percent split. Runtime was 15:11:40. Peak CUDA memory was 66,373 MiB allocated
and 66,984 MiB reserved under the 70 GiB allocator cap.

## Artifacts

| Artifact | Path |
| --- | --- |
| Main log | `/home/dtrofimov/repositories/Pi3/logs/megapose_gso_geometry_pi3_finetune_ray_n5k1_336_warmup1000_photoaug_20260817_102019.log` |
| Output root | `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_geometry_pi3_finetune_ray_n5k1_336_warmup1000_photoaug_20260817_102019` |
| TensorBoard | `<output root>/tensorboard/megapose_gso_geometry_pi3_finetune_ray_n5k1_336_warmup1000_photoaug_20260817_102019` |
| Best scene-loss checkpoint | `<output root>/ckpts/best_model` at update 15,000 / epoch 29 |
| Retained periodic checkpoints | updates 10,000, 12,500, and 15,000 (`checkpoint_19`, `checkpoint_24`, `checkpoint_29`) |

The event contains 421 aggregate scalar tags and 14 image tags. Each image tag
has six snapshots, at updates 500 through 13,000. Per-object cards were disabled
as intended.

## Optimization and loss curves

The first values below are the first complete training/validation epoch at
update 500, not the checkpoint's unadapted initialization.

| Metric | First | Best | Final |
| --- | ---: | ---: | ---: |
| Train loss | 0.56295 | **0.19528** (15,000) | **0.19528** |
| Train local points | 0.04885 | **0.01094** (15,000) | **0.01094** |
| Train translation loss | 0.04005 | **0.01410** (13,500) | 0.01415 |
| Train rotation loss | 1.13576 | **0.42864** (15,000) | **0.42864** |
| Scene validation loss | 0.35513 | **0.16759** (15,000) | **0.16759** |
| Render validation loss | 0.51211 | **0.20614** (14,000) | 0.20634 |

Train, scene-validation, and render-validation losses fell by 65.3%, 52.8%,
and 59.7%, respectively. Scene loss still decreased over every one of the final
five evaluations. Render loss was effectively flat after update 14,000.

## Held-out pose and geometry metrics

Every validation point contains one deterministic N=5/K=1 sample for each of
189 held-out objects. “Best” is selected independently for each metric, so the
entries do not describe one jointly best checkpoint.

| Metric | Scene: first / best (update) / final | Render: first / best (update) / final |
| --- | ---: | ---: |
| Median normalized query pose | 1.701d / **0.765d** (10,500) / 0.783d | 1.511d / **0.421d** (14,000) / 0.439d |
| Median query rotation | 59.46 / **14.85** (11,000) / 15.45 deg | 44.52 / **10.65** (14,500) / 11.01 deg |
| Median query translation | 0.3615 / **0.1620** (10,500) / 0.1701 m | 0.3365 / **0.0966** (13,500) / 0.0988 m |
| Query camera-center median | 0.8373 / **0.3172** (9,000) / 0.3415 m | 0.8449 / **0.2072** (14,000) / 0.2132 m |
| Recall below 0.5d | 7.41 / **34.39** (10,500) / 33.33% | 15.34 / **53.44** (13,000) / 52.91% |
| Strict ADD at 0.1d | 0 / **2.12** (12,000) / 2.12% | 0.53 / **5.82** (9,000) / 4.76% |
| ADD-S-for-all at 0.1d | 0.53 / **13.76** (14,500) / 13.76% | 4.23 / **22.22** (12,500) / 21.16% |
| Correspondence L2 median | 0.1417 / **0.0583** (12,500) / 0.0587 | 0.2407 / **0.0948** (10,000) / 0.0961 |
| Query-ray angular median | 1.6129 / **0.3056** (14,500) / 0.3159 deg | 2.5063 / **0.5088** (12,000) / 0.5294 deg |

No GSO symmetric IDs are configured. Therefore the official configured
ADD(-S) value is identical to strict ADD here. “ADD-S-for-all” is a useful
nearest-neighbor diagnostic, not a symmetry-correct official GSO score.

### Final distribution and alignment diagnostics

| Metric | Scene references | Render references |
| --- | ---: | ---: |
| Normalized pose median / mean / p90 | 0.783 / 1.134 / 2.445d | 0.439 / 0.810 / 1.785d |
| Query rotation median / mean | 15.45 / 42.07 deg | 11.01 / 31.89 deg |
| Query translation median / mean | 0.170 / 0.245 m | 0.099 / 0.175 m |
| Query camera-center median / mean | 0.342 / 0.599 m | 0.213 / 0.438 m |
| Reference camera rotation median / mean | 5.23 / 8.99 deg | 3.57 / 11.78 deg |
| Reference camera-center median / mean | 0.105 / 0.177 m | 0.026 / 0.063 m |
| Fitted query focal relative error, x/y | 3.66 / 3.26% | 3.61 / 3.61% |
| Fitted query principal-point error, x/y | 3.44 / 2.50 px | 3.42 / 2.72 px |
| Sim(3) underconstrained samples | 0 / 189 | 0 / 189 |

The median-to-mean and p90 gaps are large. This is a real failure tail: most
samples are substantially better than the mean, while a minority still has
large rotation and camera-center errors. The aggregate event cannot determine
whether this is a stable subset of difficult objects or difficult sampled
views, because this run intentionally stored neither per-object cards nor raw
prediction rows and evaluates only one fixed sample per object.

The ray adapter learned calibration well. Intrinsics fits were valid for 99.5%
or more of query/reference views, ray angular medians were below 0.53 degrees,
and focal/principal-point residuals were small. All Sim(3) alignments were
constrained. Those diagnostics do not support focal conditioning or an
underconstrained alignment as the dominant remaining problem.

## Comparison with the compact scratch run

The comparison uses the final values from the
[40,000-update compact scratch run](2026-08-15_megapose_gso_geometry_n5_k1_scratch_run.md).
It is not a single-variable architecture ablation: the models, LRs, batch sizes,
update budgets, and photometric augmentation differ. It nevertheless answers
whether released Pi3 geometry transfers to this dataset/task.

| Final metric | Scene: scratch -> full Pi3 | Render: scratch -> full Pi3 |
| --- | ---: | ---: |
| Validation loss | 0.4156 -> **0.1676** (-59.7%) | 0.5565 -> **0.2063** (-62.9%) |
| Median normalized pose | 1.651d -> **0.783d** (-52.6%) | 1.208d -> **0.439d** (-63.7%) |
| Median query rotation | 96.5 -> **15.5 deg** (-84.0%) | 60.0 -> **11.0 deg** (-81.7%) |
| Median query translation | 0.380 -> **0.170 m** (-55.2%) | 0.258 -> **0.099 m** (-61.7%) |
| Query camera-center median | 1.102 -> **0.342 m** (-69.0%) | 0.879 -> **0.213 m** (-75.7%) |
| Recall below 0.5d | 2.12 -> **33.33%** (+31.21 pp) | 19.58 -> **52.91%** (+33.33 pp) |
| Correspondence L2 median | 0.232 -> **0.059** (-74.7%) | 0.249 -> **0.096** (-61.4%) |

The full Pi3 result is much better after only 390,000 sampled sequences, versus
1.92 million in the scratch run. Its final query-ray error is actually higher
than the scratch model's 0.135/0.207-degree values, while its pose is far better.
That is further evidence that sub-degree ray reconstruction alone is not the
limiting capability; the transferred multi-view geometry/common-frame prior is
the important gain.

## Interpretation

This run demonstrates useful held-out-object generalization on the constrained
MegaPose-GSO task. It no longer looks like the scratch model's registration
failure: median rotation is 11-15 degrees, median translation is 9.9-17.0 cm,
and 33-53% of samples are below 0.5 object diameters. The data and crop/ray
pipeline are therefore learnable by released Pi3 weights.

It is not yet a solved 6-DoF system. Strict 0.1d ADD remains 2.1-4.8%, and the
large mean/p90 tail makes the average sample substantially worse than the
median. The strongest evidence about the residual bottleneck is:

- render references give much better pose than scene references even though
  render validation loss is higher;
- reference cameras align much more accurately than held-out query cameras;
- calibration/ray errors are already small and every Sim(3) fit is constrained;
- the correspondence diagnostic improves strongly, but it was metrics-only
  (`correspondence_weight=0`) and the standard loss does not directly optimize
  aligned object ADD or held-out query registration.

The likely remaining problem is the difficult tail of cross-view/common-frame
registration under partial, occluded scene observations, not a universal focal
or depth-scale failure. Scene references remain harder and noisier than clean
renders. Rotation also affects the camera-center recovered through pose
inversion, so the high camera-center mean is partly coupled to the rotation
tail rather than being a pure object-center translation failure.

The curves are close to a task-metric plateau but do not show harmful
overfitting. Most pose optima occur between updates 10,500 and 14,500, their
final regressions are only a few percent, strict/0.5d recall remains near its
best, and scene loss reaches a new minimum at the final update. The OneCycle LR
has decayed to effectively zero and its horizon is exhausted at update 15,000,
so this scheduler state cannot be usefully continued. A longer new schedule
might yield modest gains, but the evidence does not suggest that more identical
updates will remove the tail.

One optimizer diagnostic should be improved before attributing a future
plateau to LR: the logged gradient norm is measured after clipping and was at
the 1.0 cap on 99.93% of updates. This run was stable, but a future run should
log the pre-clip norm to show how strongly clipping constrains it.

## What this result does and does not establish

It establishes transfer to deterministic held-out MegaPose-GSO objects with
GT object masks, calibrated crops/rays, five selected references, and one
query. It does not establish unmasked detection robustness, real-image or
cross-dataset transfer, symmetry-correct GSO ADD-S, or robustness across
multiple reference/query selections per object.

The highest-value next evaluation is an offline failure export from
`checkpoint_29`: several deterministic samples per held-out object, with raw
object/view IDs and visibility, coverage, focal/crop, depth, alignment, and
pose residuals. That would identify whether the tail follows objects, source
scenes, coverage, visibility, or viewpoint. It should avoid restoring thousands
of TensorBoard per-object cards. After that, the clean comparisons are N/view
coverage at evaluation, render-versus-scene reference quality, and an explicit
query/reference registration or correspondence objective.
