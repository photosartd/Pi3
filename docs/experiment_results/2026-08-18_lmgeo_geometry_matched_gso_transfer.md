# 2026-08-18 geometry-matched GSO to LM-O transfer

Status: implementation validated and full evaluation complete.

## Question and setup

This run tests whether the completed full-Pi3 MegaPose-GSO model transfers to
LM-O `new_val` when the input construction, not merely the nominal N=5/K=1
count, matches the GSO geometry protocol.

| Field | Value |
| --- | --- |
| Checkpoint | GSO full Pi3+ray `checkpoint_29` |
| Query split | LM-O `new_val` |
| References | clean LM-O `train/<object_id>` bank |
| Views | N=5 references, K=1 query, fixed 336x252 |
| Constraints | reference visibility >=0.3, union surface coverage >=0.5, one positive <=10 degrees, common crop focal within query +10% |
| Inputs | visible-object-only RGB and metric depth; post-crop ray maps; no visibility-mask conditioning |
| Alignment | reference-only Sim(3), query GT used only for evaluation and oracle positive retrieval |
| Eligible samples | 1,364 / 1,600 ordinary LMGeo windows (85.25%) |

The 236 rejected windows all failed the query's full-object 5%-margin crop.
There were no missing geometry rows, missing reference plans, focal failures,
or positive-angle failures after crop feasibility.

The plan catalogue contains 4,087 variants across all eight LM-O objects. The
minimum N=5 union coverage by object ranges from 0.7635 to 0.9420, comfortably
above the 0.5 contract.

## Pre-evaluation distribution audit

LM-O is not grossly OOD in pose scale or intrinsics, so the evaluation is
meaningful. The important remaining shift is small query image occupancy.
GSO scene values below are a deterministic 259,513-row sample of the 6.75M
training geometry features; LM-O values cover all 33,442 indexed eligible
instances before the one-query-per-window selection.

| Raw scene/query quantity | GSO train | LM-O `new_val` | Interpretation |
| --- | ---: | ---: | --- |
| Object-center depth mean / median | 1.298 / 1.310 m | 1.033 / 1.040 m | same metre scale; LM-O is somewhat closer |
| Camera distance / object diameter median | 5.85d | 6.63d | modest shift, not an order-of-magnitude error |
| Camera distance / diameter p90 | 7.95d | 11.14d | LM-O has a clearly harder small-object tail |
| Geometric normalized focal | p10 1.104, median 2.385 | fixed 1.034 | LM-O lies just below the low GSO tail before cropping |
| Full-object bbox area median | 9.76% | 1.53% | main OOD axis: LM-O query objects are much smaller |
| Full-object bbox area p90 | 39.08% | 5.42% | even large LM-O cases are small by GSO standards |

LM-O train references use exactly the same native intrinsics as `new_val`:
`fx=572.4114`, `fy=573.57043`, `cx=325.2611`, `cy=242.0490` at 640x480.
The GSO reference renderer deliberately uses this same LM-O intrinsic profile.
LM-O reference cameras are at 0.4 m rather than GSO's 0.5 m nominal radius;
their median distance/object-diameter ratio is 2.51d. This is a moderate
reference-scale change, while the query/reference image-size gap is larger.

Four materialized checks shared target normalized focal 1.08549 and actual
integer-crop focal 1.08110 across all six views. Their positive angles were
4.73-9.52 degrees and union coverages 0.764-0.943. Query mask fractions were
0.6-1.8%, versus 3.3-13.5% for references. All objects remained visible and no
nonzero RGB/depth survived outside the final mask.

## Full evaluation results

The run evaluated all 1,364 samples in 11 batches. It completed in 87 seconds,
with 24,877 MiB maximum allocated and 31,522 MiB maximum reserved under the
70 GiB allocator cap. All 11 batches were eligible for object-pose, camera, and
ray metrics; no Sim(3) fit was underconstrained.

| Metric | LM-O matched | Final GSO render val | Difference |
| --- | ---: | ---: | ---: |
| Validation loss | 0.1679 | 0.2063 | LM-O lower |
| Median normalized query pose | 0.463d | 0.439d | +0.024d |
| Mean / p90 normalized pose | 0.826 / 2.026d | 0.810 / 1.785d | slightly heavier LM-O tail |
| Median query rotation | 13.14 deg | 11.01 deg | +2.13 deg |
| Median query translation | **0.0802 m** | 0.0988 m | LM-O 1.86 cm better |
| Median query camera-center error | 0.2051 m | 0.2132 m | similar, LM-O 0.8 cm better |
| Recall below 0.5d | 52.64% | 52.91% | effectively equal |
| Strict ADD below 0.1d | 3.01% | 4.76% | -1.75 pp |
| Official LM-O ADD(-S) below 0.1d | 6.09% | n/a | IDs 10/11 use ADD-S |
| ADD-S-for-all below 0.1d | 14.08% | 21.16% | -7.08 pp |

Mean query rotation is 33.34 degrees and mean translation is 0.1240 m, so the
familiar difficult tail remains. Query camera-center mean is 0.3940 m versus a
0.2051 m median. Nevertheless, this is not a transfer collapse: the central
pose distribution is strikingly close to held-out GSO render-to-scene.

Ray recovery is also healthy. Reference/query median angular errors are
0.408/0.685 degrees, query focal error is about 4.0-4.2%, and 98.75% of query
intrinsics fits are valid. Reference camera alignment is accurate (2.68-degree
rotation and 1.71-cm center medians), while query alignment is worse. These
metrics again point to held-out query/common-frame registration and the
small-object tail, not a depth-unit or calibration catastrophe.

## Conclusion

The matched experiment supports cross-dataset transfer. The model retains the
same roughly 53% recall below 0.5d and nearly the same median normalized pose
as on GSO render validation. LM-O does not exhibit a metre/millimetre error,
incompatible extrinsic convention, or unlearned ray calibration.

It does not show that retrieval is solved: GT query pose chooses an adequate
positive reference, and GT visible masks isolate every input object. It also
does not eliminate the main residual distribution gap: LM-O query objects are
much smaller and have a longer distance/diameter tail. A fair next ablation is
therefore object-scale matching or small-object augmentation, while keeping
this exact focal/coverage/view protocol fixed.

## Reference-depth scale follow-up

The matched LM-O evaluator now uses metric reference depth to resolve only the
global scale ambiguity. Rotation is still averaged from the predicted/GT
reference camera orientations, and translation is still fitted from the five
reference camera centers after applying that scale. Query depth is never used.
The historical camera-center scale estimator remains the default everywhere
else, so earlier LMGeo and GSO configs are unchanged.

For each reference, the depth scale is

`s_i = exp(median_pixels(log(GT_camera_depth) - log(predicted_local_z)))`,

and the final scalar is the equal-view median in log space. These are absolute
camera-origin depths. Centering each visible point cloud before fitting its
extent would make the estimate invariant to an independent translation per
view and is not the scale of one global Sim(3).

The full 1,364-query evaluation was repeated with the same checkpoint, samples,
batching, preprocessing, model predictions, and loss. As expected, validation
loss and rotation metrics are bit-identical; only quantities downstream of the
reference scale can change.

| Metric | Camera-center scale | Reference-depth scale | Change |
| --- | ---: | ---: | ---: |
| Validation loss | 0.167931 | 0.167931 | exactly unchanged |
| Median normalized query pose | 0.4627d | 0.4504d | -0.0123d |
| Mean / p90 normalized pose | 0.8264 / 2.0263d | 0.8481 / 2.1925d | tail worse |
| Median / mean query rotation | 13.136 / 33.337 deg | 13.136 / 33.337 deg | exactly unchanged |
| Median query translation | 0.08025 m | 0.08095 m | +0.00070 m |
| Mean query translation | 0.12403 m | 0.12659 m | +0.00256 m |
| Median query camera-center | 0.20509 m | 0.20732 m | +0.00224 m |
| Mean query camera-center | 0.39395 m | 0.39338 m | -0.00057 m |
| Recall below 0.5d | 52.64% | 53.59% | +0.95 pp |
| Recall below 1d / 2d | 74.93 / 89.74% | 74.12 / 88.64% | -0.81 / -1.10 pp |
| Official ADD(-S) below 0.1d | 6.09% | 5.87% | -0.22 pp |
| Median alignment scale | 0.62934 | 0.62312 | -0.99% |
| Median reference-center residual | 0.01711 m | 0.01823 m | +0.00112 m |

All 1,364 samples used all five reference depth maps; there were zero
fallbacks. Mean per-sample cross-reference scale log-MAD was 0.00787, roughly a
0.8% multiplicative spread. Thus reference depth supplies a stable scale, but
the old and new global scales already agree to about 1% in aggregate.

The result is deliberately mixed rather than a metric improvement claim. The
central normalized-pose statistic and 0.5d recall improve slightly, but
translation, strict recall, mean, and p90 do not. The camera-center estimator is
the least-squares scale for the reference cameras, so its slightly lower
reference-center residual is expected. More importantly, replacing its scale
does not reduce query-center error. This rules out noisy global Sim(3) scale as
the dominant LM-O failure: query-specific radial translation and inconsistency
between the point-depth and camera-translation heads remain.

## Reference/query geometry audit

The full split was evaluated once more with role-separated camera-center and
local-depth diagnostics. The same one scalar estimated from the five reference
depth maps is applied to every predicted local depth map. Query GT depth is
used only to measure the residual after alignment; it does not affect the
scale, Sim(3), predictions, or poses.

| Diagnostic | Five references | Query | Query/reference ratio |
| --- | ---: | ---: | ---: |
| Camera-center total, median | 0.01823 m | 0.20732 m | 11.4x |
| Camera-center radial component, median | 0.00755 m | 0.06909 m | 9.1x |
| Camera-center tangential component, median | 0.01474 m | 0.17522 m | 11.9x |
| Camera-radius error, median | 0.00750 m | 0.05170 m | 6.9x |
| Camera-direction error, median | 2.11 deg | 9.36 deg | 4.4x |
| Camera rotation, median | 2.68 deg | 13.14 deg | 4.9x |
| Local-depth absolute error, median | 0.00417 m | 0.04618 m | 11.1x |
| Local-depth MAE, mean | 0.00607 m | 0.09393 m | 15.5x |
| Local-depth relative error, median | 1.13% | 4.44% | 3.9x |
| Per-view depth-scale log error, median | 0.00648 | 0.04416 | 6.8x |

The reference numbers are deliberately optimistic because those same
references determine the alignment. Even with that caveat, the split shows a
real query-side generalization gap in both heads. The query local-depth median
scale discrepancy is only about `exp(0.0442)-1 = 4.5%`, however, whereas its
camera-center error is 20.7 cm and is dominated by a 17.5 cm tangential/orbit
component. A perfect scalar depth correction can address radial/range error;
it cannot rotate the predicted camera center around the object.

The query object-to-camera translation gives the same qualitative picture:
median optical-axis error is 0.05003 m and median lateral error is 0.04961 m
(the component medians do not sum to the 0.08095 m median total). Thus depth is
part of the query error, but not the single decisive missing quantity.

This is causal evidence only for the alignment intervention already tested:
replacing camera-derived scale with GT **reference** depth does not improve
query center or translation. It is not yet a causal test of feeding reference
or query depth into the transformer. Such a depth-conditioned model could use
depth for more than one scalar, but it requires a controlled training ablation.
Using query GT depth at evaluation would also be a different, depth-assisted
task rather than the present RGB+ray one-shot protocol.

## Query-occupancy failure-mode audit (2026-08-19)

The evaluator now records the exact visible-object fraction in the final
336x252 query tensor, after the geometry-planned crop and resize:

`occupancy = count(object_visibility_mask > 0.5) / (336 * 252)`.

This avoids the old pre-crop BOP bbox/patch-count approximation. The opt-in
object-pose extension logs 81 compact occupancy-bin scalars and writes one row
per query with pose, rotation, translation, camera-center decomposition,
visibility, and metric-depth errors. It also writes complete and
`visib_fract >= 0.7` summaries, correlations, object/bin counts, and a plot.
No query GT quantity changes inference or alignment.

The full 1,364-query evaluation was repeated from `checkpoint_29`. Standard
loss and aggregate metrics are identical to the preceding reference-depth run;
only diagnostic accumulation was added. All query identities are unique.

| Final visible query occupancy | Queries | Pose median / p90 | Recall <0.5d | Rotation median | Translation median | Camera-center median | Tangential center median | Local-depth abs. median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <0.5% | 467 | 1.143 / 3.452d | 23.13% | 34.60 deg | 0.168 m | 0.526 m | 0.400 m | 0.111 m |
| 0.5-1% | 401 | 0.505 / 1.370d | 48.63% | 13.59 deg | 0.082 m | 0.214 m | 0.189 m | 0.048 m |
| 1-2% | 337 | 0.271 / 0.629d | 81.90% | 7.38 deg | 0.057 m | 0.115 m | 0.103 m | 0.029 m |
| 2-4% | 116 | 0.193 / 0.402d | 95.69% | 4.85 deg | 0.045 m | 0.062 m | 0.056 m | 0.016 m |
| 4-8% | 35 | 0.153 / 0.309d | 97.14% | 4.17 deg | 0.037 m | 0.035 m | 0.031 m | 0.010 m |
| >=8% | 8 | 0.083 / 0.335d | 87.50% | 2.63 deg | 0.021 m | 0.017 m | 0.009 m | 0.010 m |

The final visible-occupancy median is only 0.716%; 34.24% of queries occupy
less than 0.5%, 63.64% less than 1%, and 88.34% less than 2%. The last bin has
only eight samples and should not be compared literally with the 4-8% bin, but
the well-populated first five bins show a large, nearly monotonic capability
curve.

Spearman correlation between log occupancy and error is strongly negative:

| Error | All queries | `visib_fract >= 0.7` | Pooled within-object ranks |
| --- | ---: | ---: | ---: |
| Normalized pose | -0.667 | -0.623 | -0.652 |
| Rotation | -0.586 | -0.525 | -0.538 |
| Translation | -0.604 | -0.533 | -0.645 |
| Camera center | **-0.715** | **-0.673** | **-0.705** |
| Tangential camera center | -0.648 | -0.609 | -0.628 |
| Local-depth absolute error | -0.595 | -0.564 | -0.597 |

All eight objects independently have negative occupancy/error correlations;
the median per-object coefficients are -0.671 for pose, -0.555 for rotation,
-0.665 for translation, and -0.699 for camera center. Occlusion therefore does
not explain the trend. Among 1,081 high-visibility queries, the <0.5% bin still
has 1.059d pose, 25.03-degree rotation, 0.146 m translation, and 0.406 m center
medians, versus 0.195d, 5.05 degrees, 0.046 m, and 0.062 m at 2-4%.

Small queries dominate the failure tail. Queries below 1% are 63.64% of the
dataset but account for 89.26% of failures above 0.5d, 96.88% above 1d, and
every failure above 2d. The <0.5% bin alone contains 89.68% of failures above
2d.

The evidence supports apparent query size as the strongest measured LM-O
failure axis. It affects rotation, translation, depth, and especially
tangential/common-frame camera placement together, consistent with too few
object tokens for stable cross-view registration. It remains observational:
occupancy covaries with distance and viewpoint. The high-visibility and
within-object controls make pure occlusion or object identity implausible as
the explanation, but a causal claim requires a controlled zoom/resolution or
small-object-training ablation.

## Offline-gallery scale bug and correction

The initially exported object-6 and object-8 orthographic cards showed the cyan
cloud approximately 25-34% too small despite plausible camera overlays. This
was an exporter coordinate-state bug, not evidence of an unmodelled LM-O or
two-scene training factor.

The model forward pass returns mutually consistent raw `local_points`, camera
translations, and composed world `points`. `Pi3Loss.normalize_pred` then
normalizes `local_points` and camera translations in place for its
scale-invariant loss. The trainer deliberately runs visualizers and metrics
before this loss mutation. The offline exporter had reversed that order:

`forward -> loss normalization -> metrics/visuals`.

Consequently, its depth scale was estimated from normalized local points while
the orthographic renderer consumed the still-raw composed `points`. If `n` is
the prediction normalization factor, the displayed cloud received `n*s`
instead of the intended raw-coordinate scale `s`.

| Sample | Buggy post-loss scale | Correct raw scale | Accidental factor | Visual effect |
| --- | ---: | ---: | ---: | ---: |
| object 6, scene 49/image 75 | 0.40138 | **0.60838** | 0.65976 | 34.0% shrink |
| object 8, scene 47/image 975 | 0.38609 | **0.51479** | 0.75001 | 25.0% shrink |

Those factors match the large apparent discrepancies. After moving offline
visuals and metrics before loss calculation, the cyan/orange object scales
agree closely. Object 8 nearly overlays in all three axes; object 6 retains
ordinary local-shape and multi-view registration residuals but no longer has a
gross scale mismatch.

The normal full validation metrics were never affected: `validate()` already
uses `forward -> visuals -> metrics -> loss`. Training and the checkpoint were
also unaffected. Only galleries produced by the old exporter order are
invalid for point-cloud scale inspection. A surface-centered scale experiment
was run while tracing the misleading image, but it changes the evaluation
definition and is not needed to fix this issue.

## Artifacts and validation

| Artifact | Path |
| --- | --- |
| Full output/TensorBoard | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_geometry_render_n5k1_gso_checkpoint_full_20260818` |
| Reference-depth output/TensorBoard | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_reference_depth_scale_full_20260818` |
| Role/depth audit output/TensorBoard | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_reference_depth_role_depth_full_20260818` |
| Query-occupancy output/TensorBoard | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_query_occupancy_full_20260819` |
| Query-occupancy tables/plot | `<query-occupancy output>/query_occupancy/lmo_new_val_geometry_render_n5_k1/step_00000030` |
| Corrected reference-depth 20-example gallery | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_geometry_matched_visual_gallery_reference_depth_consistent_20_20260818` |
| Surface-centered diagnostic output/TensorBoard | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_reference_depth_centered_scale_full_20260818` |
| Offline visual report | `/media/internal/nvme/dtrofimov/spott3r/outputs/lmgeo_new_val_geometry_render_n5_k1_preflight/visuals/lmgeo_matched_report.json` |
| Geometry indexes | `/vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/lm-o/pi3_index` |

Validation completed:

- focused LMGeo/geometry suite: 11/11 passed;
- full CPU suite: 209/209 passed;
- four real six-view visual/materialization assertions passed;
- one-batch and full GPU evaluation passed under the memory cap;
- TensorBoard contains all aggregate pose/camera/ray scalars and six expected
  image cards, with per-object cards disabled.
