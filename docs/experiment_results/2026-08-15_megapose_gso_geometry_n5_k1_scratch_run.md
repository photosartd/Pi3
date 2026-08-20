# 2026-08-15 MegaPose-GSO geometry N=5/K=1 compact scratch run

Status: complete. The run finished all 40,000 updates without an exception,
OOM, NaN, or manual interruption.

## Setup

| Field | Value |
| --- | --- |
| Run | `megapose_gso_geometry_small_scratch_ray_n5k1_336_warmup500_no_perobj_20260814_163218` |
| Train config | `train_megapose_gso_geometry_small_scratch_ray_rtxpro6000_70gb_336x252` |
| Data config | `megapose_gso_geometry_n5_k1_masked` |
| Model | frozen pretrained DINOv2-S plus random compact Pi3 decoder/heads |
| Conditioning | zero-initialized ray projection; no visibility-mask branch |
| Views | fixed N=5 references and K=1 query |
| Training mixture | 50% scene-to-scene, 50% render-to-scene |
| Resolution | 336x252, calibrated 4:3 object crops |
| Batch | 48 samples / 288 images per update |
| Schedule | 80 x 500 = 40,000 updates; 500-update warm-up |
| Peak LR | `1e-4` decoder/head and ray projection |
| Trainable parameters | approximately 62.1M |

The run consumed 1.92 million sampled six-view sequences, or 11.52 million
model images. The observed mixture was 20,023 render batches and 19,977 scene
batches. Runtime was 18:58:58. Peak CUDA memory was 42,357 MiB allocated and
43,794 MiB reserved.

## Artifacts

| Artifact | Path |
| --- | --- |
| Main log | `/home/dtrofimov/repositories/Pi3/logs/megapose_gso_geometry_small_scratch_ray_n5k1_336_warmup500_no_perobj_20260814_163218.log` |
| Output root | `/media/internal/nvme/dtrofimov/spott3r/outputs/megapose_gso_geometry_small_scratch_ray_n5k1_336_warmup500_no_perobj_20260814_163218` |
| TensorBoard | `<output root>/tensorboard/megapose_gso_geometry_small_scratch_ray_n5k1_336_warmup500_no_perobj_20260814_163218` |
| Best-loss checkpoint | `<output root>/ckpts/best_model` at epoch 72 |
| Retained periodic checkpoints | epochs 59, 69, and 79 |

The TensorBoard event contains 421 aggregate scalar tags and 14 image tags.
Per-object metric cards were disabled as intended.

## Optimization and validation losses

| Metric | First (epoch 0) | Best | Final (epoch 79) |
| --- | ---: | ---: | ---: |
| Train loss | 0.9560 | 0.3478 (e77) | 0.3546 |
| Train local points | 0.1206 | 0.00957 (e77) | 0.01045 |
| Train translation loss | 0.0627 | 0.0262 (e77) | 0.0264 |
| Train rotation loss | 2.086 rad | 0.766 rad (e77) | 0.807 rad |
| Scene validation loss | 0.7380 | 0.4141 (e72) | 0.4156 |
| Render validation loss | 0.9697 | 0.5560 (e76) | 0.5565 |

The saved `best_model` uses the configured primary scene-to-scene validation
loss. Individual pose metrics reached their optima earlier, generally between
epochs 51 and 70, so the lowest-loss checkpoint is not automatically the best
pose checkpoint.

## Held-out pose and geometry metrics

Every validation point aggregates one fixed sample for each of 189 held-out
objects. Values marked best are independent per-metric optima.

| Metric | Scene: first / best / final | Render: first / best / final |
| --- | ---: | ---: |
| Median normalized query pose | 4.211d / 1.621d / 1.651d | 2.907d / 1.116d / 1.208d |
| Median query rotation | 122.1 / 92.5 / 96.5 deg | 130.1 / 58.6 / 60.0 deg |
| Median query translation | 0.914 / 0.363 / 0.380 m | 0.645 / 0.234 / 0.258 m |
| Query camera-center median | 1.141 / 1.042 / 1.102 m | 1.474 / 0.858 / 0.879 m |
| Recall at 0.5d | 0 / 4.23 / 2.12% | 1.06 / 21.16 / 19.58% |
| Correspondence L2 median | 0.468 / 0.229 / 0.232 | 0.609 / 0.244 / 0.249 |
| Query-ray angular median | 3.025 / 0.127 / 0.135 deg | 3.106 / 0.190 / 0.207 deg |

Strict ADD at 0.1d stayed at zero for scene references and finished at 0.53%
for render references. ADD-S computed for every object peaked at 2.12% and
5.82%, respectively, so unknown GSO symmetry labels cannot by themselves
explain the weak strict score.

## Interpretation

The run learned local calibrated geometry but did not learn reliable
query-to-reference registration on held-out objects:

- local-point loss fell by 69-75%; query-ray angular error reached 0.13-0.21
  degrees, reprojection reached roughly 1-1.6 pixels, and fitted focal error was
  approximately 1%;
- reference camera rotation improved to approximately 19-20 degrees, while
  query rotation remained at 60 degrees with clean render references and 96
  degrees with scene references;
- scene-to-scene query camera-center error barely improved, and geometric
  correspondence stalled near 0.23-0.25;
- render references produced much better query pose than scene references even
  though their aggregate loss was higher. This is direct evidence that the
  training loss is not a sufficient checkpoint-selection proxy for object pose.

The most probable bottleneck is cross-view object-part correspondence/common-
frame registration, amplified by partial/occluded scene references and queries.
The frozen generic DINOv2-S features and random compact decoder must learn that
transfer without an explicit correspondence or global-point loss. A gross
intrinsics failure is not supported by the ray diagnostics, although the
scale-aligned local loss alone cannot formally exclude a self-consistent depth
annotation error.

The curves plateau around epochs 60-72 and slightly regress in pose while loss
still improves. More updates with the same model/objective are unlikely to
remove the registration failure. The next controlled comparison therefore
restores the released full Pi3 geometry weights and introduces only the
zero-initialized ray adapter.
