# 2026-08-14 MegaPose-GSO geometry N=5/K=1 one-batch overfit

Status: complete. This is a memorization/learnability control on the exact
training sample, not a held-out generalization result.

## Question and setup

Can the compact scratch Pi3 configuration learn the new calibrated-crop GSO
task at all when data diversity is removed?

- train config:
  `train_megapose_gso_geometry_small_scratch_ray_336x252_overfit_one_batch`;
- data config: `megapose_gso_geometry_n5_k1_masked_overfit_one_batch`;
- one object (`object_id=0`), one deterministic scene-to-scene sample;
- five reference views from scene 7973 and one query from scene 31;
- fixed first valid coverage plan/query, shared focal target, no crop-center or
  focal randomness, and byte-identical inputs across epochs;
- object-only RGB and depth for both roles, no visibility-mask conditioning;
- calibrated post-crop ray conditioning, 336x252 input resolution;
- pretrained frozen DINOv2-S encoder; 62,101,976 randomly initialized
  trainable decoder/head/ray parameters;
- batch size one sample (six images), bf16, AdamW, no weight decay;
- 600 updates (60 artificial epochs x 10 updates), validation every 50
  updates on the exact training sample.

The OneCycle peak LR was `1e-4` for the 61.95M decoder/head parameters and
`1e-3` for the 150,912-parameter zero-initialized ray adapter. This is larger
than the production scratch LR because the objective is fast memorization with
no generalization pressure, while still being conservative for a transformer
of this size. The schedule used 5% warmup, `div_factor=10`, and
`final_div_factor=100`.

## Results

| Update | Val loss | Local points | Translation loss | Rotation loss | Query rot. | Query trans. | Pose error / diameter | ADD@0.1d | ADD-S@0.1d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.73396 | 0.26696 | 2.866e-2 | 1.8041 | 69.301 deg | 1.2705 m | 5.3392d | 0 | 0 |
| 50 | 0.05820 | 0.04824 | 3.333e-4 | 0.06629 | 2.695 deg | 0.07972 m | 0.3351d | 0 | 0 |
| 100 | 0.03264 | 0.02504 | 2.121e-4 | 0.05477 | 1.535 deg | 0.04214 m | 0.1773d | 0 | 1 |
| 200 | 0.01492 | 0.01248 | 2.155e-5 | 0.02218 | 0.489 deg | 0.00488 m | 0.0206d | 1 | 1 |
| 400 | 0.00485 | 0.00429 | 9.815e-7 | 0.00550 | 0.152 deg | 0.00392 m | 0.0165d | 1 | 1 |
| 600 | 0.00297 | 0.00283 | 8.375e-9 | 0.00142 | 0.0488 deg | 0.000663 m | 0.00278d | 1 | 1 |

The exact-batch loss fell by 99.6%. The final object-pose translation error is
0.66 mm, while the aligned camera-center residual is 0.115 mm; these are
different quantities because the object-pose metric inverts the aligned query
camera transform. The query rotation metric reached 0.049 degrees. Strict ADD
first passed at update 200 and remained successful through the end.

Other independent diagnostics improved in the same direction:

- query/reference correspondence median L2: `1.0292 -> 0.0133`;
- query ray angular median: `21.38 -> 0.205` degrees;
- query ray reprojection median: `238.4 -> 2.17` pixels;
- query ray fitted focal error: `100.1% -> 1.41%` in x and
  `100.1% -> 7.01%` in y;
- query ray fitted principal-point error: `23.6 -> 0.87` px in x and
  `66.4 -> 4.09` px in y;
- reference camera-center median residual: `0.6984 m -> 0.000155 m`.

The aligned scale converged from an invalid negative random value to about
`1.169`. It need not converge to one because Pi3's common frame is
scale-ambiguous and evaluation solves a reference-only Sim(3).

## Conclusion

Yes: the complete constrained sampler, crop/intrinsics path, ray-conditioned
model, loss, and reference-only alignment metric can learn and nearly exactly
memorize this physical sample. The loss and metric curves occasionally wobble
around updates 200-300, but the low-LR tail resumes improvement and finishes
at sub-millimetre/sub-tenth-degree query pose error. This rules out a basic
incompatibility or an irreducible error floor in the implemented path.

It does **not** prove that the full scene/render mixture will generalize. This
control contains one object, one reference scene, one query scene, no
augmentation, and evaluates on the training input. The next production result
must still be judged on the held-out scene and render validation loaders.

## Artifacts

Run root:

```text
/media/internal/nvme/dtrofimov/runs/overfit_gso_geometry_scene_n5k1_small_ray_336_20260814
```

TensorBoard log directory:

```text
/media/internal/nvme/dtrofimov/runs/overfit_gso_geometry_scene_n5k1_small_ray_336_20260814/tensorboard/overfit_gso_geometry_scene_n5k1_small_ray_336_20260814
```

TensorBoard includes the step-0 input/depth/reconstruction panels, train
panels at updates 100-600, all loss curves, aligned object/camera metrics,
correspondence metrics, and ray-geometry metrics. The final and best states are
under `ckpts/checkpoint_59` and `ckpts/best_model`; the best loss occurred at
the final update. Total training time was 3 minutes 19 seconds on one RTX PRO
6000 Blackwell; steady-state updates took about 0.30 seconds and peak reserved
GPU memory was about 2.0 GiB.

## Launch

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py \
  train=train_megapose_gso_geometry_small_scratch_ray_336x252_overfit_one_batch \
  data=megapose_gso_geometry_n5_k1_masked_overfit_one_batch \
  name=overfit_gso_geometry_scene_n5k1_small_ray_336 \
  train.auto_resume=false
```

For an external run directory, also override `log.output_dir`,
`log.ckpt_dir`, `log.tensorboard_dir`, and `hydra.run.dir` as done in the
recorded run.
