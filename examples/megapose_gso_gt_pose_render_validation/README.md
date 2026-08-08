# MegaPose-GSO ground-truth-pose render examples

These six examples are curated from the successful 2026-08-07 Panda3D
validation run. They include three prepared-training objects and three held-out
validation objects, including thin, articulated-looking, and unusual-pose
geometry.

Open `overview.png` for the compact summary. Each object folder contains:

- `source_rgb.png`: original multi-object MegaPose dataset image;
- `dataset_full_mask.png` and `dataset_visible_mask.png`: released binary masks;
- `dataset_full_masked_rgb.png` and `dataset_visible_masked_rgb.png`: source RGB
  restricted to those mask supports;
- `rendered_rgb.png` and `rendered_mask.png`: isolated textured Panda3D render
  at the indexed intrinsics and metric `T_C_O`;
- `mask_comparison.png`: source, full-mask RGB, visible-mask RGB, and render;
- `comparison.png`: original four-panel contour and visible-depth comparison.

The full mask is used for silhouette IoU. Metric depth is compared only where
the released visible mask, source depth, and rendered depth are valid.

| Example | Split | Mask IoU | Median depth error |
|---|---:|---:|---:|
| `train_fraction_fun` | train | 0.9982 | 0.123 mm |
| `train_rocksmith_cable` | train | 0.9901 | 0.080 mm |
| `train_fire_engine` | train | 0.9950 | 0.135 mm |
| `val_lion` | val | 0.9929 | 0.106 mm |
| `val_barnyard_puzzle` | val | 0.9973 | 0.039 mm |
| `val_boat_shoe` | val | 0.9971 | 0.046 mm |

`manifest.json` records the exact source sample/object names and unrounded
metrics. The complete validation output remains outside Git under
`MegaPose-GSO-assets/render_validation/gt_pose_panda3d`.

Repository PNGs are globally ignored. To intentionally version this curated
set, use `git add -f examples/megapose_gso_gt_pose_render_validation`.
