# MegaPose-GSO rendering

## Clean reference bank

`megapose_gso_references.py` renders clean, split-neutral onboarding/reference
views. Each numeric object directory has the same internal BOP contract as
`lm-o/train/<object_id>`, but the bank itself is deliberately not named
`train` or `test`:

```text
reference_renders/
├── reference_bank.json
├── 000000/
│   ├── rgb/000000.png
│   ├── depth/000000.png
│   ├── mask/000000_000000.png
│   ├── mask_visib/000000_000000.png
│   ├── scene_camera.json
│   ├── scene_gt.json
│   ├── scene_gt_info.json
│   └── reference_manifest.json
└── ...
```

The recommended full command on this workstation is:

```bash
MEGAPOSE_REPO=/tmp/megapose6d
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$MEGAPOSE_REPO/src" \
/media/internal/nvme/shared_conda/envs/megapose/bin/python -u \
  datasets/preprocess/render/megapose_gso_references.py \
  --assets-root /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets \
  --index-path /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso.sqlite \
  --output-root /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/reference_renders
```

The command is deterministic and resumable. A complete object with the same
configuration fingerprint is checked and reused. A different camera/render
configuration is rejected at the same output root, preventing mixed banks.
Use repeated `--object-id` or `--max-objects` for a smoke test.

For a new bank whose numeric frame IDs already form a smooth evaluation
trajectory, add `--view-order sequential`. The renderer keeps the same uniform
full pose set, orders it using camera SO(3) distance (therefore accounting for
both sphere motion and in-plane roll), and only then assigns filenames
`000000...`. Each pose retains its original uniform-coverage index as
`coverage_view_id` in `reference_manifest.json`. Use a new output root because
ordering is part of the bank fingerprint. The default `coverage` order remains
for backward-compatible resumption of existing banks.

After the full bank completes, validate it and build the compact worker-facing
index once:

```bash
python datasets/preprocess/render/object_reference_index.py \
  --bank-root /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/renders \
  --namespace gso \
  --mapping-path /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/gso_models.json \
  --required-object-ids-path /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/models_eval/models_info.json \
  --verify-images sample \
  --verify-workers 8
```

This writes `renders/pi3_index/references.sqlite` atomically. Omit
`--allow-incomplete` for the production index: every validated model-catalogue
object and every configured view must be complete. The raw GSO mapping includes
source entries rejected during model preparation, so it is not itself the
required render set. `--verify-images all` performs a final full PNG decode;
`--verify-workers` only parallelizes those independent decodes.
Training details are in `docs/object_pose_composition.md`.

Defaults are 256 views, 720×540, scaled LM-O intrinsics and 0.5 m nominal
camera distance. Camera positions use a full Fibonacci sphere rather than the
LM-O upper hemisphere because MegaPose's object-relative query rotations cover
arbitrary SO(3). A low-discrepancy in-plane roll sequence supplies the third
rotation degree of freedom. Per-object deterministic perturbations are ±2° in
view direction, ±5% in distance and ±3° around each base roll. This avoids an
identical pose lattice across every object without making the bank
irreproducible. Every applied perturbation and exact metric `T_C_O` is stored in
the object manifest. The 0.5 m distance includes enough vertical field-of-view
margin for the largest normalized GSO bounding boxes even at the nearest -5%
distance perturbation; 0.4 m can clip box-diagonal views of larger objects.

The [BOP format documentation](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md#acquisition-of-training-images)
describes the classic training views as recursively subdivided icosahedron
samples. The local LM-O bank has 1,313 upper-hemisphere views per object. That
is unnecessarily large here: Pi3 consumes only 5–16 reference images in one
sample, and 256 full-sphere views already give dense candidate coverage. Use
512 only as a later reference-density ablation; do not change the production
bank in place.

The output conventions are:

- `T_C_O` maps centered GSO/CAD coordinates to an OpenCV camera frame;
- `camera_pose = inv(T_C_O) = T_O_C`;
- BOP translations and uint16 depth are in millimetres with `depth_scale=1`;
- RGB uses the original normalized GSO texture, fixed studio lighting and a
  black background;
- clean `mask` and `mask_visib` are identical and fully visible.

On the RTX PRO 6000 workstation, 32 simultaneous Panda3D cameras plus eight
PNG writers measured 115–136 views/s and 1.17 GiB peak host RSS in the initial
0.4 m smoke. Batch sizes 64 and 128 were not faster. The EGL driver was
confirmed as NVIDIA 580.95.05; `p3headlessgl` is selected explicitly so
headless jobs exit cleanly. Six complete 256-view objects occupy 197 MiB
(23–58 MiB/object), so budget roughly 30–50 GiB for all 944 objects at 256
views. At the measured throughput, raw single-GPU render/write time is about
35 minutes, with model loading and filesystem variability on top.

The corrected 0.5 m framing was additionally stress-tested on all 512 views of
the ten largest-diameter models: all 5,120 views passed the boundary check at
148–160 views/s with 1.22 GiB peak host RSS. The largest catalogue bounding
sphere has a 1.08° vertical angular margin at the nearest -5% distance jitter;
the former 0.4 m default had a negative 4.65° margin and could clip those views.

Each completed object is removed from the scene and its prepared geometry and
4K texture are explicitly released from Panda3D's GPU resource pools. A real
40-object/128-view EGL test held the renderer at exactly 1,155 MiB of GPU memory
when sampled after 6, 19 and 34 objects; the earlier dictionary-only eviction
grew by roughly 77–90 MiB per object because `noCache=True` does not bypass
Panda3D's global texture pool.

Parallelize across GPUs by object shard, with one process and one EGL context
per GPU. This is a complete four-GPU launcher:

```bash
MEGAPOSE_REPO=/tmp/megapose6d
GSO_ASSETS=/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets
GSO_INDEX=/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso.sqlite
GSO_REFERENCES="$GSO_ASSETS/reference_renders"
mkdir -p "$GSO_REFERENCES"

for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$MEGAPOSE_REPO/src" \
  /media/internal/nvme/shared_conda/envs/megapose/bin/python -u \
    datasets/preprocess/render/megapose_gso_references.py \
    --assets-root "$GSO_ASSETS" \
    --index-path "$GSO_INDEX" \
    --output-root "$GSO_REFERENCES" \
    --object-shard-count 4 \
    --object-shard-index "$GPU" \
    > "$GSO_REFERENCES/shard-$GPU.log" 2>&1 &
done
wait
```

The object shards are disjoint and may share the output root. Do not start
multiple renderer processes on the same GPU; camera batching already exposes
parallelism without duplicating mesh textures and EGL contexts.

The current `anchor_pair` MegaPose-GSO configs still take references from a
rendered scene track. Generating this bank does not silently change training;
the next integration step is an explicit dataset option that uses these clean
object folders as references while retaining indexed MegaPose scenes as
queries.

## Ground-truth-pose validation

`megapose_gso_validate.py` validates the downloaded textured meshes before a
clean reference bank is generated. It renders isolated objects with the exact
indexed intrinsics and metric `T_C_O`, then compares the silhouette with the
released full mask and depth with the released depth over visible pixels.

Keep renderer dependencies outside `pi3-lmgeo`. The official MegaPose checkout
uses Python 3.9 and lists Panda3D/Panda3D-GLTF in
`conda/environment_full.yaml`; its Dockerfile additionally documents a patched
Panda3D build for headless EGL. On this workstation the existing isolated
environment is:

```text
/media/internal/nvme/shared_conda/envs/megapose
```

It already contains Python 3.9.23, Panda3D 1.10.16, Panda3D-GLTF 1.3.0,
NumPy 1.26.4, SciPy 1.13.1, Trimesh 4.12.2, and PyTorch 2.8.0+cu128. No package
was installed into or removed from `pi3-lmgeo`. To reproduce the exact source
checkout used here:

```bash
git clone https://github.com/megapose6d/megapose6d.git /tmp/megapose6d
git -C /tmp/megapose6d checkout f3b8e1247f133f3d098833a251b8f2d744c03e1f
```

Run against a clean MegaPose checkout rather than changing Pi3's environment:

```bash
MEGAPOSE_REPO=/path/to/clean/megapose6d
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$MEGAPOSE_REPO/src" \
/media/internal/nvme/shared_conda/envs/megapose/bin/python \
  datasets/preprocess/render/megapose_gso_validate.py \
  --data-root /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed \
  --assets-root /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets \
  --output-dir /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/render_validation/gt_pose_panda3d
```

The default deterministic protocol selects five high-visibility objects from
each prepared object split. Each sample directory contains source/rendered RGB,
a contour/depth comparison panel, and compressed geometry arrays. `report.json`
records the exact environment and fails the command when any sample has full
mask IoU below 0.90 or median visible-depth error above 5 mm.

RGB pixels are not expected to match: the released scene used BlenderProc,
random materials/lights, backgrounds, and occluders, while validation uses an
isolated ambient-lit Panda3D render. Silhouette and metric depth are the tests
of mesh scale, object pose, camera convention, and intrinsics.

## Validated workstation result

The complete protocol above was run on 2026-08-07 on five deterministic train
objects and five validation objects. All 10 passed: minimum/median full-mask
IoU was 0.9901/0.9972, the median of per-image median depth errors was 0.063 mm,
and the worst per-image median was 0.135 mm. The comparison panels also showed
matching texture orientation and object pose. Results are stored at:

```text
/media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/
  render_validation/gt_pose_panda3d/report.json
```

The tested MegaPose commit contains a typo in its optional
`render_binary_mask=True` path (`rendering` versus `rendering_n`). This script
does not patch the external checkout: it requests depth and applies the exact
intended definition, `rendered_mask = rendered_depth > 0`. The first validation
run used the wheel's GLX pipe: it wrote correct renders and passed every metric,
but Panda3D could abort during interpreter teardown. Both scripts now select
the wheel's included `p3headlessgl` EGL pipe by default and explicitly release
camera buffers. On this workstation that path uses the NVIDIA driver and exits
cleanly; `--display-backend pandagl` remains available only for diagnosis.
