# MegaPose-GSO geometry index

`scripts/prepare_megapose_gso_geometry_index.py` builds resumable derived
indexes for query-first, one-reference-track sampling.  It does not modify or
duplicate the 511 GB source dataset.  The existing source SQLite database and
tar byte offsets remain authoritative.

The derived SQLite file stores searchable pose, crop and coverage metadata.
Packed visible-surface masks are appended to a separate memory-mapped file.
The surface atlas is deterministic and is reused by subsequent builds and
benchmarks.

## Environment

Use the project environment so the CUDA backend can import Torch:

```bash
conda activate pi3-lmgeo
```

The CPU multiprocessing backend is available with `--device cpu`, but the
CUDA backend is substantially faster on the local RTX PRO 6000 Blackwell.

## Benchmark

Run four progressively larger disposable pilots before committing to the full
index:

```bash
python scripts/prepare_megapose_gso_geometry_index.py benchmark \
  --surface-points 8192 \
  --frame-counts 64 256 1024 4096 \
  --workers 8 \
  --batch-frames 16 \
  --device cuda
```

The command reports both the naive last-pilot extrapolation and a linear fit
over the largest three pilots.  The latter removes fixed SQL/GPU startup cost.
Every pilot has its own tqdm progress bar.

## Build and resume

```bash
python scripts/prepare_megapose_gso_geometry_index.py build \
  --surface-points 8192 \
  --workers 8 \
  --batch-frames 16 \
  --device cuda
```

The defaults write:

- `pi3_index/megapose_gso_geometry_v1.sqlite` under the GSO data root;
- `megapose_gso_geometry_v1.sqlite.surface_bits` beside it;
- `pi3_index/gso_surface_atlas_8192.f32`.

Re-running the same command skips complete source frames.  A different split,
surface resolution, tolerance or crop policy must use a different `--output`;
the script refuses to mix incompatible fingerprints.

For a bounded pilot that can later be resumed:

```bash
python scripts/prepare_megapose_gso_geometry_index.py build --max-frames 8192
```

## Inspect progress

```bash
python scripts/prepare_megapose_gso_geometry_index.py summary \
  --index /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso_geometry_v1.sqlite
```

The summary includes frame/instance/track counts, crop feasibility, source
border truncation, and surface-coverage/visibility threshold counts.

## Capacity grid

The following counts actual constructible ordered reference-track/query-track
units for the requested `N` reference and `K` query counts:

```bash
python scripts/prepare_megapose_gso_geometry_index.py stats \
  --index /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso_geometry_v1.sqlite \
  --n 2 5 8 \
  --k 1 5 10 \
  --query-visibility 0.1 \
  --reference-visibility 0.3 \
  --union-surface 0.5 \
  --positive-angle 10 \
  --focal-tolerance 0.1 \
  --json-output /tmp/gso_capacity.json
```

By default, references may independently land anywhere in the query's focal
band, retaining Pi3-like between-frame focal variation.  Add
`--common-focal-target` to require one focal value shared by the selected
references.  In both modes, the crop-only feasibility interval guarantees
that the configured full-object bounding-box margin remains in the source
image.  Cropping never pretends that it can reduce focal length.

For the current GSO one-scene reference policy, the two focal modes are
expected to give the same capacity counts.  Frames in a physical scene share
their native intrinsics, so all crop-feasible reference intervals have the
same lower endpoint.  If each interval overlaps the query's tolerance band,
their joint intersection does as well.  The distinction remains useful for
future datasets with per-frame native intrinsics or reference sets spanning
different cameras.  Actual per-frame focal jitter is a dataset postprocessing
choice; this command only proves that a feasible crop/focal target exists.

Pi3's local `BaseDataset` samples `aug_focal` inside each view's crop/resize
call, so it can vary between frames.  The existing MegaPose-GSO configs set
`aug_focal: false`; consuming this index therefore still requires a later
dataset-policy integration.  Also, `aug_focal: 0.9` samples a crop scale in
`[0.9, 1.0]`, corresponding to an effective focal multiplier in approximately
`[1.0, 1.111]`; it is not a symmetric `0.9`--`1.1` focal multiplier.  The
index's `--focal-tolerance 0.1` instead expresses the query-relative target
band directly.

The reference subset is the deterministic maximum-marginal-coverage greedy
plan.  Consequently every counted pair is constructible, but the count is a
lower bound: a query-seeded alternative subset may produce additional valid
pairs.  References always come from a single physical `(scene_id, gt_id)`
track, and reference/query tracks must have different scene IDs.

The JSON output contains per-object counts.  Use `--include-per-object` only
when that large section should also be printed to the terminal.

## Runtime plan catalogues

Training does not greedily solve surface coverage on every `__getitem__` call.
Build compact immutable plan catalogues once:

```bash
python scripts/prepare_megapose_gso_geometry_index.py plan \
  --index /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso_geometry_v1.sqlite \
  --output /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso_geometry_train_n5_plans.sqlite \
  --n 5 --reference-visibility 0.3 --union-surface 0.5 \
  --reference-source-kind scene
```

Each scene plan is five unique frames from one physical `(scene_id, gt_id)`
track whose exact rasterized surface union is at least 50%. Query compatibility
is still evaluated at sampling time: reference and query scenes differ, at
least one reference direction is within 10 degrees, and all six crop-feasible
focal intervals admit a common target within ±10% of the query focal.

Validation requires a separate derived geometry index and plan catalogue so it
cannot accidentally select train scenes:

```bash
python scripts/prepare_megapose_gso_geometry_index.py build \
  --split val \
  --output /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso_geometry_val_v1.sqlite \
  --surface-points 8192 --workers 8 --batch-frames 16 --device cuda

python scripts/prepare_megapose_gso_geometry_index.py plan \
  --index /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso_geometry_val_v1.sqlite \
  --output /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso_geometry_val_n5_plans.sqlite \
  --n 5 --reference-visibility 0.3 --union-surface 0.5 \
  --reference-source-kind scene
```

The render bank uses the same exact mesh-surface test. Dense renders need many
positive-direction candidates, so their plan catalogue anchors variants on
different render views and fills the other four slots from a greedy coverage
support pool. Every retained variant is checked again against the exact 50%
union constraint.

```bash
python scripts/prepare_megapose_gso_geometry_index.py build-render \
  --surface-points 8192 --workers 8 --batch-frames 16 --device cuda

python scripts/prepare_megapose_gso_geometry_index.py plan \
  --index /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/renders/pi3_index/render_geometry_v1.sqlite \
  --output /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets/renders/pi3_index/render_geometry_n5_plans.sqlite \
  --n 5 --reference-visibility 0.3 --union-surface 0.5 \
  --reference-source-kind render --variants-per-track 512
```

## Dataset integration

The new behavior is opt-in through
`data=megapose_gso_geometry_n5_k1_masked`. Existing `ScenePairPolicy`,
`RenderToScenePolicy`, source readers, and legacy configs are unchanged.
`GeometryConstrainedScenePairPolicy` and
`GeometryConstrainedRenderToScenePolicy` attach a crop specification to the
ordinary source record. The generic object-view pipeline then:

1. loads and masks native RGB/depth with the existing source adapter;
2. applies one 4:3 crop consistently to RGB, depth, object mask, and `K`;
3. resizes to 336x252 with the existing Pi3 crop/resize stage;
4. reapplies object-only masking after interpolation to prevent RGB/depth
   halos outside the nearest-resized mask;
5. leaves `T_C_O` and `camera_pose` unchanged and derives the ray map from the
   updated intrinsics.

Training independently samples each of the six focal targets from their proven
common feasible interval and jitters each crop center inside the exact
object-margin-safe range. This retains bounded between-frame focal variation.
Validation uses the shared interval midpoint and no jitter. Both remain
crop-only operations: they never fabricate a focal reduction that would require
padding or new pixels.

To inspect the tensors before launching a job:

```bash
python scripts/visualize_megapose_gso_geometry_inputs.py \
  --component GSOSceneGeometryN5K1 --samples 3 \
  --output /tmp/pi3_gso_geometry_scene_inputs

python scripts/visualize_megapose_gso_geometry_inputs.py \
  --component GSORenderGeometryN5K1 --samples 3 \
  --output /tmp/pi3_gso_geometry_render_inputs
```

Each sheet shows the exact six RGB model inputs, metric object-only depths, and
ray maps derived from the post-crop intrinsics. Its JSON sidecar records crop
boxes, center shifts, focal ratios, scenes, visibility, coverage, and measured
positive angle, and the command fails if any background RGB/depth survives.
