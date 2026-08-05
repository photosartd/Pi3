# MegaPose-GSO one-time preprocessing

`megapose_gso.py` builds the lightweight random-access metadata and prepared
train/validation entity split used by
`MegaPoseGSOObjectDataset`. It does **not** extract or rewrite images,
depth, or masks, and it does not require the GSO meshes. Run it once after the
available `MegaPose-GSO-fixed` shards and their corruption manifests have been
downloaded. Rerunning the same command is safe: unchanged shards are skipped,
while a changed/new shard or changed manifest is indexed transactionally.

The script uses only the Python standard library. From the Pi3 repository:

```bash
python datasets/preprocess/megapose_gso.py \
  --data-root /vol/coro/dtrofimov/data/projects/gfm-6dof/datasets/MegaPose-GSO-fixed
```

The current project environment works as well:

```bash
conda activate pi3-lmgeo
python -m datasets.preprocess.megapose_gso \
  --data-root /path/to/MegaPose-GSO-fixed
```

No shard-count argument is needed. The command discovers every
`shard-*.tar`, so it works for a 10-shard local subset and for the complete
dataset. Copy the script (or pull this repository) on the training node and run
the same command after all desired shards are present. Adding shards later and
rerunning incrementally extends the index.

By default the generated files are:

```text
MegaPose-GSO-fixed/
├── shard-000000.tar
├── ...
├── GSO_broken_depth_maps/
└── pi3_index/
    ├── megapose_gso.sqlite
    ├── megapose_gso.splits.json
    └── megapose_gso.summary.json
```

The summary is human-readable and records the exact indexed shard IDs, counts,
whether this is a partial dataset, and how many shards were processed or
skipped in the latest invocation. The split JSON is a model-neutral
`pi3_entity_split_v1` manifest containing explicit train/validation object and
scene ID lists. The SQLite file and split manifest are the training inputs.
Keep all generated files with the dataset, not in Git.

The default seed is `2026`. Exactly 20% (rounded to the nearest entity) of
available object IDs and 20% of available scene IDs are selected by stable
seeded BLAKE2b ranking. Repeating preparation over the same indexed population
therefore writes identical lists. Training uses only train objects in train
scenes; validation uses only held-out target objects in held-out source scenes.
Crossed pairs (held-out target/train scene or train target/held-out scene) are
intentionally excluded.

Object exclusivity is target/supervision-level. MegaPose renders are
multi-object and this profile keeps full RGB, so a held-out object may still be
visible incidentally in a training scene while never being selected as its
target. Source scene IDs are fully disjoint. Pixel-level absence of held-out
objects would additionally require RGB masking or discarding every training
scene containing one, which is a substantially different and much smaller
training distribution.

The defaults can be changed explicitly when creating a different experimental
split:

```bash
python datasets/preprocess/megapose_gso.py \
  --data-root /path/to/MegaPose-GSO-fixed \
  --split-seed 77 \
  --validation-object-fraction 0.2 \
  --validation-scene-fraction 0.2
```

The loader rejects a missing or stale split manifest, and runtime split ratios
are not accepted. This prevents different workers or later runs from silently
using different partitions.

To intentionally recreate the index:

```bash
python datasets/preprocess/megapose_gso.py \
  --data-root /path/to/MegaPose-GSO-fixed \
  --rebuild
```

The official fixed-depth corruption manifests are important. By default a
missing per-shard manifest is an error. `--allow-missing-corruption-manifests`
is available for inspection-only data; its instance rows receive
`depth_corrupt=NULL` and are therefore excluded by the recommended
`depth_corrupt = 0` training filter.

## What the index contains

The source is BOP-WebDataset: each key is `SSSSSS_VVVVVV`, with camera JSON,
GT pose JSON, GT-info JSON, RGB JPEG, 16-bit depth PNG, and full/visible COCO
RLE mask JSON. The local files are 720×540 (4:3), and the released scenes have
40 independently rendered viewpoints. These are views of the same static
scene, not adjacent frames in a temporal video.

The normalized tables are:

- `shards`: source tar identity and resume signature, including its corruption
  manifest signature;
- `frames`: scene/view IDs, image size, intrinsics, optional `T_C_W`, depth
  scale, and tar byte offset/size for all seven source members;
- `instances`: one row per `(frame_id, gt_id)`, including object ID, metric
  `T_C_O`, visibility statistics/bboxes, and depth-corruption state;
- `scene_tracks`: stable physical instances identified by
  `(scene_id, gt_id)`, with available/visible/clean view counts;
- `scenes`: available scene views and whether all 40 are indexed;
- `objects`: observation, scene, and track counts for every object ID;
- `metadata`: schema, unit, and pose-convention declarations.

`gt_id` is essential: the same object ID can occur more than once in a scene.
Therefore `(scene_id, object_id)` does not uniquely identify a physical object,
whereas `(scene_id, gt_id)` does. The aggregate tables deliberately do not bake
in a visibility threshold; a training config can use, for example, 0.1 or 0.2
without reprocessing.

All pose blobs are little-endian float32 matrices in row-major order. In a
loader:

```python
T_C_O = np.frombuffer(row["T_C_O_f32"], dtype="<f4").reshape(4, 4).copy()
camera_pose = np.linalg.inv(T_C_O).astype(np.float32)  # T_O_C for Pi3
K = np.frombuffer(row["K_f32"], dtype="<f4").reshape(3, 3).copy()
```

`T_C_O` maps CAD/object coordinates to camera coordinates. Translations are
converted from BOP millimetres to metres during preprocessing. Metric depth is
`raw_uint16 * depth_unit_m`, where `depth_unit_m` is normally `0.0001` for this
dataset (`depth_scale_raw=0.1`). `T_C_W` is retained cheaply for future
scene-centric experiments, but the proposed object-centric regime does not use
it and does not depend on meshes.

## Sampling queries

Open read-only in worker processes and use named rows:

```python
uri = f"file:{index_path}?mode=ro&immutable=1"
db = sqlite3.connect(uri, uri=True)
db.row_factory = sqlite3.Row
```

All available views of a scene, with no image decoding:

```sql
SELECT * FROM frames WHERE scene_id = ? ORDER BY view_id;
```

All scenes/tracks containing an object with at least two clean visible views:

```sql
SELECT scene_id, gt_id, clean_visible_frame_count
FROM scene_tracks
WHERE object_id = ? AND clean_visible_frame_count >= 2
ORDER BY scene_id, gt_id;
```

Candidate records for one physical instance, applying the threshold at load
time:

```sql
SELECT i.*, f.frame_key, f.shard_id, f.K_f32, f.depth_unit_m,
       f.rgb_offset, f.rgb_size, f.depth_offset, f.depth_size,
       f.mask_visib_offset, f.mask_visib_size, s.relative_path
FROM instances AS i
JOIN frames AS f ON f.id = i.frame_id
JOIN shards AS s ON s.id = f.shard_id
WHERE i.scene_id = ? AND i.gt_id = ?
  AND i.depth_corrupt = 0 AND i.visib_fract >= ?
ORDER BY i.view_id;
```

For an anchored cross-scene sample, select an `object_id`, choose two distinct
rows from `scene_tracks`, and query each selected `(scene_id, gt_id)`. Within a
track, use `KeyQuerySamplingPolicy.select_records(..., allow_repeat=False)`.
Distinct scene rows plus distinct `frame_id`s guarantee no accidental view
repetition. Same-scene context is also possible whenever a track's clean view
count is at least the required K. With a partial shard download, many scenes
are incomplete; the counts make that limitation explicit rather than silently
repeating frames.

## Direct payload access and dataset adapter

Tar files are uncompressed. For each selected row the stored offset points
directly to the member data, so workers can use `os.pread` without scanning or
extracting the archive:

```python
fd = os.open(data_root / row["relative_path"], os.O_RDONLY)
try:
    rgb_bytes = os.pread(fd, row["rgb_size"], row["rgb_offset"])
finally:
    os.close(fd)
rgb = np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
```

Depth and mask payloads are loaded the same way. The visible mask JSON is a
list indexed by `gt_id`, matching the instance row. This keeps storage compact
and lets a loader decode only the records actually sampled.

The implemented `datasets.megapose_gso_dataset.MegaPoseGSOObjectDataset`
subclasses `ObjectDatasetAdapter`:

1. its sampling plan queries `scene_tracks`/`instances`, chooses distinct
   reference and query records, and passes them to `KeyQuerySamplingPolicy`;
2. `load_raw_object_view` seeks RGB, depth, and visible-mask payloads, decodes
   the chosen `gt_id`, and returns `RawObjectView` with metric depth, `T_C_O`,
   `camera_pose=inv(T_C_O)`, intrinsics, and object mask;
3. the inherited crop/resize stage converts 720×540 to configured Pi3 sizes
   such as 560×420 while adjusting intrinsics and keeping the 4:3 field of
   view. Preprocessing intentionally preserves source resolution.

SQLite is a good fit here because it is dependency-free, indexed, read-only
worker friendly, resumable per shard, and supports both directions needed for
training: scene→frames/tracks and object→scenes/tracks. Keeping byte ranges
avoids millions of extracted files and preserves flexibility for future depth,
mask, world-pose, or scene-context regimes without another source scan.

## Training with the index

Compose the provided mesh-free 560×420 profile before launching:

```bash
python scripts/train_pi3.py --cfg job \
  train=train_megapose_gso_finetune_560 \
  data=megapose_gso_cross_scene
```

The default `independent_scenes` regime selects reference/query observations
from distinct scenes. Train and validation load their prepared object and scene
lists from `megapose_gso.splits.json`; no entity is assigned by the loader at
runtime.

For a true two-scene anchor pair, use `sampling_regime=anchor_pair`. All
references then come from one stable `(scene_id, gt_id)` track and all queries
from another scene. A local one-reference/one-query diagnostic composes with:

```bash
python scripts/train_pi3.py --cfg job \
  train=train_megapose_gso_finetune_560 \
  data=megapose_gso_cross_scene \
  gso.sampling_regime=anchor_pair \
  gso_profile.num_reference_range='[1,1]' \
  gso_profile.num_query_range='[1,1]' \
  gso.eval_num_reference=1 gso.eval_num_query=1 gso.eval_image_num=2 \
  train.image_num_range='[2,2]' test.image_num_range='[2,2]'
```

The image corpus has no local GSO meshes, diameters, or symmetry declarations.
The dataset exposes object poses but marks the object-model capability
unavailable. Camera alignment, reference reconstruction, and correspondence
remain valid; ADD/ADD-S, diameter-normalized Chamfer, and CAD pose overlays are
routed away from these batches.

For strict two-scene anchor-pair training on CITEc A40 nodes, use the dedicated
profile, dataset config, environment block, smoke, and production commands in
[`docs/megapose_gso_a40.md`](../../docs/megapose_gso_a40.md).
