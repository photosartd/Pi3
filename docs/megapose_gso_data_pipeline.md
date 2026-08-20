# MegaPose-GSO data pipeline: download to training

This is the single front-to-back path for MegaPose-GSO: acquire the scene
corpus, acquire the GSO meshes, render a clean reference bank, pack it, and
point a Pi3 training config at the result. Every step below is documented in
more depth at the linked page; this page only chains the commands together in
order and says what each one is for.

Two datasets are involved and they are **not** the same thing, despite living
under similar names:

- **`MegaPose-GSO-fixed`** -- the released scene corpus (RGB/depth/masks of
  GSO objects composited into cluttered scenes). It ships pre-packaged as
  uncompressed tar shards (`shard-000000.tar`, ...); nothing in this repo
  repacks it, only indexes it.
- **`MegaPose-GSO-assets`** -- everything derived or downloaded locally: the
  raw GSO meshes, the portable model catalogue, and the clean reference-render
  bank this repo generates and packs. The reference bank is the one part of
  this pipeline that starts as ~1.9 million loose files and needs the packing
  step below.

## 0. Environment

Preprocessing scripts use only the Python standard library plus NumPy/Pillow
and run fine in the main project environment:

```bash
conda activate pi3-lmgeo
```

The renderer (`megapose_gso_references.py`, `megapose_gso_validate.py`) needs
Panda3D and a separate, older Python; keep it in an isolated environment, not
`pi3-lmgeo` -- see [`render/README.md`](../datasets/preprocess/render/README.md#ground-truth-pose-validation).

## 1. Acquire the MegaPose-GSO-fixed scene shards

This step is external to this repository: `MegaPose-GSO-fixed`'s `shard-*.tar`
files and their depth-corruption manifests are the official released dataset,
downloaded once per cluster/workstation from wherever your group mirrors the
[MegaPose6D dataset release](https://github.com/megapose6d/megapose6d#dataset)
(review its dataset notice before downloading). Nothing here re-packages
them -- they arrive as tar shards already. Place them under one
`MegaPose-GSO-fixed/` root:

```text
MegaPose-GSO-fixed/
├── shard-000000.tar
├── shard-000001.tar
├── ...
└── GSO_broken_depth_maps/   # official depth-corruption manifests
```

A partial download (a subset of shards) works for smoke-testing every command
below; every command in this pipeline discovers whichever shards are present.

## 2. Index the scene shards

```bash
python datasets/preprocess/megapose_gso.py \
  --data-root /path/to/MegaPose-GSO-fixed
```

Builds `MegaPose-GSO-fixed/pi3_index/megapose_gso.sqlite` -- a byte-offset
index into the existing tar shards (see "Direct payload access" below) plus a
deterministic train/validation object+scene split
(`megapose_gso.splits.json`). Safe to rerun after adding more shards: unchanged
shards are skipped. Full detail, schema, and sampling queries:
[`datasets/preprocess/README.md`](../datasets/preprocess/README.md).

## 3. Download the GSO meshes

```bash
python datasets/preprocess/download/megapose_gso_meshes.py \
  --output-root /path/to/MegaPose-GSO-assets \
  --index-path /path/to/MegaPose-GSO-fixed/pi3_index/megapose_gso.sqlite \
  --include-bop-meshes \
  --include-pointclouds \
  --accept-license
```

Downloads the official 23.26 GiB Google Scanned Objects archive once and
extracts the textured (`models_normalized`), BOP-scaled (`models_bop-renderer_scale=0.1`),
and point-cloud representations needed later (~944 objects, a few thousand
files total, ~13 GiB -- not part of the file-count problem this pipeline
otherwise solves, and not currently repackaged). Resumable; review the
[MegaPose](https://github.com/megapose6d/megapose6d#dataset) and
[Google Scanned Objects](https://research.google/pubs/google-scanned-objects-a-high-quality-dataset-of-3d-scanned-household-items/)
dataset notices first. Full detail:
[`datasets/preprocess/download/README.md`](../datasets/preprocess/download/README.md).

## 4. Build the portable model catalogue

```bash
python datasets/preprocess/megapose_gso_models.py \
  --assets-root /path/to/MegaPose-GSO-assets \
  --index-path /path/to/MegaPose-GSO-fixed/pi3_index/megapose_gso.sqlite
```

Writes `pi3_gso_models.json`, exact BOP `models_info.json` diameters, and
`models_eval/obj_*.ply` -- the CAD catalogue `BOPObjectModelCatalog` and the
ADD/ADD-S metrics load at training/eval time. Does not touch the downloaded
meshes or the scene shards.

## 5. Validate the renderer against released ground truth

Do this once per renderer/mesh-scale change, in the isolated Panda3D
environment (not `pi3-lmgeo`):

```bash
MEGAPOSE_REPO=/path/to/megapose6d
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$MEGAPOSE_REPO/src" \
/path/to/envs/megapose/bin/python \
  datasets/preprocess/render/megapose_gso_validate.py \
  --data-root /path/to/MegaPose-GSO-fixed \
  --assets-root /path/to/MegaPose-GSO-assets \
  --output-dir /path/to/MegaPose-GSO-assets/render_validation/gt_pose_panda3d
```

Renders isolated objects at the indexed intrinsics/pose and compares against
the released mask/depth; `report.json` fails the command below full mask IoU
0.90 or median depth error 5 mm. Full detail, including the exact validated
MegaPose commit and environment:
[`render/README.md`](../datasets/preprocess/render/README.md#ground-truth-pose-validation).

## 6. Render the clean reference bank

Same isolated environment as step 5:

```bash
MEGAPOSE_REPO=/path/to/megapose6d
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$MEGAPOSE_REPO/src" \
/path/to/envs/megapose/bin/python -u \
  datasets/preprocess/render/megapose_gso_references.py \
  --assets-root /path/to/MegaPose-GSO-assets \
  --index-path /path/to/MegaPose-GSO-fixed/pi3_index/megapose_gso.sqlite \
  --output-root /path/to/MegaPose-GSO-assets/renders
```

Deterministic and resumable; a complete object with the same configuration
fingerprint is reused. Writes one numeric directory per object, each holding
loose `rgb/`, `depth/`, `mask/`, `mask_visib/` PNGs plus a handful of JSON
manifests -- at the default 512 views this is where the "too many small
files" problem actually comes from: 946 objects x (512 views x 4 payload
types + 3 manifests) = **1,937,093 files, 41 GB**, averaging ~21 KB each.
Multi-GPU launch and the exact throughput/memory numbers measured on this
workstation are in
[`render/README.md`](../datasets/preprocess/render/README.md#clean-reference-bank).

## 7. Pack the reference bank into tar shards and index it

This is the step that fixes step 6's file count. It does not touch the
renderer or the images themselves -- it repacks the already-rendered PNGs,
byte-for-byte, into a handful of uncompressed tar shards, and builds a SQLite
index recording each payload's shard + exact byte offset/size, the same
random-access pattern `datasets/preprocess/megapose_gso.py` already uses for
the pre-packaged scene shards in step 2:

```bash
python datasets/preprocess/render/object_reference_index.py \
  --bank-root /path/to/MegaPose-GSO-assets/renders \
  --namespace gso \
  --mapping-path /path/to/MegaPose-GSO-assets/gso_models.json \
  --required-object-ids-path /path/to/MegaPose-GSO-assets/models_eval/models_info.json \
  --verify-images all \
  --verify-workers 8
```

This writes `renders/shard-000000.tar`, `renders/shard-000001.tar`, ... (one
finished, temp-then-renamed file each, ~512 MiB/shard by default, an object
is never split across two shards) and
`renders/pi3_index/references.sqlite` (format `pi3_object_reference_index_v2`,
also temp-then-renamed). `--verify-images all` re-decodes every packed
payload read back through its real shard offset, so a successful run proves
the packed data is byte-correct, not just that the source PNGs were.

At the default 512-view/946-object scale this takes the render bank from
**1,937,093 files (41 GB) to about 80 files** -- a >99.99% file-count
reduction, which is the actual fix for inode-quota/backup-limited cluster
filesystems.

Once `--verify-images all` has passed, reclaim the now-redundant loose files:

```bash
python datasets/preprocess/render/object_reference_index.py \
  --bank-root /path/to/MegaPose-GSO-assets/renders \
  --namespace gso \
  --mapping-path /path/to/MegaPose-GSO-assets/gso_models.json \
  --required-object-ids-path /path/to/MegaPose-GSO-assets/models_eval/models_info.json \
  --verify-images all \
  --verify-workers 8 \
  --delete-source-after-verify
```

`--delete-source-after-verify` deletes each packed object's loose
`rgb/depth/mask/mask_visib` files and per-object JSON manifests (all of their
content already lives in the shards and the SQLite index); the bank-wide
`reference_bank.json` is kept. It is a separate, explicit flag rather than the
default specifically so a first migration can be inspected before anything is
deleted. Rerunning this command later (e.g. after rendering more objects) is
a full rebuild, not incremental, and also removes any stale `shard-*.tar`
left over from a previous, larger build at the same bank root.

Full flag reference and the `pi3_object_reference_index_v1` -> `v2` migration
note: [`render/README.md`](../datasets/preprocess/render/README.md#clean-reference-bank).

### Why tar shards + SQLite, not HDF5 or WebDataset

- **Not HDF5**: every payload here is a whole opaque PNG/JSON blob read as one
  unit -- there is no chunked/sliceable numeric array to justify HDF5's
  complexity, its extra dependency, or its known rough edges under concurrent
  multi-worker reads on shared/network filesystems. `os.pread` on an
  already-open file descriptor is exactly "byte offset into a flat file,"
  which tar+SQLite already gives for free.
- **Not the WebDataset library**: WebDataset is built for *sequential* shard
  streaming with shard-level shuffling. Every sampling policy here does
  random keyed lookup (`groups_for_object(object_id)`, jumping to an
  arbitrary object/view per training step), which WebDataset does not
  support directly -- you would still need to bolt a random-access index on
  top, which is exactly what step 2's and step 7's SQLite indexes already
  are, with exact rather than approximate random access.
- The result **is** "packed shards," just using the same custom
  indexed-tar-plus-`pread` reader this repository already trusts for the
  much larger scene corpus in step 2, instead of a second storage
  architecture.

## 8. Use it in a training/eval config

Nothing about steps 1-7 changes any dataset config, sampling policy, or
training code -- they only change how bytes are stored on disk, behind the
same `ObjectViewSource` interface. Compose the GSO geometry profile as usual,
e.g.:

```bash
python scripts/train_pi3.py \
  train=train_megapose_gso_geometry_pi3_finetune_ray_rtxpro6000_70gb_336x252 \
  data=megapose_gso_geometry_n5_k1_masked \
  gso.data_root=/path/to/MegaPose-GSO-fixed \
  gso.assets_root=/path/to/MegaPose-GSO-assets \
  gso.references_root=/path/to/MegaPose-GSO-assets/renders \
  name=my_run
```

`gso.references_root` points at the packed `renders/` directory from step 7;
`IndexedBOPReferenceSource` reads `renders/pi3_index/references.sqlite` and
`renders/shard-*.tar` and requires the packed `pi3_object_reference_index_v2`
format (a `pi3_object_reference_index_v1` index raises a clear error asking
you to rerun step 7). Sampling policies, crop/focal geometry, and the
render-to-scene/scene-to-scene protocols are documented in
[`docs/object_pose_composition.md`](object_pose_composition.md) and
[`docs/megapose_gso_geometry_index.md`](megapose_gso_geometry_index.md), and
are completely unaffected by steps 1-7's on-disk format.

## Command index

| Step | Script | Reads | Writes |
| --- | --- | --- | --- |
| 1 | (external download) | -- | `MegaPose-GSO-fixed/shard-*.tar` |
| 2 | `datasets/preprocess/megapose_gso.py` | fixed shards | `pi3_index/megapose_gso.sqlite`, `.splits.json` |
| 3 | `datasets/preprocess/download/megapose_gso_meshes.py` | official GSO archive | `google_scanned_objects/`, `gso_models.json` |
| 4 | `datasets/preprocess/megapose_gso_models.py` | meshes + scene index | `pi3_gso_models.json`, `models_eval/` |
| 5 | `datasets/preprocess/render/megapose_gso_validate.py` | meshes + scene index | `render_validation/` report |
| 6 | `datasets/preprocess/render/megapose_gso_references.py` | meshes | `renders/<object_id>/{rgb,depth,mask,mask_visib}/*.png` |
| 7 | `datasets/preprocess/render/object_reference_index.py` | step 6's loose files | `renders/shard-*.tar`, `renders/pi3_index/references.sqlite` |
| 8 | `scripts/train_pi3.py` | steps 2, 4, 7 | trained checkpoint |
