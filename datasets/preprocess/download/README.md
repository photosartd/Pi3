# MegaPose-GSO mesh download

`megapose_gso_meshes.py` downloads MegaPose's prepared Google Scanned Objects
archive and the numeric `obj_id` to GSO-directory mapping. It is separate from
scene preprocessing: it does not modify `shard-*.tar` or `pi3_index`.

Review the [MegaPose dataset notice](https://github.com/megapose6d/megapose6d#dataset)
and the [Google Scanned Objects dataset page](https://research.google/pubs/google-scanned-objects-a-high-quality-dataset-of-3d-scanned-household-items/)
before downloading. `--accept-license` records that this check is the user's
responsibility.

## Recommended use

Extract the three useful compact representations for objects occurring in an
already prepared Pi3 dataset:

```bash
python datasets/preprocess/download/megapose_gso_meshes.py \
  --output-root /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-assets \
  --index-path /media/internal/nvme/dtrofimov/datasets/MegaPose-GSO-fixed/pi3_index/megapose_gso.sqlite \
  --include-bop-meshes \
  --include-pointclouds \
  --accept-license
```

The destination is arbitrary and may be shared by local and cluster runs. The
command is safe to rerun: complete downloads and extracted files with matching
sizes are reused. A `.part` download is resumed when the server supports HTTP
Range; otherwise it is restarted cleanly. Keep at least 2 GiB free beyond the
reported download/extraction requirement.

Default output:

```text
MegaPose-GSO-assets/
├── downloads/google_scanned_objects.zip
├── gso_models.json
├── google_scanned_objects/
│   ├── invalid_meshes.json
│   ├── models_normalized/<gso_id>/meshes/
│       ├── model.obj
│       ├── model.mtl
│       └── texture.png
│   ├── models_bop-renderer_scale=0.1/<gso_id>/meshes/model.ply
│   └── models_pointcloud/<gso_id>/meshes/model.obj
└── megapose_gso_meshes.download.json
```

The official ZIP is 23.26 GiB. `models_normalized` is needed for textured
reference rendering. The two optional compact forms add only about 602 MiB for
all 944 objects: BOP-scaled PLYs support diameters and ADD/ADD-S, while the
point-cloud OBJs preserve a convenient sampled surface representation. Avoid
`--extract all`; it additionally retains large representations that this
pipeline does not use. Once the manifest and some meshes have been checked, add
`--delete-archive-after-extract` on a rerun to recover the ZIP space.
Object filtering reduces extracted storage, not network traffic: the official
server publishes one ZIP, so the complete 23.26 GiB archive is downloaded
before selected members can be extracted.

## Useful options

Download and validate the archive without extracting it:

```bash
python datasets/preprocess/download/megapose_gso_meshes.py \
  --output-root /path/to/MegaPose-GSO-assets \
  --extract none --accept-license
```

Extract only a few objects for a renderer smoke test:

```bash
python datasets/preprocess/download/megapose_gso_meshes.py \
  --output-root /path/to/MegaPose-GSO-assets \
  --object-id 0 --object-id 1 \
  --include-bop-meshes --include-pointclouds \
  --accept-license
```

`--archive-url` and `--mapping-url` accept a local institutional mirror. A
repackaged archive can use `--expected-archive-bytes 0` only after independent
verification. `--force-download` replaces existing downloads. Run
`python datasets/preprocess/download/megapose_gso_meshes.py --help` for the
complete CLI.

Important geometry details:

- scene `obj_id` values are zero-based numeric IDs; always resolve them through
  `gso_models.json`, never directory sorting;
- reference rendering should use the textured `models_normalized` OBJ and
  apply MegaPose's `0.1` mesh scale;
- the Pi3 training environment does not need to load these meshes once RGB,
  depth, masks, intrinsics, and rigid `T_C_O` reference renders are prepared.

The renderer and render-to-scene dataset integration are separate steps; this
script intentionally only acquires and validates their immutable mesh inputs.
Useful MegaPose source keywords for that next step are `gso.normalized`,
`GoogleScannedObjectDataset`, `RigidObjectDataset`, `Panda3dSceneRenderer`,
`GSO_SCALE`, and `models_bop-renderer_scale=0.1`.
