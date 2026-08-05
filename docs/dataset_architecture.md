# Shared dataset architecture

Pi3 keeps the common scene-dataset contract small while allowing object-centric
datasets to add key/query sampling, canonical object poses, masks, and paired
views. The object layer is an upper set of `BaseDataset`; it does not make those
fields mandatory for CO3D, ScanNet, TartanAir, or another geometric dataset.

## Contracts

`BaseDataset` continues to consume one dictionary per view with:

- `img`, metric `depthmap`, `camera_intrinsics`, and `camera_pose`;
- `dataset`, `label`, and `instance` identities.

It derives `pts3d` and `valid_mask`. The contract and optional capabilities are
defined in `datasets/base/observation.py`.

Object adapters may additionally expose:

| Capability | Principal fields |
| --- | --- |
| `key_query` | `view_role`, `is_reference`, `is_query` |
| `object_pose` | metric `T_C_O`, `object_id`, object visibility mask |
| `object_model` | a locally available CAD model/diameter for the object ID |
| `visibility_condition` | condition and known maps |
| `correspondence` | key/query roles plus depth, intrinsics, and `T_C_O` |
| `paired_query` | pair index, variant flags, final homography, original pose |

The flags added by `BaseDataset` are descriptive. Models must continue using
the actual observation fields, and optional consumers must declare their
required capabilities.

An unavailable condition is represented as `[condition=0, known=0]`. It is not
represented as a known empty mask.

## Adding an object dataset

Subclass `ObjectDatasetAdapter` and normally implement only:

1. `load_raw_object_view(record)`, which decodes RGB, metric depth, object mask,
   intrinsics, metric `T_C_O`, and `camera_pose = inv(T_C_O)` into
   `RawObjectView`;
2. `build_sample_plan(index, resolution, rng)`, which returns reference and
   query records through `KeyQuerySamplingPolicy.plan`.

The inherited path applies masking, dataset geometry transforms, crop/resize,
role-level photometric augmentation, visibility conditioning, metadata
assembly, and finally `BaseDataset` point-map enrichment. Override
`assemble_processed_object_view` only when richer source metadata is needed.

```python
class MyObjectDataset(ObjectDatasetAdapter):
    def load_raw_object_view(self, record):
        return RawObjectView(
            rgb=...,
            depthmap=...,              # meters
            object_mask=...,
            camera_intrinsics=...,
            T_C_O=...,                 # object -> camera
            camera_pose=np.linalg.inv(T_C_O),
            record=record,
        )

    def build_sample_plan(self, index, resolution, rng):
        references, queries = self.index[index]
        return self.key_query_sampling_policy.plan(
            reference_records=references,
            query_records=queries,
            reference_rgb_masking=True,
            query_rgb_masking=False,
        )
```

`tests/test_dataset_architecture.py::_MinimalObjectAdapter` is an executable
minimal example.

## Processing plugins

`ObjectViewProcessor` executes `ObjectViewTransform` stages. Every stage
declares `requires` and `provides`; invalid order, missing dependencies, and
duplicate names fail during dataset construction.

The default stages are:

1. role/source-aware RGB and depth masking;
2. dataset geometry transformation;
3. crop/resize with the object mask carried through the same geometry;
4. role-level photometric augmentation;
5. condition/known-map construction.

LMGeo's recenter/zoom path remains a query-only geometry hook within the
geometry stage. Ray conditioning stays model-side because it must use final
intrinsics. GT correspondence construction stays loss-side.

## LMGeo compatibility

The existing public classes and Hydra targets are retained:

- `LMGeoDataset`;
- `LMGeoSequenceDataset`;
- `LMGeoAnchorScenePairSequenceDataset`;
- `LMGeoSameSceneCeilingDataset`;
- `LMGeoRecenterZoomSequenceDataset`.

Their selection logic now produces declarative `SamplePlan`s. Physical query
records carry a `model_view_cost`, so paired recenter/original queries still
consume two model views without changing dynamic budget semantics.

## MegaPose-GSO

`MegaPoseGSOObjectDataset` reads the SQLite index produced by
`datasets/preprocess/megapose_gso.py` and seeks RGB, depth, and uncompressed
COCO-RLE masks directly from the original tar shards. It never extracts the
corpus. Metric `T_C_O` is loaded from the index and inverted to Pi3's `T_O_C`.

The default `independent_scenes` regime makes every selected view come from a
different generated scene. The `anchor_pair` regime instead uses two stable
physical tracks: one scene for all references and another for all queries.
Both reject known corrupt-depth observations and sample without frame
repetition by default. Preprocessing writes deterministic 80/20 object and
scene lists to a common `pi3_entity_split_v1` JSON manifest. Training uses only
the train-target/train-scene intersection and validation only the held-out-
target/held-out-scene intersection, so supervised target IDs and source scene
IDs do not cross the split even though source shards are globally shuffled.
Because RGB is not object-masked, other held-out objects can still occur as
unsupervised background instances inside a training scene.

A MegaPose scene may contain several physical copies with the same GSO object
ID. The adapter always samples a concrete `(scene_id, gt_id)` track and decodes
that track's pose and RLE mask. If any active object/scene group contains more
than one such track, dataset construction requires query RGB masking or known
query-mask conditioning; an unmasked, unconditioned query is rejected as
target-ambiguous. Scene- and frame-level same-ID multiplicities are retained in
each view for inspection and visualization. The supplied cross-scene profile is
the mask-conditioned variant for both reference and query views. It therefore
assumes an instance-specific query mask is available (ground truth during this
training experiment, or an external segmentation source at deployment); it is
not an unconditioned-query evaluation profile.

The image release has poses but no local GSO meshes. Its views therefore have
`object_pose=true` and `object_model=false`. Mesh-dependent LM-O metrics and
overlays skip GSO batches, while camera alignment, correspondence, depth, and
reference reconstruction remain eligible.

## Mixing datasets

Weighted training configs with more than one dataset automatically use
`HomogeneousDynamicBatchSampler`. A dataset is selected before a batch is
filled:

```text
batch 0: LMGeo
batch 1: ScanNet
batch 2: LMGeo
batch 3: CO3D
```

No new mixture-specific Hydra syntax is needed. Put the adapters under the
existing weighted dataset section:

```yaml
train_dataset:
  length: auto
  weights:
    LMGeo: 1
    ScanNet: 3
  LMGeo:
    _target_: datasets.lmgeo_dataset.LMGeoSequenceDataset
    # ordinary LMGeo arguments
  ScanNet:
    _target_: datasets.scannet_dataset.ScannetDataset
    data_root: /path/to/scannet
    # ordinary ScanNet arguments
```

View count and component choice are synchronized across distributed ranks;
resolution may remain rank-local. Each component can narrow the global view
range through `supported_frame_counts`; LMGeo derives valid totals from its
reference/query ranges and paired-query view cost. The collator verifies that
samples within a batch have the same union schema and model-view count.
Optional fields may still differ between reference and query roles inside the
sample.

Heterogeneous samples inside one batch are intentionally not implemented. That
future extension will require a union-schema collator and per-sample capability
masks; the current explicit rejection prevents accidental partial collation.

## Model and consumer routing

`Pi3BatchAdapter` always stacks the core image/intrinsics input. When visibility
conditioning is enabled, general scene batches receive unknown zero maps.

Correspondence loss runs only on correspondence-capable batches. Metrics and
visualizers declare `required_capabilities`; their managers skip incompatible
batches and report `eligible_batches` and `skipped_batches`. TensorBoard also
receives a `train_dataset_batches/<dataset>` event for every optimizer batch.

## Validation gates

The architecture is covered by:

```bash
python -m unittest discover -s tests -v
```

The suite includes the minimal adapter, synthetic BOP processing, collation
guards, plugin dependencies, condition defaults, loss/metric routing,
distributed mixture sampling, and an end-to-end `create_dataloader` mixture.

For LMGeo changes, also compare a frozen legacy revision and the current class
on the same real sample. Compare selected IDs and all geometric/image fields,
then run a CUDA optimizer smoke with one object batch and one core-only scene
batch. The implementation validation for this migration used object 8,
`train_pbr` scene 20/subscene 29 at 224 px and the Pi3 base checkpoint.
