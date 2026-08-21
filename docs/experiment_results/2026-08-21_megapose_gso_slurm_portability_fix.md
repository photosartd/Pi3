# MegaPose-GSO geometry Slurm portability fix (2026-08-21)

## Failure sequence

The first cluster submissions exposed separate transfer and path issues in
dependency order:

1. `MegaPose-GSO-assets/models_eval/models_info.json` and the corresponding
   `google_scanned_objects/models_bop-renderer_scale=0.1` mesh targets were not
   initially present. Metrics and the dataset model catalogue require them.
2. The scene-side geometry SQLite files and N=5 plan catalogues under
   `MegaPose-GSO-fixed/pi3_index` were not initially transferred.
3. The Slurm launcher did not recognize `megapose_gso_geometry_*` as an
   asset-backed data family. Commit `af8f20f` added explicit Hydra overrides
   and basic existence checks for the six geometry/plan paths.
4. The remaining 2026-08-21 failure occurred after those overrides were visibly
   correct. `GeometryPlanIndex` opened the cluster plan catalogue, then trusted
   its embedded `geometry_index_path`, which was the absolute workstation path
   `/media/internal/nvme/...`. That metadata was written by
   `build_reference_plan_catalog`; it was not coming from Hydra or a leaked
   environment variable.

The packed reference index itself was not affected. Its `shards.relative_path`
values are relative to `gso.references_root`; its absolute `bank_root` metadata
is informational and unused by the runtime reader. The base scene index uses
relative TAR paths in the same way.

## Fix

- `GeometryPlanIndex` accepts an explicit reference geometry index. Every GSO
  scene/render train/validation policy now supplies the corresponding resolved
  Hydra path, so copied legacy metadata cannot override the selected runtime
  roots.
- Without an explicit path, relative metadata is resolved from the plan
  directory and a missing legacy absolute path falls back to a same-named
  sibling. This preserves old LM-O/GSO callers and copied v1 catalogues.
- New geometry sidecars and plan catalogues store relative `bits_path` and
  `geometry_index_path` metadata. Existing absolute sidecar metadata is also
  relocatable for analysis or plan regeneration.
- `scripts/validate_megapose_gso_geometry_runtime.py` performs a stdlib-only,
  read-only transfer audit. The submission wrapper runs it on the login node
  before requesting a GPU, and the batch script reruns it defensively.

No existing cluster SQLite file needs to be regenerated or edited. The cluster
only needs the complete files already identified by the validator and a repo
revision containing this fix.

## Validation

- focused geometry/config suite: 30/30 passed;
- complete CPU suite: 221/221 passed;
- shell syntax and Python compilation passed;
- the validator passed the actual local bundle: 1,040 scene TARs (506.9 GiB),
  70 reference TARs (39.2 GiB), 944 models, 483,328 reference views, and all
  three plan/geometry fingerprint pairs;
- real 560x420 materialization passed for both scene-to-scene and
  render-to-scene: six views each, correct reference/query ordering, and
  `(3, 420, 560)` tensors. Both policies reported `runtime override` as the
  selected plan geometry path source.
