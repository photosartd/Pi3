# 2026-08-20 GSO higher-resolution A40 memory/compute smoke

Status: complete. No training run yet -- this is the pre-flight sizing pass
for running the completed GSO N=5/K=1 recipe at higher resolution on Slurm
A40s, requested before committing GPU-hours to a real run.

## Question

The completed [15k-update 336x252 run](2026-08-18_megapose_gso_geometry_pi3_finetune_ray_run.md)
used 336x252 (24x18=432 patch tokens/view) on a 70 GiB workstation GPU. Before
repeating it at higher resolution on Slurm A40s (40 GiB), three things needed
answers: what MegaPose-GSO's native resolution actually is, whether Pi3's
architecture supports fine-tuning above its training resolution at all, and
how many images/GPU a 40 GiB A40 can actually hold at each candidate
resolution -- determined empirically, not guessed.

## Native resolution and candidate resolutions

MegaPose-GSO scene shards and the rendered reference bank are both 720x540.
720x540 is **not** divisible by Pi3's `patch_size=14` (720/14=51.43,
540/14=38.57), so it cannot be used directly. The nearest exact-4:3,
patch-14-aligned resolution is **728x546** (52x39 patches), +1.1%/+1.1% over
native -- effectively the original resolution.

| Resolution | Patches/view | Tokens ratio vs 336x252 |
| --- | ---: | ---: |
| 336x252 (established) | 24x18 = 432 | 1.00x |
| 560x420 (candidate) | 40x30 = 1200 | 2.78x |
| 728x546 (~native, candidate) | 52x39 = 2028 | 4.69x |

## Can Pi3 fine-tune above its training resolution?

Yes, architecturally, with normal caveats. Two independent mechanisms make
this safe:

- The frozen DINOv2 encoder (`pi3/models/dinov2/models/vision_transformer.py`)
  already implements `interpolate_pos_encoding` -- the standard, widely-used
  DINOv2 mechanism for handling input resolutions other than the one its
  absolute position embedding was trained at.
- The trainable decoder (`pi3/models/pi3.py`) uses `RoPE2D` rotary position
  encoding, not absolute embeddings -- RoPE generalizes to different sequence
  lengths/resolutions by construction, no interpolation needed.

The encoder is frozen in this recipe (`freeze_encoder: true`), so only the
already-resolution-flexible decoder and the ray adapter actually learn at the
new resolution; the encoder's zero-shot interpolated-position-embedding
behavior is the only thing not directly fine-tuned. This is a low-risk setup,
not a novel architectural stretch -- Pi3/VGGT-family models are designed for
variable input resolution.

## Method

`PI3_CUDA_MEMORY_LIMIT_GIB` (`scripts/train_pi3.py`) sets a genuine PyTorch
CUDA allocator cap, already used elsewhere in this repo to let a larger local
GPU emulate a smaller one. Set to 40, it lets this workstation's larger GPU
stand in for an A40. For each resolution: ran the real training config with
`test.iters_per_test=0`, `metrics.enabled=false`, `visuals.enabled=false`,
swept `train.max_img_per_gpu` (multiples of 6, one six-view N=5/K=1 sample
each) up/down until finding the boundary between OK and
`torch.OutOfMemoryError`. Validation batch size was sized separately (forward
only, no grad/optimizer state, cheaper per image) with one real validation
batch including metrics and visuals. Both new production config files were
then validated end to end (one train step, full validation with real
metrics/visuals) at their final settings.

One false start worth recording: the first 560x420/728x546 attempts all OOM'd
identically regardless of train batch size, including at the minimum possible
N=6. Root cause was not training memory at all -- `test.max_img_per_gpu` was
still inherited at 768 from the 336x252 base config, and a full validation
pass at 768 images and 2.78-4.69x the tokens/image reliably blew the 40 GiB
cap on its own. Fixed by scaling `test.max_img_per_gpu` down per resolution
before re-measuring the actual training ceiling.

## Results

| Resolution | Train max (OOM boundary) | Train max, GiB reserved | Production `train.max_img_per_gpu` | Validation budget used |
| --- | --- | ---: | --- | ---: |
| 336x252 | 84 images (14 samples) | 40.2 | 78 (13 samples) -- established, unchanged | 768 (established) |
| 560x420 | 30 images (5 samples) | 39.7 | **24 (4 samples)** | 132 |
| 728x546 | 18 images (3 samples) | 40.1 (hard 40 GiB cap) | **18 (3 samples)** | 132 |

728x546 was revised on 2026-08-20 after the initial pass: the first sizing
used a hard 40 GiB cap and shipped the extra-conservative 12 images (2
samples). Re-measured with a real train step followed by validation (so the
training optimizer state is genuinely resident during validation, unlike an
isolated eval-only check) against a 46 GiB cap -- this repo's own established
"realistic usable A40 memory" figure elsewhere (`train_*_a40_46gb` configs),
not a hard 40 GiB emulation -- 18 images train + 132 images validation
together peaked at **34.2 GiB**, comfortable margin. Production does not
hard-enforce any GPU memory cap; `PI3_CUDA_MEMORY_LIMIT_GIB` is a
smoke-testing tool only, never set in the actual Slurm launch command.

GPU-only compute time per image (wall time minus dataloader wait, from the
sweep's `[1/2]` step log lines) roughly tracks the token-count ratio, i.e.
close to linear in tokens rather than quadratic at these per-GPU sample
counts (N<=14 samples, global cross-view attention sequence length <=84
tokens x samples):

| Resolution | s/image (GPU-only) | Ratio vs 336x252 | Token-count ratio |
| --- | ---: | ---: | ---: |
| 336x252 | 0.0336 | 1.00x | 1.00x |
| 560x420 | 0.0923 | 2.75x | 2.78x |
| 728x546 | 0.1637 | 4.87x | 4.69x |

These are single/noisy 2-iteration measurements, not a rigorous benchmark,
but the close match to the analytical token-count ratio in both directions is
a real corroboration, not a coincidence.

**Practical reading**: 560x420 is the recommended ceiling for this A40
budget. It roughly triples compute per training sample (2.75-2.78x) for a
resolution increase that keeps a workable per-GPU sample count (4) and a
comfortable margin under the cap. 728x546 (near-native) works at 3
samples/GPU with real margin (34.2 GiB against a realistic ~46 GiB budget)
and roughly 4.7-4.9x compute per sample -- across the 4-GPU cap this Slurm
setup allows, that is 12 samples/step total, still below the 26 samples/step
the original 336x252 recipe's peak LR was tuned against. 560x420 across 4
GPUs gives 16 samples/step -- a smaller gap than 728x546's 12; matching or
exceeding 26 at either resolution would need gradient accumulation or more
GPUs than this cluster's per-job A40 cap.

## Configs produced

- [`configs/train/train_megapose_gso_geometry_pi3_finetune_ray_a40_40gb_560x420.yaml`](../../configs/train/train_megapose_gso_geometry_pi3_finetune_ray_a40_40gb_560x420.yaml) --
  recommended.
- [`configs/train/train_megapose_gso_geometry_pi3_finetune_ray_a40_40gb_728x546.yaml`](../../configs/train/train_megapose_gso_geometry_pi3_finetune_ray_a40_40gb_728x546.yaml) --
  near-native, tight margins, smaller effective batch; available if wanted
  despite the above.

Both inherit everything else (15,000-update schedule, optimizer/LR, ray
adapter, GSO N=5/K=1 scene+render mixture, both eval sets) unchanged from
`train_megapose_gso_geometry_pi3_finetune_ray_rtxpro6000_70gb_336x252`, only
overriding `train.resolution`, `train.max_img_per_gpu`, and
`test.max_img_per_gpu`.

## What this does not establish

This is a memory/compute sizing pass only -- no training happened, so it says
nothing about whether higher resolution actually improves the pose-error
numbers from the completed 336x252 run. It also does not resolve the
effective-batch-size question above; that is a training-dynamics decision,
not a memory one, and is left open for whoever launches the run.
