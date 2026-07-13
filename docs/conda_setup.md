# Conda environment setup

This is the tested environment setup for the training branch, including the
LMGeo training configuration. The same environment is used by the existing
`pi3-lmgeo` installation.

## Prerequisites

- Linux x86-64 and Conda or Miniconda.
- An NVIDIA GPU and a working NVIDIA driver visible through `nvidia-smi`.
- Driver 550.54.14 or newer is the recommended baseline for the CUDA 12.4
  runtime used by PyTorch 2.5.1. CUDA 12.x minor-version compatibility can work
  with older drivers, but using R550 or newer avoids its feature restrictions.
- Enough free disk space for the environment. The PyTorch wheel and its CUDA
  libraries require several GB.

The system CUDA toolkit and `nvcc` are not required. PyTorch installs the CUDA
12.4 runtime libraries into the environment. The CUDA version printed by
`nvidia-smi` is the newest CUDA version supported by the driver; it does not
need to equal PyTorch's runtime version.

## Create the environment

Run these commands from the repository root:

```bash
conda env create -f environment_lmgeo.yaml
conda activate pi3-lmgeo
```

The YAML installs Python 3.11 and then installs `requirements.txt` with pip.
That requirements file includes both the base Pi3 dependencies and the
additional training/LMGeo dependencies.

To synchronize an existing environment after either file changes:

```bash
conda env update -n pi3-lmgeo -f environment_lmgeo.yaml --prune
conda activate pi3-lmgeo
```

The Gradio demo is optional and has separate dependencies:

```bash
python -m pip install -r requirements_demo.txt
```

## Verify the installation

First check package consistency and the repository tests:

```bash
python -m pip check
python -m unittest discover -s tests -v
```

Then verify that PyTorch can execute a BF16 CUDA operation on the selected GPU:

```bash
python - <<'PY'
import torch

assert torch.__version__.startswith("2.5.1")
assert torch.version.cuda == "12.4"
assert torch.cuda.is_available()

device = torch.cuda.current_device()
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(device))
print("Compute capability:", torch.cuda.get_device_capability(device))

x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert y.isfinite().all()
print("BF16 CUDA check: OK")
PY
```

Finally, check the training and LMGeo imports:

```bash
python - <<'PY'
from datasets.lmgeo_dataset import LMGeoDataset
from pi3.models.pi3_training import Pi3
from trainers import Pi3Trainer

print("Pi3/LMGeo training imports: OK")
PY
```

To verify that a particular training configuration composes without starting a
training run:

```bash
python scripts/train_pi3.py --cfg job \
  train=train_lmgeo_finetune_lowres \
  data=lmgeo_all_trainpbr_test_bop
```

## GPU compatibility and launch settings

The environment uses PyTorch 2.5.1 with CUDA 12.4 and BF16 training. It does
not need a different dependency set for the GPUs below.

| GPU | CUDA compute capability | Environment change | Runtime/configuration notes |
| --- | --- | --- | --- |
| GeForce RTX 4090 (24 GB) | 8.9 | None; verified on this GPU | Use BF16 as configured. On a single-GPU workstation, pass `--num_processes 1`. Reduce the image budget if a workload runs out of memory. |
| NVIDIA RTX A6000 (48 GB, Ampere) | 8.6 | None, provided its host driver meets the requirement above | BF16 is supported. The same settings should fit because it has more memory than the 4090, although its performance characteristics differ. Do not confuse it with the newer RTX 6000 Ada. |
| NVIDIA H200 (141 GB, Hopper) | 9.0 | None, provided the server driver meets the requirement above | Keep BF16 unless the training code is explicitly changed to support another precision. For multi-GPU H200 nodes, use one Accelerate process per GPU and set `--num_processes`/`--num_machines` to match the allocation. NVSwitch/Fabric Manager and cluster drivers are system-administrator concerns, not Conda packages. |

For one GPU, adapt the README commands as follows:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_pi3.py train=<training-config> name=<run-name>
```

For multiple GPUs on one machine, set `--num_processes` to the number of GPUs.
For multiple machines, also set `--num_machines`, `--machine_rank`, and the
rendezvous/network parameters required by the cluster.

Changing GPU type does not require changing the model's learning rate. If the
additional memory is used to change the effective batch size, review the
learning rate and gradient accumulation as a separate training decision.

## Known optional components

Importing the Pi3 model currently prints:

```text
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
```

This is expected for this repository: the `models.curope` extension and its
build sources are not included. Pi3 automatically uses its functionally
equivalent PyTorch implementation. No additional installation is required.
`nvcc` is only needed if a buildable CUDA extension is added separately.

`pi3.utils.debug` additionally requires `debugpy`, but that helper is optional
and is not used by the training entry point. Install it only when remote Python
debugging is needed:

```bash
python -m pip install debugpy
```

## Reproduction record

This procedure was verified on 2026-07-13 by creating a separate clean
environment with:

```bash
conda env create -n pi3-lmgeo-verify -f environment_lmgeo.yaml
```

The host had an NVIDIA GeForce RTX 4090, driver 575.57.08, and no system
`nvcc`. The clean environment installed Python 3.11.15, PyTorch 2.5.1+cu124,
torchvision 0.20.1+cu124, and NumPy 1.26.4. Package checks, the repository test
suite, BF16 CUDA execution, training imports, and LMGeo Hydra configuration
composition all passed.

The pre-existing `pi3-lmgeo` environment and the clean environment used the
same pinned core versions. Only unpinned package patch releases differed at
verification time (`anyio`, `huggingface_hub`, `regex`, `timm`, and
`transformers`). This is expected while those requirements remain unpinned.

References:

- [PyTorch 2.5.1 CUDA 12.4 installation variants](https://pytorch.org/get-started/previous-versions/#v251)
- [CUDA 12.4 driver compatibility](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html#cuda-driver)
- [NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda-gpus)
