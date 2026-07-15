# Checkpoints

This repo uses the original Pi3 checkpoint from Hugging Face:

```text
repo: yyfz233/Pi3
file: model.safetensors
url:  https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
```

The upstream Pi3 model card says that if automatic Hugging Face download is
slow, the checkpoint can be downloaded manually and passed via `--ckpt`. The
training configs in this repo expect the same `model.safetensors` file.

## Local Repo Copy

From the repo root:

```bash
conda activate pi3-lmgeo
mkdir -p ckpts/Pi3

python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

target_dir = Path("ckpts/Pi3")
target_dir.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id="yyfz233/Pi3",
    filename="model.safetensors",
    local_dir=target_dir,
)
print(path)
PY
```

Equivalent direct download:

```bash
mkdir -p ckpts/Pi3
curl -L --fail --retry 5 \
  -o ckpts/Pi3/model.safetensors \
  https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
```

Expected local path:

```text
ckpts/Pi3/model.safetensors
```

That is the default path used by the LMGeo configs, for example
`configs/train/train_lmgeo_finetune_518_a40_dynamic.yaml`.

## Shared CITEc Copy

On the CITEc cluster, prefer storing the base checkpoint on shared `/vol`
storage, close to the dataset and run outputs:

```bash
conda activate pi3-lmgeo
mkdir -p /vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base

python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

target_dir = Path("/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base")
target_dir.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id="yyfz233/Pi3",
    filename="model.safetensors",
    local_dir=target_dir,
)
print(path)
PY
```

The CITEc Slurm script will use this shared checkpoint automatically if it
exists:

```text
/vol/coro/dtrofimov/data/projects/gfm-6dof/checkpoints/pi3/base/model.safetensors
```

If the checkpoint lives somewhere else, pass it explicitly:

```bash
PI3_CKPT=/path/to/model.safetensors \
  scripts/slurm/submit_citec_lmgeo_518_a40_dynamic.sh smoke
```

## Verify The File

Quick structural check without loading all tensors into RAM:

```bash
python - <<'PY'
from safetensors import safe_open

path = "ckpts/Pi3/model.safetensors"
with safe_open(path, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())

print("tensor count:", len(keys))
print("first tensor:", keys[0])
PY
```

The current Pi3 checkpoint used here reports `tensor count: 1210`.

Checksum for the copy used during the 518px A40 probes:

```bash
sha256sum ckpts/Pi3/model.safetensors
```

```text
33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d
```

If Hugging Face updates the `main` file later, the checksum can differ. For
reproducing the local memory probes exactly, keep a copy with the checksum
above.
