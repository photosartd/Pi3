import os
import sys
sys.path.append('.')

import hydra
import torch
import trainers


def configure_cuda_memory_limit_from_env():
    """Optionally cap PyTorch's CUDA allocator for reproducible memory smokes.

    ``PI3_CUDA_MEMORY_LIMIT_GIB`` is intentionally opt-in and does not alter
    ordinary training. It lets a larger GPU exercise a smaller GPU profile
    under the smaller device's allocator ceiling.
    """

    value = os.environ.get("PI3_CUDA_MEMORY_LIMIT_GIB")
    if value is None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("PI3_CUDA_MEMORY_LIMIT_GIB requires a CUDA device")

    limit_gib = float(value)
    if limit_gib <= 0:
        raise ValueError("PI3_CUDA_MEMORY_LIMIT_GIB must be positive")

    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    limit_bytes = limit_gib * 1024**3
    if limit_bytes > total_bytes:
        raise ValueError(
            f"Requested CUDA allocator limit {limit_gib:.2f} GiB exceeds "
            f"device {device_index} capacity {total_bytes / 1024**3:.2f} GiB"
        )

    torch.cuda.set_per_process_memory_fraction(limit_bytes / total_bytes, device_index)
    print(
        f"CUDA allocator limit: {limit_gib:.2f} GiB on device {device_index} "
        f"({torch.cuda.get_device_name(device_index)})"
    )

@hydra.main(version_base="1.2", config_path="../configs", config_name="default")
def main(hydra_cfg):
    configure_cuda_memory_limit_from_env()
    trainer = eval(hydra_cfg.trainer)(hydra_cfg)
    trainer.train()

if __name__ == '__main__':
    main()
