"""Role-consistent RGB augmentation shared by object-centric datasets.

One recipe is sampled for every reference in a sample and an independent
recipe for every query.  Geometry, masks, and camera metadata never pass
through this module.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


_RESAMPLING = getattr(Image, "Resampling", Image)
PHOTOMETRIC_INTERPOLATIONS = (
    _RESAMPLING.LANCZOS,
    _RESAMPLING.BICUBIC,
    _RESAMPLING.BILINEAR,
)


def _pair(value, name, cast=float):
    values = tuple(cast(item) for item in value)
    if len(values) != 2 or values[1] < values[0]:
        raise ValueError(f"{name} must be an ordered two-value range")
    return values


def _adjust_hue(image, hue_delta):
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    hue_shift = int(round(float(hue_delta) * 255.0))
    hsv[..., 0] = (
        (hsv[..., 0].astype(np.int16) + hue_shift) % 256
    ).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _adjust_gamma(image, gamma):
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.clip(array ** float(gamma), 0.0, 1.0)
    return Image.fromarray(
        (array * 255.0 + 0.5).astype(np.uint8), mode="RGB"
    )


def apply_role_photometric_spec(image, spec):
    """Apply one already-sampled RGB recipe without touching other fields."""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")

    image = TF.adjust_brightness(image, spec["brightness"])
    image = TF.adjust_contrast(image, spec["contrast"])
    image = TF.adjust_saturation(image, spec["saturation"])
    image = _adjust_hue(image, spec["hue"])
    image = _adjust_gamma(image, spec["gamma"])

    if spec["jpeg_enabled"]:
        image_cv = np.asarray(image)[:, :, ::-1]
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            image_cv,
            [cv2.IMWRITE_JPEG_QUALITY, int(spec["jpeg_quality"])],
        )
        if not encoded_ok:
            raise RuntimeError("OpenCV failed to encode photometric JPEG")
        image_cv = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_cv is None:
            raise RuntimeError("OpenCV failed to decode photometric JPEG")
        image = Image.fromarray(image_cv[:, :, ::-1])

    if spec["blur_enabled"]:
        width, height = image.size
        ratio = float(spec["blur_resize_ratio"])
        resized_small = image.resize(
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            resample=_RESAMPLING.LANCZOS,
        )
        image = resized_small.resize(
            (width, height),
            resample=PHOTOMETRIC_INTERPOLATIONS[
                int(spec["blur_interpolation_index"])
            ],
        )

    return image


class RoleConsistentPhotometricAugmentation:
    """Sample and apply one RGB recipe per view role and sample.

    ``requested_enabled`` records configuration intent. ``enabled`` is true
    only in training mode, making a shared dataset definition safe if a caller
    changes its mode to validation.
    """

    def __init__(
        self,
        *,
        enabled=False,
        mode="train",
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3),
        saturation=(0.7, 1.3),
        hue=(-0.1, 0.1),
        gamma=(0.7, 1.3),
        jpeg_prob=0.5,
        jpeg_quality=(20, 100),
        blur_prob=0.5,
        blur_resize_ratio=(0.25, 1.0),
    ):
        self.requested_enabled = bool(enabled)
        self.enabled = bool(enabled) and str(mode) == "train"
        self.brightness = _pair(brightness, "photometric_brightness")
        self.contrast = _pair(contrast, "photometric_contrast")
        self.saturation = _pair(saturation, "photometric_saturation")
        self.hue = _pair(hue, "photometric_hue")
        self.gamma = _pair(gamma, "photometric_gamma")
        self.jpeg_quality = _pair(
            jpeg_quality, "photometric_jpeg_quality", int
        )
        self.blur_resize_ratio = _pair(
            blur_resize_ratio, "photometric_blur_resize_ratio"
        )
        self.jpeg_prob = float(jpeg_prob)
        self.blur_prob = float(blur_prob)
        if not 0.0 <= self.jpeg_prob <= 1.0:
            raise ValueError("photometric_jpeg_prob must be in [0, 1]")
        if not 0.0 <= self.blur_prob <= 1.0:
            raise ValueError("photometric_blur_prob must be in [0, 1]")
        if self.gamma[0] <= 0.0:
            raise ValueError("photometric_gamma values must be positive")
        if not -0.5 <= self.hue[0] <= self.hue[1] <= 0.5:
            raise ValueError("photometric_hue must stay inside [-0.5, 0.5]")
        if self.jpeg_quality[0] < 1 or self.jpeg_quality[1] > 101:
            raise ValueError("photometric_jpeg_quality must stay inside [1, 101]")
        if self.blur_resize_ratio[0] <= 0.0:
            raise ValueError("photometric_blur_resize_ratio must be positive")
        self.role_specs = {}

    def _sample_spec(self, rng):
        jpeg_enabled = bool(rng.random() < self.jpeg_prob)
        blur_enabled = bool(rng.random() < self.blur_prob)
        return {
            "brightness": float(rng.uniform(*self.brightness)),
            "contrast": float(rng.uniform(*self.contrast)),
            "saturation": float(rng.uniform(*self.saturation)),
            "hue": float(rng.uniform(*self.hue)),
            "gamma": float(rng.uniform(*self.gamma)),
            "jpeg_enabled": jpeg_enabled,
            # Preserve the historical LMGeo upper-exclusive sampling rule.
            "jpeg_quality": (
                int(rng.integers(*self.jpeg_quality))
                if jpeg_enabled
                else 100
            ),
            "blur_enabled": blur_enabled,
            "blur_resize_ratio": (
                float(rng.uniform(*self.blur_resize_ratio))
                if blur_enabled
                else 1.0
            ),
            "blur_interpolation_index": (
                int(rng.integers(0, len(PHOTOMETRIC_INTERPOLATIONS)))
                if blur_enabled
                else 0
            ),
        }

    def begin_sample(self, rng):
        if not self.enabled:
            self.role_specs = {}
            return
        self.role_specs = {
            "reference": self._sample_spec(rng),
            "query": self._sample_spec(rng),
        }

    def apply(self, image, view_role):
        spec = self.role_specs.get(str(view_role))
        if spec is None:
            return image
        return apply_role_photometric_spec(image, spec)
