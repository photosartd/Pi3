import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "preprocess"
    / "render"
    / "megapose_gso_references.py"
)
SPEC = importlib.util.spec_from_file_location("megapose_gso_references", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
references = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(references)


class MegaPoseGsoReferenceRenderingTest(unittest.TestCase):
    def test_camera_bank_is_deterministic_right_handed_and_object_specific(self):
        kwargs = dict(
            num_views=32,
            sphere="full",
            radius_m=0.4,
            view_jitter_deg=2.0,
            radius_jitter_fraction=0.05,
            roll_mode="low_discrepancy",
            roll_jitter_deg=3.0,
            seed=2026,
        )
        first = references.generate_camera_bank(object_id=7, **kwargs)
        replay = references.generate_camera_bank(object_id=7, **kwargs)
        other = references.generate_camera_bank(object_id=8, **kwargs)
        np.testing.assert_allclose(first[0]["T_C_O"], replay[0]["T_C_O"])
        self.assertFalse(np.allclose(first[0]["T_C_O"], other[0]["T_C_O"]))
        centers = []
        for pose in first:
            T_C_O = pose["T_C_O"]
            center = np.asarray(pose["camera_center_O_m"])
            np.testing.assert_allclose(T_C_O[:3, :3] @ T_C_O[:3, :3].T, np.eye(3), atol=1e-7)
            self.assertAlmostEqual(np.linalg.det(T_C_O[:3, :3]), 1.0)
            np.testing.assert_allclose(
                T_C_O @ np.r_[center, 1.0], np.asarray((0.0, 0.0, 0.0, 1.0)), atol=1e-7
            )
            np.testing.assert_allclose(T_C_O[:2, 3], 0.0, atol=1e-7)
            self.assertAlmostEqual(T_C_O[2, 3], pose["radius_m"])
            centers.append(center)
        centers = np.asarray(centers)
        self.assertLess(centers[:, 2].min(), 0)
        self.assertGreater(centers[:, 2].max(), 0)

    def test_scaled_lmo_intrinsics_preserve_field_of_view(self):
        K = references.scaled_lmo_intrinsics(720, 540)
        np.testing.assert_allclose(K[0], references.LMO_K[0] * 1.125)
        np.testing.assert_allclose(K[1], references.LMO_K[1] * 1.125)
        np.testing.assert_allclose(K[2], (0.0, 0.0, 1.0))

    def test_sequential_order_preserves_poses_and_reduces_camera_jumps(self):
        bank = references.generate_camera_bank(
            num_views=64,
            sphere="full",
            radius_m=0.5,
            view_jitter_deg=2.0,
            radius_jitter_fraction=0.05,
            roll_mode="low_discrepancy",
            roll_jitter_deg=3.0,
            seed=2026,
            object_id=17,
        )
        ordered = references.sequential_camera_order(bank)
        replay = references.sequential_camera_order(bank)
        greedy_seed = references.sequential_camera_order(bank, two_opt_passes=0)
        self.assertEqual(
            [pose["coverage_view_id"] for pose in ordered],
            [pose["coverage_view_id"] for pose in replay],
        )
        self.assertEqual([pose["view_id"] for pose in ordered], list(range(64)))
        self.assertEqual(
            sorted(pose["coverage_view_id"] for pose in ordered), list(range(64))
        )

        def jumps(poses):
            rotations = np.asarray([pose["T_C_O"][:3, :3] for pose in poses])
            dots = np.einsum("nij,nij->n", rotations[:-1], rotations[1:])
            return np.arccos(np.clip((dots - 1.0) / 2.0, -1.0, 1.0))

        original_jumps = jumps(bank)
        greedy_jumps = jumps(greedy_seed)
        sequential_jumps = jumps(ordered)
        self.assertLess(sequential_jumps.mean(), original_jumps.mean())
        self.assertLess(sequential_jumps.max(), original_jumps.max())
        self.assertLessEqual(sequential_jumps.sum(), greedy_jumps.sum() + 1e-12)

    def test_look_at_basis_is_stable_near_optical_poles(self):
        for center in (
            np.asarray((1e-3, -2e-3, 0.5)),
            np.asarray((-1e-3, 2e-3, -0.5)),
        ):
            T_C_O = references.look_at_T_C_O(center, np.deg2rad(77.0))
            R_C_O = T_C_O[:3, :3]
            np.testing.assert_allclose(R_C_O @ R_C_O.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R_C_O), 1.0)
            np.testing.assert_allclose(
                T_C_O @ np.r_[center, 1.0],
                np.asarray((0.0, 0.0, 0.0, 1.0)),
                atol=1e-12,
            )

    def test_bop_metadata_uses_millimetres_and_inclusive_bbox(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:6, 3:8] = True
        T_C_O = np.eye(4)
        T_C_O[2, 3] = 0.4
        camera, gt, info = references.bop_frame_metadata(
            object_id=12,
            T_C_O=T_C_O,
            K=np.eye(3),
            mask=mask,
            depth_scale=1.0,
        )
        self.assertEqual(camera["depth_scale"], 1.0)
        self.assertEqual(gt[0]["obj_id"], 12)
        self.assertEqual(gt[0]["cam_t_m2c"], [0.0, 0.0, 400.0])
        self.assertEqual(info[0]["bbox_obj"], [3, 2, 5, 4])
        self.assertEqual(info[0]["px_count_all"], 20)
        self.assertEqual(info[0]["visib_fract"], 1.0)

    def test_complete_output_validation_checks_every_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fingerprint = "abc"
            references._write_json_atomic(
                root / "reference_manifest.json",
                {
                    "format": references.OBJECT_FORMAT,
                    "complete": True,
                    "config_fingerprint": fingerprint,
                    "view_count": 1,
                },
            )
            for path in references._expected_paths(root, 1):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertTrue(
                references.validate_object_output(
                    root, num_views=1, config_fingerprint=fingerprint
                )
            )
            (root / "depth/000000.png").unlink()
            self.assertFalse(
                references.validate_object_output(
                    root, num_views=1, config_fingerprint=fingerprint
                )
            )


if __name__ == "__main__":
    unittest.main()
