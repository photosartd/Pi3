import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "preprocess"
    / "megapose_gso_models.py"
)
SPEC = importlib.util.spec_from_file_location("megapose_gso_models", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)


def write_ascii_ply(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "element face 0",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(" ".join(map(str, vertex)) for vertex in vertices)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class MegaPoseGsoModelsPreprocessTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _make_dataset(self):
        assets = self.root / "assets"
        gso_id = "cube_object"
        mesh_root = assets / "google_scanned_objects"
        normalized = mesh_root / "models_normalized" / gso_id / "meshes"
        normalized.mkdir(parents=True)
        (normalized / "model.obj").write_text("mtllib model.mtl\n", encoding="ascii")
        (normalized / "model.mtl").write_text("map_Kd texture.png\n", encoding="ascii")
        (normalized / "texture.png").write_bytes(b"texture")
        pointcloud = mesh_root / "models_pointcloud" / gso_id / "meshes"
        pointcloud.mkdir(parents=True)
        (pointcloud / "model.obj").write_text("v 0 0 0\n", encoding="ascii")
        cube = np.asarray(
            [
                [x, y, z]
                for x in (-100.0, 100.0)
                for y in (-100.0, 100.0)
                for z in (-100.0, 100.0)
            ],
            dtype=np.float64,
        )
        bop_ply = (
            mesh_root
            / "models_bop-renderer_scale=0.1"
            / gso_id
            / "meshes"
            / "model.ply"
        )
        write_ascii_ply(bop_ply, cube)
        (assets / "gso_models.json").write_text(
            json.dumps([{"obj_id": 0, "gso_id": gso_id}]), encoding="utf-8"
        )

        index = self.root / "pi3_index" / "megapose_gso.sqlite"
        index.parent.mkdir()
        with sqlite3.connect(index) as connection:
            connection.execute("CREATE TABLE objects (object_id INTEGER PRIMARY KEY)")
            connection.execute("INSERT INTO objects VALUES (0)")
        split = index.with_name("megapose_gso.splits.json")
        split.write_text(
            json.dumps(
                {
                    "format": "pi3_entity_split_v1",
                    "splits": {
                        "train": {"object_ids": [0], "scene_ids": [0]},
                        "val": {"object_ids": [], "scene_ids": []},
                    },
                }
            ),
            encoding="utf-8",
        )
        return assets, index, split, bop_ply

    def test_exact_point_diameter_handles_full_and_degenerate_geometry(self):
        cube = np.asarray(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
            dtype=np.float64,
        )
        self.assertAlmostEqual(models.exact_point_diameter(cube), np.sqrt(12.0))
        line = np.asarray([[0, 0, 0], [2, 0, 0], [1, 0, 0]], dtype=np.float64)
        self.assertAlmostEqual(models.exact_point_diameter(line), 2.0)

    def test_prepare_models_writes_portable_catalog_and_bop_view(self):
        assets, index, split, bop_ply = self._make_dataset()
        catalog = models.prepare_models(
            assets_root=assets,
            index_path=index,
            split_path=split,
        )
        self.assertEqual(catalog["format"], models.CATALOG_FORMAT)
        self.assertEqual(catalog["object_count"], 1)
        entry = catalog["objects"][0]
        self.assertEqual(entry["object_id"], 0)
        self.assertEqual(entry["split"], "train")
        self.assertEqual(entry["symmetry"], "unknown")
        self.assertAlmostEqual(entry["diameter_m"], np.sqrt(0.12))
        self.assertFalse(Path(entry["normalized_obj"]).is_absolute())

        link = assets / "models_eval" / "obj_000000.ply"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), bop_ply.resolve())
        info = json.loads((assets / "models_eval/models_info.json").read_text())
        self.assertAlmostEqual(info["0"]["diameter"], np.sqrt(120000.0))
        self.assertEqual(info["0"]["size_x"], 200.0)

        second = models.prepare_models(
            assets_root=assets,
            index_path=index,
            split_path=split,
        )
        self.assertEqual(second["object_count"], 1)
        self.assertTrue(link.is_symlink())


if __name__ == "__main__":
    unittest.main()
