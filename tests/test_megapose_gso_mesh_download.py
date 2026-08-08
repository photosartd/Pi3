import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "preprocess"
    / "download"
    / "megapose_gso_meshes.py"
)
SPEC = importlib.util.spec_from_file_location(
    "megapose_gso_mesh_download", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class MegaPoseGsoMeshDownloadTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_archive(self):
        archive_path = self.root / "meshes.zip"
        prefix = "google_scanned_objects"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(f"{prefix}/invalid_meshes.json", "[]")
            for name in ("object_a", "object_b"):
                archive.writestr(
                    f"{prefix}/models_normalized/{name}/meshes/model.obj",
                    f"mesh-{name}",
                )
                archive.writestr(
                    f"{prefix}/models_normalized/{name}/meshes/texture.png",
                    f"texture-{name}",
                )
                archive.writestr(
                    f"{prefix}/models_orig/{name}/meshes/model.obj",
                    f"original-{name}",
                )
                archive.writestr(
                    f"{prefix}/models_bop-renderer_scale=0.1/{name}/meshes/model.ply",
                    f"bop-{name}",
                )
                archive.writestr(
                    f"{prefix}/models_pointcloud/{name}/meshes/model.obj",
                    f"points-{name}",
                )
        return archive_path

    def test_mapping_and_index_select_only_requested_normalized_meshes(self):
        mapping_path = self.root / "gso_models.json"
        mapping_path.write_text(
            json.dumps(
                [
                    {"obj_id": 0, "gso_id": "object_a"},
                    {"obj_id": 1, "gso_id": "object_b"},
                ]
            ),
            encoding="utf-8",
        )
        mapping = download.load_object_mapping(mapping_path)
        self.assertEqual(mapping, {0: "object_a", 1: "object_b"})

        index_path = self.root / "index.sqlite"
        with sqlite3.connect(index_path) as connection:
            connection.execute("CREATE TABLE objects (object_id INTEGER PRIMARY KEY)")
            connection.execute("INSERT INTO objects VALUES (1)")
        self.assertEqual(download.load_index_object_ids(index_path), {1})

        archive_path = self._write_archive()
        output = self.root / "output"
        with zipfile.ZipFile(archive_path) as archive:
            members = download.select_archive_members(
                archive,
                extract_mode="normalized",
                selected_gso_ids={mapping[1]},
                include_bop_meshes=True,
                include_pointclouds=True,
            )
            extracted, skipped = download.extract_members(archive, members, output)

        self.assertGreater(extracted, 0)
        self.assertEqual(skipped, 0)
        expected = (
            output
            / "google_scanned_objects/models_normalized/object_b/meshes/model.obj"
        )
        self.assertEqual(expected.read_text(encoding="utf-8"), "mesh-object_b")
        self.assertFalse(
            (
                output
                / "google_scanned_objects/models_normalized/object_a/meshes/model.obj"
            ).exists()
        )
        self.assertFalse((output / "google_scanned_objects/models_orig").exists())
        self.assertEqual(
            (
                output
                / "google_scanned_objects/models_bop-renderer_scale=0.1/"
                "object_b/meshes/model.ply"
            ).read_text(encoding="utf-8"),
            "bop-object_b",
        )
        self.assertEqual(
            (
                output
                / "google_scanned_objects/models_pointcloud/"
                "object_b/meshes/model.obj"
            ).read_text(encoding="utf-8"),
            "points-object_b",
        )
        self.assertTrue(
            (output / "google_scanned_objects/invalid_meshes.json").is_file()
        )

        with zipfile.ZipFile(archive_path) as archive:
            extracted, skipped = download.extract_members(archive, members, output)
        self.assertEqual(extracted, 0)
        self.assertEqual(skipped, len(members))

    def test_unsafe_archive_member_is_rejected(self):
        archive_path = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("../outside", "bad")
        with zipfile.ZipFile(archive_path) as archive:
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                download.select_archive_members(
                    archive,
                    extract_mode="all",
                    selected_gso_ids=None,
                )

    def test_cli_requires_license_and_rejects_ambiguous_filter(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                download.parse_args(["--output-root", str(self.root)])
            with self.assertRaises(SystemExit):
                download.parse_args(
                    [
                        "--output-root",
                        str(self.root),
                        "--index-path",
                        str(self.root / "index.sqlite"),
                        "--object-id",
                        "1",
                        "--accept-license",
                    ]
                )

    def test_main_downloads_local_sources_and_writes_manifest(self):
        source_dir = self.root / "source"
        source_dir.mkdir()
        archive_path = self._write_archive()
        mapping_path = source_dir / "gso_models.json"
        mapping_path.write_text(
            json.dumps(
                [
                    {"obj_id": 0, "gso_id": "object_a"},
                    {"obj_id": 1, "gso_id": "object_b"},
                ]
            ),
            encoding="utf-8",
        )
        output = self.root / "assets"

        with contextlib.redirect_stdout(io.StringIO()):
            result = download.main(
                [
                    "--output-root",
                    str(output),
                    "--archive-url",
                    archive_path.as_uri(),
                    "--mapping-url",
                    mapping_path.as_uri(),
                    "--expected-archive-bytes",
                    str(archive_path.stat().st_size),
                    "--object-id",
                    "1",
                    "--include-bop-meshes",
                    "--include-pointclouds",
                    "--accept-license",
                ]
            )

        self.assertEqual(result, 0)
        self.assertTrue(
            (
                output
                / "google_scanned_objects/models_normalized/object_b/meshes/model.obj"
            ).is_file()
        )
        self.assertFalse(
            (
                output
                / "google_scanned_objects/models_normalized/object_a/meshes/model.obj"
            ).exists()
        )
        manifest = json.loads(
            (output / "megapose_gso_meshes.download.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["format"], "pi3_megapose_gso_mesh_download_v1")
        self.assertEqual(manifest["selected_numeric_object_ids"], [1])
        self.assertEqual(manifest["normalized_mesh_scale_for_megapose"], 0.1)
        self.assertTrue(manifest["include_bop_meshes"])
        self.assertTrue(manifest["include_pointclouds"])


if __name__ == "__main__":
    unittest.main()
