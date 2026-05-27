import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np


def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_sam_benchmark_from_annotations.py"
    spec = importlib.util.spec_from_file_location("run_sam_benchmark_from_annotations", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_object_ids_from_mask_ignores_background():
    runner = load_runner_module()
    mask = np.array([[0, 2], [1, 2]], dtype=np.uint8)

    assert runner.object_ids_from_mask(mask) == [1, 2]


def test_compose_index_mask_preserves_object_ids():
    runner = load_runner_module()
    mask1 = np.array([[True, False], [False, False]])
    mask2 = np.array([[False, False], [True, True]])

    output = runner.compose_index_mask([1, 2], [mask1, mask2], 2, 2)

    assert output.tolist() == [[1, 0], [2, 2]]


def test_zip_submission_uses_sample_submission_root(tmp_path):
    runner = load_runner_module()
    submission = tmp_path / "sample_submission"
    video_dir = submission / "abc"
    video_dir.mkdir(parents=True)
    (video_dir / "00000.png").write_bytes(b"fake")

    zip_path = tmp_path / "submission.zip"
    runner.zip_submission(submission, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["sample_submission/abc/00000.png"]


def test_overlay_index_mask_changes_foreground_only():
    runner = load_runner_module()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)

    overlay = runner.overlay_index_mask(image, mask)

    assert overlay[0, 0].tolist() == [0, 0, 0]
    assert overlay[0, 1].sum() > 0
    assert overlay[1, 0].sum() > 0
