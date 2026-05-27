import importlib.util
import sys
from pathlib import Path

def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mosev2_from_annotations.py"
    spec = importlib.util.spec_from_file_location("run_mosev2_from_annotations", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_zip_submission_direct_has_video_dirs_at_root(tmp_path):
    runner = load_runner_module()
    submission_dir = tmp_path / "submission"
    video_dir = submission_dir / "abc"
    video_dir.mkdir(parents=True)
    (video_dir / "00000.png").write_bytes(b"fake")

    zip_path = tmp_path / "submission.zip"
    runner.zip_submission_direct(submission_dir, zip_path)

    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["abc/00000.png"]
