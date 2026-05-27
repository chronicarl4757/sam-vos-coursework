import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def load_fix_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fix_mosev2_submission_zip.py"
    spec = importlib.util.spec_from_file_location("fix_mosev2_submission_zip", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_frame_name_strips_mask_prefix():
    module = load_fix_module()

    assert module.normalize_frame_name("mask_00127.png") == "00127.png"
    assert module.normalize_frame_name("31.png") == "00031.png"


def test_normalize_rgb_binary_mask_to_indexed_mask():
    module = load_fix_module()
    arr = np.zeros((4, 5, 3), dtype=np.uint8)
    arr[1:3, 2:4] = 255

    image = module.normalize_mask(Image.fromarray(arr, mode="RGB"))
    output = np.asarray(image)

    assert image.mode == "L"
    assert set(np.unique(output).tolist()) == {0, 1}

