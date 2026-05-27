import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mosev2_from_annotations.py"
    spec = importlib.util.spec_from_file_location("run_mosev2_from_annotations", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_object_prompts_from_index_mask():
    runner = load_runner_module()
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    mask[6:8, 1:3] = 2

    prompts = runner.object_prompts_from_mask(mask, margin=1)

    assert [(p.object_id, p.box) for p in prompts] == [
        (1, (2, 1, 7, 5)),
        (2, (0, 5, 3, 7)),
    ]
