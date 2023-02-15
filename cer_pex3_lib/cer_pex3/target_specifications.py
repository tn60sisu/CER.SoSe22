import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

TargetSpecifications = NamedTuple("TargetSpecifications", (
    ("init_q", np.ndarray), ("init_dq", np.ndarray), ("q_tar", np.ndarray), ("dq_tar", np.ndarray),
    ("q_error_bounds", np.ndarray), ("dq_error_bounds", np.ndarray)))

with (Path(__file__).parent / "target_specs.json").open() as f:
    target_specs_raw = json.load(f)

target_specs_pd = [TargetSpecifications(**{k: np.array(v) for k, v in s.items()}) for s in target_specs_raw["pd"]]
target_specs_pid = [TargetSpecifications(**{k: np.array(v) for k, v in s.items()}) for s in target_specs_raw["pid"]]
