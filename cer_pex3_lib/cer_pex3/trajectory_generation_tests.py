import json
from pathlib import Path

import numpy as np

with (Path(__file__).parent / "traj_gen_tests_1d.json").open() as f:
    traj_gen_tests_1d = json.load(f)

with (Path(__file__).parent / "traj_gen_tests_2d.json").open() as f:
    traj_gen_tests_2d = json.load(f)

with (Path(__file__).parent / "traj_trans_tests.json").open() as f:
    traj_trans_tests = json.load(f)
