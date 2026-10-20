"""The job hook and guard with real MLX. Needs an installed environment:

    MODELLAB_TEST_MLX_ENV=<root>/envs/mlx python3 -m pytest tests/test_mlx.py
"""

import os
from pathlib import Path

import pytest

from modellab import runs

MLX_ENV = os.environ.get("MODELLAB_TEST_MLX_ENV")
pytestmark = pytest.mark.skipif(not MLX_ENV, reason="set MODELLAB_TEST_MLX_ENV to an installed mlx environment")
G = 2**30


@pytest.fixture
def mlx_lab(lab):
    (lab.root / "envs").mkdir(parents=True)
    (lab.root / "envs" / "mlx").symlink_to(Path(MLX_ENV))
    return lab


def test_mlx_limits_are_held_to_the_budget_including_mlx_lm_own_wired_limit(mlx_lab):
    script = (
        "import mlx.core as mx, mlx.nn as nn\n"
        "from mlx_lm.generate import wired_limit\n"
        "mx.set_memory_limit(10**12); print(mx.set_memory_limit(0))\n"
        "mx.set_cache_limit(10**12); print(mx.set_cache_limit(0))\n"
        "with wired_limit(nn.Linear(8, 8)):\n"
        "    inside = mx.set_wired_limit(0)\n"
        "print(inside)\n")
    path = mlx_lab.run("python", "-c", script, budget="1G", env="mlx")
    state, status = mlx_lab.wait(path, timeout=60)
    assert (state, status["exit_code"]) == ("exited", 0), (path / "log").read_text()
    memory, cache, wired = (int(line) for line in (path / "log").read_text().split()[-3:])
    assert (memory, cache, wired) == (G, G // 4, G)
    loaded = {entry["module"]: entry for entry in runs.libraries(path)}
    assert {"mlx.core", "mlx_lm"} <= loaded.keys()
    assert all(entry["inside_env"] and entry["version"] for entry in loaded.values()), loaded


def test_a_job_growing_on_the_gpu_is_killed_near_its_budget(mlx_lab):
    script = (
        "import mlx.core as mx, time\n"
        "held = []\n"
        "for _ in range(16):\n"
        "    a = mx.ones((32 * 2**20,), dtype=mx.float32); mx.eval(a); held.append(a); time.sleep(0.1)\n")
    path = mlx_lab.run("python", "-c", script, budget="1G", env="mlx")
    state, status = mlx_lab.wait(path, timeout=60)
    assert state == "killed" and "passed budget" in status["guard_action"]
    assert G < status["peak"] < 1.5 * G
