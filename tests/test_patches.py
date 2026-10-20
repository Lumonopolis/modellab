"""Reviewed patches: applied by env sync to the exact version they fix, recorded in every run."""

import json
import os
import subprocess
import venv
from pathlib import Path

import pytest

from modellab import env

PATCH = """A comment above the diff says why the patch exists and when to delete it.

--- a/demo_lib/core.py
+++ b/demo_lib/core.py
@@ -1,2 +1,3 @@
 def pick(gates):
     inds = gates.argmax()
+    inds = stop(inds)
"""


def fake_env(tmp_path, version="1.2.3"):
    """A project with one patch, and an environment with the distribution it is for."""
    project = tmp_path / "project"
    (project / "patches").mkdir(parents=True)
    (project / "patches" / "demo-lib-1.2.3-stop-gradient.patch").write_text(PATCH)
    site = tmp_path / "env" / "lib" / "python3.14" / "site-packages"
    (site / "demo_lib").mkdir(parents=True)
    (site / "demo_lib" / "core.py").write_text("def pick(gates):\n    inds = gates.argmax()\n")
    (site / f"demo_lib-{version}.dist-info").mkdir()
    return project, tmp_path / "env", site


def test_a_patch_applies_once_and_is_recorded(tmp_path):
    project, env_dir, site = fake_env(tmp_path)
    assert env.apply_patches(project, env_dir) == ["demo-lib-1.2.3-stop-gradient.patch: applied"]
    assert (site / "demo_lib" / "core.py").read_text().endswith("    inds = stop(inds)\n")
    [record] = json.loads((site / "modellab_patches.json").read_text())
    assert record["patch"] == "demo-lib-1.2.3-stop-gradient.patch" and record["distribution"] == "demo-lib"
    assert record["version"] == "1.2.3" and len(record["sha256"]) == 64
    # a sync that kept the installed files finds the patch in place
    assert env.apply_patches(project, env_dir) == ["demo-lib-1.2.3-stop-gradient.patch: already applied"]
    assert (site / "demo_lib" / "core.py").read_text().count("stop(inds)") == 1
    assert not list(site.rglob("*.orig")) and not list(site.rglob("*.rej"))


def test_a_patch_for_another_version_or_that_no_longer_fits_stops_the_sync(tmp_path):
    project, env_dir, _ = fake_env(tmp_path / "newer", version="1.2.4")
    with pytest.raises(SystemExit, match="is for demo-lib 1.2.3, but 1.2.4 is installed.*delete the patch"):
        env.apply_patches(project, env_dir)
    project, env_dir, site = fake_env(tmp_path / "changed")
    (site / "demo_lib" / "core.py").write_text("def pick(gates):\n    return gates.top(1)\n")
    with pytest.raises(SystemExit, match="does not apply"):
        env.apply_patches(project, env_dir)
    assert (site / "demo_lib" / "core.py").read_text() == "def pick(gates):\n    return gates.top(1)\n"
    (project / "patches" / "demo-lib-1.2.3-stop-gradient.patch").rename(project / "patches" / "stray.patch")
    with pytest.raises(SystemExit, match="name a patch"):
        env.apply_patches(project, env_dir)


def test_a_relative_project_path_works(tmp_path, monkeypatch):
    project, env_dir, site = fake_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert env.apply_patches(Path("project"), env_dir) == ["demo-lib-1.2.3-stop-gradient.patch: applied"]


def test_no_patches_leaves_an_empty_record(tmp_path):
    project, env_dir, site = fake_env(tmp_path)
    (project / "patches" / "demo-lib-1.2.3-stop-gradient.patch").unlink()
    assert env.apply_patches(project, env_dir) == []
    assert json.loads((site / "modellab_patches.json").read_text()) == []


def test_runs_record_the_patches_beside_the_library_version(tmp_path):
    env_dir = tmp_path / "env"
    venv.create(env_dir)
    env.install_hook(env_dir)
    [site] = env_dir.glob("lib/python*/site-packages")
    (site / "mlx_vlm").mkdir()
    (site / "mlx_vlm" / "__init__.py").write_text("__version__ = '0.7.3'\n")
    (site / "modellab_patches.json").write_text(json.dumps([
        {"patch": "mlx-vlm-0.7.3-moe-router-stop-gradient.patch", "distribution": "mlx-vlm", "version": "0.7.3",
         "sha256": "0" * 64},
        {"patch": "mlx-lm-0.31.3-other.patch", "distribution": "mlx-lm", "version": "0.31.3", "sha256": "1" * 64}]))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = subprocess.run([env_dir / "bin" / "python", "-c", "import mlx_vlm"], capture_output=True, text=True,
                            env={**os.environ, "MODELLAB_RUN_DIR": str(run_dir), "MODELLAB_BUDGET": str(2**30),
                                 "MODELLAB_CACHE_LIMIT": str(2**28)})
    assert result.returncode == 0, result.stderr
    [entry] = [json.loads(line) for line in (run_dir / "libraries.jsonl").read_text().splitlines()]
    assert entry["module"] == "mlx_vlm" and entry["version"] == "0.7.3"
    assert entry["patches"] == ["mlx-vlm-0.7.3-moe-router-stop-gradient.patch"]  # only its own distribution's
