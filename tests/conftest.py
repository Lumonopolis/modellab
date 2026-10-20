import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modellab import HOME, config, runs  # noqa: E402

CLI = HOME / "bin" / "modellab"


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A host root and config of its own, so tests never touch a real installation."""
    path = tmp_path / "config.toml"
    monkeypatch.setenv(config.CONFIG_ENV, str(path))
    return Lab(tmp_path, path)


class Lab:
    def __init__(self, tmp, config_path):
        self.root = tmp / "root"
        self.config_path = config_path
        self.configure()

    def configure(self, **memory):
        lines = [f'[hosts.local]\nroot = "{self.root}"\n[memory]\npoll = 0.2\n']
        lines += [f"{key} = {json.dumps(value)}\n" for key, value in memory.items()]
        self.config_path.write_text("".join(lines))

    def run(self, *command, budget="256M", env="none", name="test", extra=()):
        result = subprocess.run([sys.executable, CLI, "run", "--budget", budget, "--env", env,
                                 "--name", name, *extra, "--", *command],
                                capture_output=True, text=True, check=True)
        return self.root / "runs" / result.stdout.split()[0]

    def wait(self, path, states=("exited", "killed", "stopped", "error", "lost"), timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state, status = runs.state(path)
            if state in states:
                return state, status
            time.sleep(0.1)
        raise AssertionError(f"{path.name} still {runs.state(path)} after {timeout} s")
