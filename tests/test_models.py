"""Model aliases pinned to Hub commits, and passing pulled models to jobs."""

import os
import subprocess
import sys

import pytest

from conftest import CLI
from modellab import config, models, runs

ENTRY = {"repo": "org/tiny-model", "revision": "a" * 40, "kind": "mlx-lm", "quantization": "4-bit, group size 64"}
CONFIG = {"model_type": "llama", "quantization": {"bits": 4, "group_size": 64}}


@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path / "instance" / "models.lock"
    monkeypatch.setenv(models.LOCK_ENV, str(path))
    return path


def test_the_pins_sit_next_to_the_config_so_every_copy_of_modellab_reads_them(tmp_path, monkeypatch):
    monkeypatch.delenv(models.LOCK_ENV, raising=False)
    monkeypatch.setenv(config.CONFIG_ENV, str(tmp_path / "settings" / "config.toml"))
    assert models.lock_path() == tmp_path / "settings" / "models.lock"
    monkeypatch.setenv(models.LOCK_ENV, str(tmp_path / "elsewhere.lock"))
    assert models.lock_path() == tmp_path / "elsewhere.lock"


def test_hub_operations_run_in_the_environment_with_modellab_importable(tmp_path, monkeypatch):
    import modellab
    monkeypatch.setenv(config.CONFIG_ENV, str(tmp_path / "config.toml"))
    (tmp_path / "config.toml").write_text(f'[hosts.local]\nroot = "{tmp_path}/root"\n')
    host = config.load()["local"]
    python = host.root / "envs" / host.env / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    seen = {}

    def run(command, cwd, env, stdout, text):
        seen.update(command=command, cwd=cwd, env=env)
        return subprocess.CompletedProcess(command, 0, stdout='{"repo": "org/x"}\n')
    monkeypatch.setattr(models.subprocess, "run", run)
    assert models.hub(host, {"op": "resolve", "repo": "org/x"}) == {"repo": "org/x"}
    assert seen["command"][:3] == [python, "-m", "modellab.hub"]
    assert seen["cwd"] == modellab.HOME and seen["env"]["PYTHONPATH"] == str(modellab.HOME)


@pytest.fixture
def fake_hub(monkeypatch):
    """Stand-in for the Hub: resolves to a canonical name and commit, and records calls."""
    calls = []

    def hub(host, request):
        calls.append(request)
        if request["op"] == "resolve":
            return {"repo": "Org/Tiny-Model", "revision": (request["revision"] or "c" * 40), "config": CONFIG}
        raise SystemExit("Hub pull of Org/Tiny-Model failed (see above)")

    monkeypatch.setattr(models, "hub", hub)
    return calls


def pulled(host, snapshot, entry=ENTRY):
    runs.write_json(host.root / "pulled.json", {f"{entry['repo']}@{entry['revision']}": {"path": str(snapshot)}})


def test_the_lock_round_trips_and_rejects_unpinned_entries(lock):
    models.save({"tiny": ENTRY, "other-model": {**ENTRY, "kind": "mlx-vlm"}})
    assert models.load() == {"tiny": ENTRY, "other-model": {**ENTRY, "kind": "mlx-vlm"}}
    assert lock.read_text().startswith("# Written by")
    lock.write_text(lock.read_text().replace("a" * 40, "main", 1))
    with pytest.raises(SystemExit, match="40-character commit"):
        models.load()


def test_names_kinds_and_quantization():
    assert models.default_alias("mlx-community/Qwen2.5-0.5B-Instruct-4bit") == "qwen2-5-0-5b-instruct-4bit"
    assert models.env_var("qwen3-vl-8b-4bit") == "MODELLAB_MODEL_QWEN3_VL_8B_4BIT"
    assert models.kind({"vision_config": {}}) == "mlx-vlm"
    assert models.kind({"model_type": "llama"}) == "mlx-lm"
    assert models.quantization({"quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}) \
        == "4-bit, group size 64, affine"
    assert models.quantization({"quantization_config": {"quant_method": "fp8"}}) == "fp8"
    assert models.quantization({}) == "none"


def test_a_bad_alias_is_refused_before_any_hub_call(lock, lab, fake_hub):
    with pytest.raises(SystemExit, match="Alias"):
        models.add(config.load()["local"], "org/tiny", alias="Bad_Alias")
    assert fake_hub == []


def test_add_stores_the_canonical_name_and_refuses_to_repoint_silently(lock, lab, fake_hub):
    host = config.load()["local"]
    alias, entry, old = models.add(host, "org/tiny-model")
    assert (alias, entry["repo"], entry["revision"], old) == ("tiny-model", "Org/Tiny-Model", "c" * 40, None)
    assert entry["quantization"] == "4-bit, group size 64"
    with pytest.raises(SystemExit, match="--replace"):
        models.add(host, f"org/tiny-model@{'d' * 40}")
    assert models.get("tiny-model")["revision"] == "c" * 40
    _, entry, old = models.add(host, f"org/tiny-model@{'d' * 40}", replace=True)
    assert (entry["revision"], old["revision"]) == ("d" * 40, "c" * 40)


def test_a_failed_pull_records_nothing(lock, lab, fake_hub):
    models.save({"tiny": ENTRY})
    host = config.load()["local"]
    with pytest.raises(SystemExit, match="failed"):
        models.pull(host, "tiny")
    assert not (host.root / "pulled.json").exists()
    assert models.path(host, "tiny") is None


def test_a_snapshot_folder_alone_does_not_count_as_pulled(lock, lab, tmp_path):
    models.save({"tiny": ENTRY})
    host = config.load()["local"]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    assert models.path(host, "tiny") is None
    pulled(host, snapshot)
    assert models.path(host, "tiny") == str(snapshot)
    snapshot.rmdir()
    assert models.path(host, "tiny") is None


def test_run_passes_pulled_models_and_refuses_the_rest(lock, lab, tmp_path):
    models.save({"tiny": ENTRY, "absent": {**ENTRY, "revision": "b" * 40}})
    host = config.load()["local"]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pulled(host, snapshot)
    refused = subprocess.run([sys.executable, CLI, "run", "--budget", "256M", "--env", "none",
                              "--model", "absent", "--", "true"], capture_output=True, text=True)
    assert refused.returncode != 0 and "not pulled" in refused.stderr
    show = "import os; print(os.environ['MODELLAB_MODEL'], os.environ['MODELLAB_MODEL_TINY'])"
    path = lab.run(sys.executable, "-c", show, extra=("--model", "tiny"))
    assert lab.wait(path)[0] == "exited"
    assert (path / "log").read_text().split() == [str(snapshot), str(snapshot)]
    assert runs.read_json(path / "run.json")["models"]["tiny"]["revision"] == ENTRY["revision"]


def test_jobs_see_only_the_models_chosen_for_their_run(lock, lab, tmp_path, monkeypatch):
    second = {**ENTRY, "repo": "org/second"}
    models.save({"tiny": ENTRY, "second": second})
    host = config.load()["local"]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    runs.write_json(host.root / "pulled.json", {f"{entry['repo']}@{entry['revision']}": {"path": str(snapshot)}
                                                for entry in (ENTRY, second)})
    monkeypatch.setenv("MODELLAB_MODEL", "/stale/model")
    monkeypatch.setenv("MODELLAB_MODEL_OLD", "/stale/old")
    show = "import os; print(sorted(k for k in os.environ if k.startswith('MODELLAB_MODEL') and k != 'MODELLAB_MODELS_LOCK'))"
    path = lab.run(sys.executable, "-c", show, extra=("--model", "tiny", "--model", "second"))
    assert lab.wait(path)[0] == "exited"
    assert (path / "log").read_text().strip() == "['MODELLAB_MODEL_SECOND', 'MODELLAB_MODEL_TINY']"


@pytest.mark.skipif(not (os.environ.get("MODELLAB_TEST_MLX_ENV") and os.environ.get("MODELLAB_TEST_HUB")),
                    reason="set MODELLAB_TEST_MLX_ENV and MODELLAB_TEST_HUB=1 to pull from the Hub")
def test_add_and_pull_a_real_model_from_the_hub(lock, lab):
    (lab.root / "envs").mkdir(parents=True)
    (lab.root / "envs" / "mlx").symlink_to(os.environ["MODELLAB_TEST_MLX_ENV"])
    host = config.load()["local"]
    alias, entry, _ = models.add(
        host, "mlx-community/smollm2-135m-instruct-8bit@0f0d9b8218915bc34d401e1a340b8c049d300d5e", "smol")
    assert entry["repo"] == "mlx-community/SmolLM2-135M-Instruct-8bit"  # canonical casing, one cache folder
    assert (entry["kind"], entry["quantization"]) == ("mlx-lm", "8-bit, group size 64")
    assert models.pull(host, alias)["checked"] >= 3
    assert os.path.isdir(models.path(host, alias))


@pytest.mark.skipif(not (os.environ.get("MODELLAB_TEST_MLX_ENV") and os.environ.get("MODELLAB_TEST_HUB")),
                    reason="set MODELLAB_TEST_MLX_ENV and MODELLAB_TEST_HUB=1 to pull from the Hub")
def test_serve_answers_on_localhost_with_logprobs_and_stops(lock, lab):
    import json
    import urllib.request
    (lab.root / "envs").mkdir(parents=True)
    (lab.root / "envs" / "mlx").symlink_to(os.environ["MODELLAB_TEST_MLX_ENV"])
    host = config.load()["local"]
    models.add(host, "mlx-community/SmolLM2-135M-Instruct-8bit@0f0d9b8218915bc34d401e1a340b8c049d300d5e", "smol")
    models.pull(host, "smol")
    result = subprocess.run([sys.executable, CLI, "serve", "smol", "--budget", "2G"], capture_output=True, text=True)
    assert "ready: http://127.0.0.1:" in result.stdout, result.stdout + result.stderr
    path = lab.root / "runs" / result.stdout.split()[0]
    url = runs.read_json(path / "run.json")["serve"]["url"]
    request = urllib.request.Request(f"{url}/chat/completions", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"messages": [{"role": "user", "content": "Say hi"}],
                                                      "max_tokens": 3, "logprobs": True, "top_logprobs": 3}).encode())
    with urllib.request.urlopen(request, timeout=30) as response:
        choice = json.load(response)["choices"][0]
    assert choice["message"]["content"] and choice["logprobs"]["content"]
    assert subprocess.run([sys.executable, CLI, "stop", path.name], capture_output=True).returncode == 0
    assert lab.wait(path)[0] == "stopped"
