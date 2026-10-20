"""Remote hosts: how commands and copies are built, and a real host end to end on request."""

import functools
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from conftest import CLI
import modellab
from modellab import cli, config, models, remote

G = 2**30
ROOT = "/r/modellab"


def host(**fields):
    base = {"name": "lab2", "root": ROOT, "env": "mlx", "hf_home": "/r/hf", "ssh": "lab2",
            "launcher": None, "python": "/r/python3", "offline": True, "memory": dict(config.MEMORY_DEFAULTS)}
    return config.Host.from_json({**base, **fields})


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(remote.subprocess, "run", run)
    return seen


@pytest.fixture
def two_hosts(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(f'[hosts.local]\nroot = "{tmp_path}/local"\n[hosts.lab2]\nssh = "lab2"\nroot = "{ROOT}"\n')
    monkeypatch.setenv(config.CONFIG_ENV, str(path))
    return config.load()


def test_remote_commands_arrive_as_one_quoted_command(calls):
    remote.invoke(host(), ["run", "--name", "two words", "--", "python", "-c", "print('hi; there')"])
    [(ssh, alias, command)] = calls
    assert (ssh, alias) == ("ssh", "lab2")
    assert shlex.split(command) == [
        "/usr/bin/env", f"MODELLAB_CONFIG={ROOT}/config.toml", f"MODELLAB_MODELS_LOCK={ROOT}/app/instance/models.lock",
        "/r/python3", f"{ROOT}/app/bin/modellab", "run", "--name", "two words", "--", "python", "-c",
        "print('hi; there')"]


def test_a_launcher_prefixes_commands_and_rsync(calls):
    lab2 = host(launcher="/r/run")
    remote.shell(lab2, ["/bin/ls"])
    remote.rsync(lab2, ["/a"], "/b", copy_links=True)
    remote.rsync(host(), [f"{ROOT}/runs/x/"], "/here", pull=True)
    assert shlex.split(calls[0][2]) == ["/r/run", "/bin/ls"]
    assert calls[1] == ["rsync", "-a", "--exclude", "__pycache__", "--copy-links", "--rsync-path", "/r/run rsync",
                        "/a", "lab2:/b"]
    assert calls[2][-2:] == [f"lab2:{ROOT}/runs/x/", "/here"]


def test_pushed_code_only_replaces_modellab_folders(calls, tmp_path, monkeypatch):
    monkeypatch.setenv(models.LOCK_ENV, str(tmp_path / "absent.lock"))
    remote.push_code(host())
    deleting = [command for command in calls if command[0] == "rsync" and "--delete" in command]
    assert [command[-1] for command in deleting] == [f"lab2:{ROOT}/app/"]
    assert [os.path.basename(source) for source in deleting[0][5:-1]] == ["bin", "modellab", "envs", "REVISION"]
    assert shlex.split(calls[-1][2]) == ["/bin/rm", "-f", f"{ROOT}/app/instance/models.lock"]  # no lock here


def test_a_model_travels_as_plain_files_into_its_revision_folder(calls, monkeypatch, tmp_path):
    snapshot = tmp_path / "hub" / "models--org--tiny" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(remote.models, "path", lambda local, alias: str(snapshot))
    monkeypatch.setattr(remote, "push_code", lambda host: None)
    copied = remote.models_pull(host(), host(ssh=None, name="local", root=str(tmp_path)), ["tiny"])
    target = f"/r/hf/hub/models--org--tiny/snapshots/{'a' * 40}"
    assert copied == {"tiny": target}
    [copy] = [command for command in calls if command[0] == "rsync"]
    assert "--copy-links" in copy and copy[-1] == f"lab2:{target}/"


def test_setting_up_a_host_without_uv_says_so(calls, monkeypatch, tmp_path):
    monkeypatch.setattr(remote.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="needs uv"):
        remote.push_uv(host(), host(ssh=None, name="local", root=str(tmp_path)))


def test_remote_hosts_never_take_paths_from_this_mac(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[hosts.lab2]\nssh = "lab2"\n')
    with pytest.raises(ValueError, match="absolute root"):
        config.load(path)
    path.write_text('[hosts.lab2]\nssh = "lab2"\nroot = "~/modellab"\n')
    with pytest.raises(ValueError, match="absolute root"):
        config.load(path)
    path.write_text(f'[hosts.lab2]\nssh = "lab2"\nroot = "{ROOT}"\n')
    assert str(config.load(path)["lab2"].hf_home) == f"{ROOT}/hf"


def test_the_host_gets_its_own_config_as_its_local_host(tmp_path):
    lab2 = host(memory={**config.MEMORY_DEFAULTS, "reserve": "max(16G, 25%)"})
    path = tmp_path / "config.toml"
    path.write_text(remote.host_config(lab2))
    [local] = config.load(path).values()
    assert (local.name, local.root, local.hf_home, local.python, local.offline, local.ssh) == \
        ("local", lab2.root, lab2.hf_home, "/r/python3", True, None)
    assert local.allowance(128 * G) == 96 * G
    assert tomllib.loads(path.read_text())["memory"]["reserve"] == "max(16G, 25%)"


class Given:
    def __init__(self, *words):
        self.argv_given = list(words)


def test_forwarding_drops_only_the_host_option():
    assert cli.forwarded(Given("logs", "--host", "lab2", "abc", "-f", "--", "--host", "kept")) == \
        ["logs", "abc", "-f", "--", "--host", "kept"]
    assert cli.forwarded(Given("ps", "--host=lab2", "--all")) == ["ps", "--all"]
    assert cli.forwarded(Given("serve", "lab2", "--host", "lab2", "--port", "9")) == ["serve", "lab2", "--port", "9"]


def test_a_remote_run_gets_its_own_work_folder_and_provenance(two_hosts, monkeypatch):
    sent = []
    monkeypatch.setattr(remote, "push_code", lambda host: None)
    monkeypatch.setattr(remote, "prepare_work", lambda host, sources: f"{ROOT}/work/T")
    monkeypatch.setattr(remote, "invoke", lambda host, argv: sent.append(argv) or subprocess.CompletedProcess(argv, 0))
    for cwd in ([], ["--cwd", "job"], ["--cwd", "/abs"]):
        cli.main(["run", "--host", "lab2", "--env", "none", "--model", "tiny", *cwd, "--", "python", "x.py"])
    cwds = [argv[argv.index("--cwd") + 1] for argv in sent]
    assert cwds == [f"{ROOT}/work/T", f"{ROOT}/work/T/job", "/abs"]
    first = sent[0]
    assert first[first.index("--commit") + 1]  # this Mac's commit, or "none" outside a repository
    assert first[first.index("--"):] == ["--", "python", "x.py"]
    assert ("--env", "none") == tuple(first[first.index("--env"):first.index("--env") + 2])
    assert first[first.index("--model") + 1] == "tiny"
    assert "--resumable" not in first
    cli.main(["run", "--host", "lab2", "--resumable", "--", "python", "train.py"])
    assert "--resumable" in sent[-1][:sent[-1].index("--")]


def test_fetch_refuses_anything_but_a_run_id(two_hosts):
    with pytest.raises(SystemExit, match="Not a run id"):
        cli.main(["fetch", "../..", "--host", "lab2"])


def test_fetch_keeps_runs_under_modellab_and_says_when_out_is_empty(two_hosts, tmp_path, monkeypatch, capsys):
    run_id = "20261001-074500-ab12"
    written = {"cwd": f"{ROOT}/work/20261001-074459-cd34"}

    def pull(host, sources, destination, **options):  # what rsync brings back from the host
        (Path(destination) / "out").mkdir(parents=True)
        (Path(destination) / "run.json").write_text(json.dumps({"id": run_id, **written}))
    monkeypatch.setattr(remote, "rsync", pull)
    cli.main(["fetch", run_id, "--host", "lab2"])
    out, err = capsys.readouterr()
    assert out.strip() == str(tmp_path / "local" / "fetched" / "lab2" / run_id)  # never into the current folder
    assert "out/ is empty" in err and written["cwd"] in err
    (Path(out.strip()) / "out" / "result.txt").write_text("done")
    monkeypatch.setattr(remote, "rsync", lambda *a, **k: None)
    cli.main(["fetch", run_id, "--host", "lab2"])
    assert capsys.readouterr().err == ""


def test_options_are_never_abbreviated(two_hosts):
    with pytest.raises(SystemExit):
        cli.main(["ps", "--hos", "lab2"])


REMOTE = os.environ.get("MODELLAB_TEST_REMOTE")


@pytest.mark.skipif(not REMOTE, reason="set MODELLAB_TEST_REMOTE to a configured remote host, e.g. lab2")
def test_a_remote_host_runs_guards_and_returns_a_job(tmp_path, monkeypatch):
    monkeypatch.setenv(config.CONFIG_ENV, str(config.DEFAULT_CONFIG.expanduser()))
    report = subprocess.run([sys.executable, CLI, "host", "init", REMOTE], capture_output=True, text=True)
    assert report.returncode == 0, report.stdout + report.stderr
    inputs = tmp_path / "job"
    inputs.mkdir()
    (inputs / "write.py").write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['MODELLAB_RUN_DIR'], 'out', 'result.txt').write_text('from the host')\n")
    started = subprocess.run([sys.executable, CLI, "run", "--host", REMOTE, "--budget", "256M", "--sync",
                              str(inputs), "--", "python", "job/write.py"], capture_output=True, text=True)
    assert started.returncode == 0, started.stdout + started.stderr
    run_id = started.stdout.split()[0]
    deadline = time.monotonic() + 60
    while "exited" not in (listing := subprocess.run([sys.executable, CLI, "ps", "--host", REMOTE],
                                                     capture_output=True, text=True).stdout.split(run_id)[-1]):
        assert time.monotonic() < deadline and ("running" in listing or "queued" in listing), listing
        time.sleep(1)
    fetched = tmp_path / "fetched"
    subprocess.run([sys.executable, CLI, "fetch", run_id, "--host", REMOTE, "--to", str(fetched)], check=True,
                   capture_output=True)
    assert (fetched / "out" / "result.txt").read_text() == "from the host"


def test_the_modellab_version_is_the_last_commit_that_changed_its_code(tmp_path, monkeypatch):
    home = tmp_path / "repo" / "ModelLab"
    (home / "modellab").mkdir(parents=True)
    git = functools.partial(subprocess.run, cwd=tmp_path / "repo", check=True, capture_output=True, text=True)
    commit = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm"]
    git(["git", "init", "-q"])
    (home / "modellab" / "code.py").write_text("x = 1\n")
    git(["git", "add", "."])
    git([*commit, "modellab"])
    ours = git(["git", "rev-parse", "HEAD"]).stdout.strip()
    (tmp_path / "repo" / "other.txt").write_text("another project\n")
    git(["git", "add", "."])
    git([*commit, "other project"])
    monkeypatch.setattr(modellab, "HOME", home)
    assert modellab.revision() == ours
    (home / "modellab" / "code.py").write_text("x = 2\n")
    assert modellab.revision() == ours + "+dirty"


def fake_ssh(tmp_path, monkeypatch, script):
    """An `ssh` on PATH that runs script instead of reaching a host."""
    ssh = tmp_path / "bin" / "ssh"
    ssh.parent.mkdir()
    ssh.write_text(f"#!/bin/sh\n{script}\n")
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{ssh.parent}{os.pathsep}{os.environ['PATH']}")


def wait_on_lab2(*args):
    return subprocess.run([sys.executable, CLI, "wait", "20261020-000000-0000", "--host", "lab2", *args],
                          capture_output=True, text=True, check=False, timeout=60)


def test_wait_on_a_host_says_its_runs_go_on_when_ssh_fails(two_hosts, tmp_path, monkeypatch):
    fake_ssh(tmp_path, monkeypatch, "echo 'Connection to lab2 closed by remote host.' >&2; exit 255")
    result = wait_on_lab2()
    assert result.returncode == 255
    assert "closed by remote host" in result.stderr and "runs there are not affected" in result.stderr


def test_wait_on_a_host_with_older_code_names_the_fix(two_hosts, tmp_path, monkeypatch):
    fake_ssh(tmp_path, monkeypatch, "echo \"modellab: error: argument command: invalid choice: 'wait'\" >&2; exit 2")
    result = wait_on_lab2()
    assert result.returncode == 2 and "`modellab host init lab2` updates it" in result.stderr


def test_wait_on_a_silent_host_ends_at_its_own_timeout(two_hosts, tmp_path, monkeypatch, capsys):
    fake_ssh(tmp_path, monkeypatch, "sleep 30")
    monkeypatch.setattr(cli, "REMOTE_MARGIN", 0.5)
    began = time.monotonic()
    assert cli.main(["wait", "20261020-000000-0000", "--host", "lab2", "--timeout", "0"]) == cli.TIMED_OUT
    assert time.monotonic() - began < 10
    assert "No answer from lab2 within 0.5 s" in capsys.readouterr().err


def test_wait_on_a_host_keeps_its_status_when_the_host_writes_bytes_that_are_not_text(two_hosts, tmp_path,
                                                                                      monkeypatch):
    fake_ssh(tmp_path, monkeypatch, "printf 'banner \\377\\n' >&2; exit 0")
    result = wait_on_lab2()
    assert result.returncode == 0 and "banner" in result.stderr and "Traceback" not in result.stderr
