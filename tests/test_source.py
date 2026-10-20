"""What a run records about the code it ran."""

import functools
import subprocess

from modellab import source


def committed_repo(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    git = functools.partial(subprocess.run, cwd=repo, check=True, capture_output=True)
    git(["git", "init", "-q"])
    (repo / "tracked.py").write_text("x = 1\n")
    git(["git", "add", "."])
    git(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "start"])
    return repo, git


def test_a_snapshot_keeps_the_files_the_command_names_first(tmp_path, monkeypatch):
    repo, _ = committed_repo(tmp_path)
    (repo / "aaa").mkdir()
    for index in range(5):
        (repo / "aaa" / f"state-{index}.json").write_text("s" * 400)
    (repo / "job").mkdir()
    (repo / "job" / "train.py").write_text("t" * 400)
    (repo / "job" / "helper.py").write_text("h" * 400)
    monkeypatch.setattr(source, "TOTAL_LIMIT", 1000)
    manifest = source.snapshot(repo, tmp_path / "code", ["python", "job/train.py"])
    assert [entry["path"] for entry in manifest["untracked"]] == ["job/train.py", "job/helper.py"]
    assert {entry["path"] for entry in manifest["skipped"]} == {f"aaa/state-{index}.json" for index in range(5)}


def test_a_snapshot_lists_an_unreadable_file_and_ignores_git_colour_settings(tmp_path):
    repo, git = committed_repo(tmp_path)
    git(["git", "config", "color.ui", "always"])
    (repo / "tracked.py").write_text("x = 2\n")
    locked = repo / "locked.py"
    locked.write_text("secret = 1\n")
    locked.chmod(0)
    try:
        manifest = source.snapshot(repo, tmp_path / "code")
    finally:
        locked.chmod(0o644)
    [skipped] = manifest["skipped"]
    assert skipped["path"] == "locked.py" and "PermissionError" in skipped["reason"]
    patch = (tmp_path / "code" / "diff.patch").read_text()
    assert "\x1b[" not in patch and patch.startswith("diff --git a/tracked.py b/tracked.py")


def test_a_snapshot_lists_binary_changes_it_does_not_keep(tmp_path):
    repo, git = committed_repo(tmp_path)
    (repo / "weights.bin").write_bytes(bytes(range(256)))
    git(["git", "add", "."])
    git(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "weights"])
    (repo / "weights.bin").write_bytes(bytes(reversed(range(256))))
    manifest = source.snapshot(repo, tmp_path / "code")
    assert {"path": "weights.bin", "size": None, "reason": "binary change, not kept"} in manifest["skipped"]
