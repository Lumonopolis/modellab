"""Run folders under a host's root.

<root>/runs/<id>/
  run.json     what to run; written once
  status.json  state and outcome
  live.json    footprint and processes, refreshed while the job runs
  log          the job's output
  lock         held with flock by the supervisor and the job while either lives
  stop         present when someone asked the run to stop
  guard-stop   {"reason", "kind"} when another supervisor asks this job to give memory or disk
               back (kind "memory short", "disk short" or "make room")
  libraries.jsonl  mlx.core, mlx_lm and mlx_vlm as the job loaded them (written by the job hook)
  code/        uncommitted code under the job's working folder (see source.py)
  out/         the job's results ($MODELLAB_RUN_DIR/out)
"""

import fcntl
import json
import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path

from . import macos

ACTIVE = ("queued", "running")


def runs_dir(root):
    return Path(root) / "runs"


def new_run(root, spec):
    run_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    path = runs_dir(root) / run_id
    (path / "out").mkdir(parents=True)
    write_json(path / "run.json", {"id": run_id, "created": time.time(), **spec})
    write_json(path / "status.json", {"state": "queued", "reason": "starting"})
    return path


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        if default is None:
            raise
        return default


def write_json(path, data):
    """Replace the file atomically, so readers never see a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=1))
    os.replace(temporary, path)


def all_runs(root):
    """Run folders, oldest first. Anything else in runs/, or a folder still being created, is skipped."""
    folder = runs_dir(root)
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir()
                  if (path / "run.json").is_file() and (path / "status.json").is_file())


def libraries(path):
    """What the job hook recorded: one entry per library, version and file. Lines that do not
    parse (a job killed mid-write) stay in the file and add one entry whose inside_env is None."""
    try:
        lines = (Path(path) / "libraries.jsonl").read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []
    entries, unreadable = {}, 0
    for line in lines:
        try:
            entry = json.loads(line)
            key = (entry["module"], str(entry.get("version")), str(entry.get("file")))
        except (ValueError, TypeError, KeyError):
            unreadable += 1
            continue
        entries[key] = {"inside_env": False, **entry}
    if unreadable:
        entries["unreadable"] = {"module": None, "version": None, "file": None, "inside_env": None,
                                 "unreadable_lines": unreadable}
    return list(entries.values())


def find(root, run_id):
    path = runs_dir(root) / run_id
    if not (path / "run.json").exists():
        raise SystemExit(f"No run {run_id!r} under {runs_dir(root)}")
    return path


def is_live(path):
    """True while the run's supervisor or job holds its lock."""
    try:
        with open(Path(path) / "lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
            return False
    except BlockingIOError:
        return True


def state(path):
    """The recorded state, corrected by what is actually alive.

    'lost': an active run whose supervisor and job are both gone.
    'unguarded': a running job whose supervisor is gone; nothing watches or stops it.
    """
    status = read_json(Path(path) / "status.json")
    if status["state"] in ACTIVE and not is_live(path):
        return "lost", status
    if status["state"] == "running" and not supervisor_alive(status):
        return "unguarded", status
    return status["state"], status


def supervisor_alive(status):
    info = macos.process(status["supervisor_pid"])
    return info is not None and info.start == status["supervisor_start"]


@contextmanager
def locked(path):
    """Hold an exclusive flock on path."""
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
