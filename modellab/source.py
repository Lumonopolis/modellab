"""Which code a job ran: the Git commit of its working folder and any uncommitted changes.

A local run keeps its uncommitted code in <run>/code/: diff.patch (tracked files under the
working folder, against HEAD) and untracked/ (files there that Git does not ignore), listed
in manifest.json. Files the command names come first, then the files nearest to them. Files
Git ignores are never copied; an untracked file Git does not ignore can be, so keep data and
secrets ignored. A remote run needs none of this: its --sync inputs stay in its own work
folder on the host.
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

from .config import format_size

FILE_LIMIT = 2**20        # a larger untracked file is listed, not copied
TOTAL_LIMIT = 16 * 2**20  # what one run keeps at most, diff included; the rest is listed
LISTED = 200              # files named in each manifest list; the rest are only counted
# Plain unified diff whatever the user's Git configuration says about colour, drivers or prefixes.
DIFF = ["diff", "--no-color", "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/", "HEAD"]


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True)


def commit(cwd):
    """HEAD of the repository holding cwd, plus "+dirty" when anything under cwd is uncommitted.
    None outside a repository."""
    head = _git(cwd, "rev-parse", "HEAD")
    if head.returncode != 0:
        return None
    dirty = _git(cwd, "status", "--porcelain", "--", ".")
    return head.stdout.decode().strip() + ("+dirty" if dirty.stdout.strip() else "")


def snapshot(cwd, folder, command=()):
    """Copy the uncommitted code under cwd into folder. Returns the manifest, or None when
    cwd is outside a repository or nothing there is uncommitted."""
    head = _git(cwd, "rev-parse", "HEAD")
    if head.returncode != 0:
        return None
    diff = _checked(cwd, *DIFF, "--", ".")
    # A binary change shows in the diff only as "Binary files differ"; list those files.
    binary = [line.split("\t", 2)[2] for line in _checked(cwd, *DIFF, "--numstat", "--", ".").decode().splitlines()
              if line.startswith("-\t-\t")]
    listing = _checked(cwd, "ls-files", "--others", "--exclude-standard", "-z", "--", ".")
    names = [name for name in listing.decode().split("\0") if name]
    if not diff and not names:
        return None
    folder = Path(folder)
    folder.mkdir(parents=True)
    manifest = {"commit": head.stdout.decode().strip(),
                "cwd_in_repository": _checked(cwd, "rev-parse", "--show-prefix").decode().strip(),
                "diff": None, "untracked": [], "skipped": [], "untracked_not_listed": 0, "skipped_not_listed": 0}
    kept = 0
    if len(diff) > TOTAL_LIMIT:
        manifest["skipped"].append({"path": "diff.patch", "size": len(diff), "reason": "larger than the limit"})
    elif diff:
        (folder / "diff.patch").write_bytes(diff)
        manifest["diff"] = "diff.patch"
        kept = len(diff)
    manifest["skipped"] += [{"path": name, "size": None, "reason": "binary change, not kept"} for name in binary]
    for name in sorted(names, key=_nearest_first(cwd, command)):
        entry, kept = _keep(Path(cwd) / name, folder / "untracked" / name, kept)
        listed = "untracked" if "sha256" in entry else "skipped"
        if len(manifest[listed]) < LISTED:
            manifest[listed].append({"path": name, **entry})
        else:
            manifest[f"{listed}_not_listed"] += 1
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _keep(source, target, kept):
    """Copy one untracked file if the limits allow. Returns its manifest entry and the new total."""
    try:
        info = source.lstat()
        if not stat.S_ISREG(info.st_mode):
            return {"size": info.st_size, "reason": "not a regular file"}, kept
        if info.st_size > FILE_LIMIT:
            return {"size": info.st_size, "reason": f"larger than {format_size(FILE_LIMIT)}"}, kept
        if kept + info.st_size > TOTAL_LIMIT:
            return {"size": info.st_size, "reason": f"the run already keeps {format_size(kept)}"}, kept
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    except OSError as error:  # unreadable, or removed since Git listed it
        return {"size": None, "reason": f"{type(error).__name__}: {error.strerror}"}, kept
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"size": info.st_size, "sha256": digest}, kept + info.st_size


def _nearest_first(cwd, command):
    """Sort key for untracked files: files the command names, then files sharing the most
    folders with them, then smaller files."""
    base = os.path.realpath(cwd)
    named = {os.path.relpath(os.path.realpath(os.path.join(base, word)), base) for word in command}
    folders = [Path(name).parent.parts for name in named]

    def key(name):
        parts = Path(name).parent.parts
        shared = max((len(os.path.commonprefix([parts, folder])) for folder in folders), default=0)
        try:
            size = (Path(cwd) / name).lstat().st_size
        except OSError:  # gone since Git listed it; _keep records that
            size = 0
        return (name not in named, -shared, size)
    return key


def _checked(cwd, *args):
    result = _git(cwd, *args)
    if result.returncode != 0:
        raise SystemExit(f"Could not record the job's uncommitted code: git {' '.join(args)} in {cwd} failed: "
                         f"{result.stderr.decode().strip()}")
    return result.stdout
