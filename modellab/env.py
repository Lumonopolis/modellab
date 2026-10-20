"""Install a locked ModelLab environment on this host, with the job hook and its reviewed patches.

A patch (envs/<env>/patches/<distribution>-<version>-<what>.patch, a unified diff against
site-packages) fixes one installed release. It is applied only to that exact version, at every
sync; when the version changes, the sync stops so the patch is deleted (the release carries the
fix) or remade. Every run records the patches beside the library versions it loaded.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess

from . import HOME

PATCH_NAME = re.compile(r"(?P<distribution>[a-z0-9]+(?:-[a-z0-9]+)*?)-(?P<version>\d+(?:\.\d+)+)-(?P<what>[a-z0-9-]+)\.patch")

ENVS = HOME / "envs"


def sync(host, name):
    project = ENVS / name
    if not (project / "uv.lock").exists():
        raise SystemExit(f"No environment {name!r}: {project / 'uv.lock'} is missing")
    target = host.root / "envs" / name
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(target), "UV_CACHE_DIR": str(host.root / "cache" / "uv"),
           "UV_PYTHON_INSTALL_DIR": str(host.root / "python")}
    uv = host.root / "bin" / "uv"  # remote hosts get their own copy from `host init`
    command = [str(uv) if uv.exists() else "uv", "sync", "--frozen", "--project", str(project)]
    if host.python:
        env["UV_PYTHON"] = host.python
    if host.offline:  # packages come from the cache pushed by `env sync --host`
        command.append("--offline")
        env["UV_PYTHON_DOWNLOADS"] = "never"
    subprocess.run(command, env=env, check=True)
    install_hook(target)
    for line in apply_patches(project, target):
        print(line)
    return target


def apply_patches(project, env_dir):
    """Apply the environment's patches to the exact versions they are for, and record them in
    site-packages/modellab_patches.json for the job hook. One line per patch: what was done."""
    [site] = env_dir.glob("lib/python*/site-packages")
    done, applied = [], []
    for patch in sorted((project / "patches").glob("*.patch")):
        name = PATCH_NAME.fullmatch(patch.name)
        if not name:
            raise SystemExit(f"{patch}: name a patch <distribution>-<version>-<what>.patch")
        distribution, version = name["distribution"], name["version"]
        installed = _installed(site, distribution)
        if installed != version:
            raise SystemExit(f"{patch.name} is for {distribution} {version}, but {installed or 'none'} is installed. "
                             "If that release carries the fix, delete the patch; otherwise remake it for it.")
        # -f: never guess that a patch is reversed (as -t does) and never ask; -C only checks.
        apply = ["/usr/bin/patch", "-p1", "-f", "-d", str(site), "-i", str(patch.resolve())]  # -d moves first
        if _quiet([*apply, "-R", "-C"]):
            done.append(f"{patch.name}: already applied")
        elif _quiet([*apply, "-C"]):
            subprocess.run(apply, check=True, capture_output=True, stdin=subprocess.DEVNULL)
            done.append(f"{patch.name}: applied")
        else:
            raise SystemExit(f"{patch.name} does not apply to {distribution} {installed}; remake it against that release")
        applied.append({"patch": patch.name, "distribution": distribution, "version": version,
                        "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()})
    (site / "modellab_patches.json").write_text(json.dumps(applied, indent=1))
    return done


def _installed(site, distribution):
    """The installed version of a distribution, from its .dist-info folder name."""
    wanted = re.sub(r"[-_.]+", "_", distribution).lower()
    for info in site.glob("*.dist-info"):
        found, _, version = info.name.removesuffix(".dist-info").partition("-")
        if re.sub(r"[-_.]+", "_", found).lower() == wanted:
            return version
    return None


def _quiet(command):
    return subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL).returncode == 0


def install_hook(env_dir):
    """Copy the job hook into an environment and load it at interpreter start."""
    [site] = env_dir.glob("lib/python*/site-packages")
    shutil.copyfile(HOME / "modellab" / "hook.py", site / "modellab_hook.py")
    (site / "modellab_hook.pth").write_text("import modellab_hook; modellab_hook.install()\n")
