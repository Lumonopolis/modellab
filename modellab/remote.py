"""Remote hosts. ModelLab runs on the host itself, under the host's root; this Mac pushes
code, packages, models and job inputs with rsync and invokes the host's own `modellab`
over SSH. A remote host needs no network access beyond this SSH link.

<root>/app/          ModelLab code, environment definitions, REVISION and models.lock
<root>/bin/uv        uv, pushed by `host init` and `env sync`
<root>/config.toml   the host's own settings, written from this Mac's config
<root>/work/<tag>/   one folder per remote run: its --sync inputs and default working folder
"""

import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import HOME, models, revision


def _prefix(host):
    return shlex.split(host.launcher) if host.launcher else []


def shell(host, argv, **kwargs):
    """Run a plain command on the host (through its launcher, if any), as one quoted command."""
    return subprocess.run(["ssh", host.ssh, shlex.join([*_prefix(host), *argv])], **kwargs)


def invoke(host, argv, **kwargs):
    """Run the host's own `modellab` with argv."""
    root = host.root
    command = ["/usr/bin/env", f"MODELLAB_CONFIG={root}/config.toml",
               f"MODELLAB_MODELS_LOCK={root}/app/instance/models.lock",
               host.python or "python3", f"{root}/app/bin/modellab", *argv]
    return shell(host, command, **kwargs)


def rsync(host, sources, destination, *, delete=False, pull=False, copy_links=False, link_dest=None):
    command = ["rsync", "-a", "--exclude", "__pycache__"]
    if copy_links:  # every link arrives as the file it points to
        command.append("--copy-links")
    if delete:
        command.append("--delete")
    if link_dest:  # unchanged files become hard links to this earlier copy
        command.append(f"--link-dest={link_dest}")
    if host.launcher:
        command += ["--rsync-path", f"{host.launcher} rsync"]
    if pull:
        command += [f"{host.ssh}:{source}" for source in sources] + [str(destination)]
    else:
        command += [str(source) for source in sources] + [f"{host.ssh}:{destination}"]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"rsync {'from' if pull else 'to'} {host.name} failed with exit code {error.returncode}")


def check(result):
    if result.returncode != 0:
        if isinstance(result.stderr, str) and result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        raise SystemExit(f"Remote command on the host failed with exit code {result.returncode}")
    return result


def host_config(host):
    """The host's own config: it is `local` there, with this Mac's settings for it."""
    lines = ["# Written by `modellab` on the controlling Mac; change that Mac's config instead.",
             "[hosts.local]"]
    fields = {"root": str(host.root), "env": host.env, "hf_home": str(host.hf_home), "python": host.python,
              "offline": host.offline}
    lines += [f"{key} = {json.dumps(value)}" for key, value in fields.items() if value is not None]
    lines += ["", "[memory]"] + [f"{key} = {json.dumps(value)}" for key, value in host.memory.items()]
    return "\n".join(lines) + "\n"


def push_code(host):
    """Code, settings and models.lock, in two rsyncs. --delete only cleans inside app/bin,
    app/modellab and app/envs; running jobs have already imported what they need."""
    root = host.root
    check(shell(host, ["/bin/mkdir", "-p", f"{root}/app/instance", f"{root}/bin", f"{root}/work"]))
    with tempfile.TemporaryDirectory() as folder:
        stage = Path(folder)
        (stage / "REVISION").write_text(revision())
        (stage / "config.toml").write_text(host_config(host))
        rsync(host, [HOME / "bin", HOME / "modellab", HOME / "envs", stage / "REVISION"], f"{root}/app/",
              delete=True)
        rsync(host, [stage / "config.toml"], f"{root}/config.toml")
    push_lock(host)


def push_lock(host):
    """The host reads the same models.lock as this Mac, or none if this Mac has none."""
    lock, target = models.lock_path(), f"{host.root}/app/instance/models.lock"
    if lock.exists():
        rsync(host, [lock], target)
    else:
        check(shell(host, ["/bin/rm", "-f", target]))


def push_uv(host, local):
    found = shutil.which("uv") or (local.root / "bin" / "uv" if (local.root / "bin" / "uv").exists() else None)
    if not found:
        raise SystemExit("Setting up a host needs uv on this Mac: https://docs.astral.sh/uv/")
    rsync(host, [Path(found).resolve()], f"{host.root}/bin/uv")


def init(host, local):
    push_code(host)
    push_uv(host, local)
    result = check(invoke(host, ["_probe"], capture_output=True, text=True))
    return json.loads(result.stdout.strip().splitlines()[-1])


def env_sync(host, local, name):
    cache = local.root / "cache" / "uv"
    if not cache.exists():
        raise SystemExit(f"No package cache at {cache}. Run `modellab env sync {name}` on this Mac first.")
    push_code(host)
    push_uv(host, local)
    check(shell(host, ["/bin/mkdir", "-p", f"{host.root}/cache/uv"]))
    rsync(host, [f"{cache}/"], f"{host.root}/cache/uv/")
    check(invoke(host, ["env", "sync", name]))


def models_pull(host, local, aliases):
    """Copy snapshots verified on this Mac into the host's store, then record them there.
    Only the pinned revision travels, as plain files: the Hub cache may link to blobs kept
    in a store shared across repositories."""
    push_code(host)
    copied = {}
    for alias in aliases:
        if models.path(local, alias) is None:
            models.pull(local, alias)
        snapshot = Path(models.path(local, alias))
        target = f"{host.hf_home}/hub/{snapshot.parent.parent.name}/snapshots/{snapshot.name}"
        check(shell(host, ["/bin/mkdir", "-p", target]))
        rsync(host, [f"{snapshot}/"], f"{target}/", copy_links=True, delete=True)
        check(invoke(host, ["_record-pulled", alias, target,
                            "copied by rsync from a snapshot verified on the controlling Mac"]))
        copied[alias] = target
    return copied


def prepare_work(host, sources):
    """A fresh folder for one run, holding copies of its --sync inputs. Files unchanged since
    the previous run's copy become hard links, so repeated runs copy only what changed."""
    work = f"{host.root}/work"
    tag = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    listing = check(shell(host, ["/bin/ls", "-1", work], capture_output=True, text=True)).stdout.split()
    previous = max((name for name in listing if name < tag), default=None)
    check(shell(host, ["/bin/mkdir", "-p", f"{work}/{tag}"]))
    if sources:
        rsync(host, [os.path.abspath(source) for source in sources], f"{work}/{tag}/",
              link_dest=f"{work}/{previous}" if previous else None)
    return f"{work}/{tag}"


def fetch(host, run_id, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rsync(host, [f"{host.root}/runs/{run_id}/"], destination, pull=True)
    return destination
