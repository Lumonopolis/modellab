"""Model aliases pinned to exact Hub commits.

models.lock sits next to this installation's config file (~/.config/modellab/ by default),
so every copy of ModelLab on the Mac reads the same pins, and the code holds none.
MODELLAB_MODELS_LOCK points elsewhere. Each entry records the
repository, the full commit, the kind of model and its quantization. Files are verified
against the Hub's recorded hashes when pulled; each host records what it has verified in
<root>/pulled.json, so a partial download never counts as pulled.

A pulled folder belongs to the shared Hugging Face cache. Jobs read it; they must not
save into it.
"""

import fcntl
import json
import os
import re
import subprocess
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path

from . import HOME, config, runs

LOCK_ENV = "MODELLAB_MODELS_LOCK"
_ALIAS = re.compile(r"[a-z0-9][a-z0-9-]*")  # no '_', so every alias has its own variable name
_REPO = re.compile(r"[\w.-]+/[\w.-]+")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_FIELDS = ("repo", "revision", "kind", "quantization")


def lock_path():
    return Path(os.environ.get(LOCK_ENV, config.config_path().parent / "models.lock")).expanduser()


def load():
    path = lock_path()
    models = tomllib.loads(path.read_text()) if path.exists() else {}
    for alias, entry in models.items():
        _check(alias, entry)
    return models


def _check(alias, entry):
    if not _ALIAS.fullmatch(alias) or not _REPO.fullmatch(entry.get("repo", "")) \
            or not _COMMIT.fullmatch(entry.get("revision", "")):
        raise SystemExit(f"{lock_path()}: entry {alias!r} needs a lowercase alias, an org/name repo "
                         f"and a full 40-character commit; found {entry}")


@contextmanager
def _editing():
    """Hold a lock on the lock file's folder while reading, changing and writing it."""
    folder = lock_path().parent
    folder.mkdir(parents=True, exist_ok=True)
    handle = os.open(folder, os.O_RDONLY)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        os.close(handle)


def save(models):
    lines = ["# Written by `modellab models add`. Each alias is pinned to one Hub commit.", ""]
    for alias in sorted(models):
        _check(alias, models[alias])
        lines.append(f"[{alias}]")
        lines += [f"{field} = {json.dumps(models[alias][field], ensure_ascii=False)}" for field in _FIELDS]
        lines.append("")
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text("\n".join(lines))
    os.replace(temporary, path)


def get(alias):
    models = load()
    if alias not in models:
        known = ", ".join(models) or f"none yet; pin one with `modellab models add <org/name>`"
        raise SystemExit(f"No model {alias!r} in {lock_path()}. Known: {known}")
    return models[alias]


def default_alias(repo):
    return re.sub(r"[^a-z0-9]+", "-", repo.split("/")[-1].lower()).strip("-")


def kind(config):
    return "mlx-vlm" if "vision_config" in config else "mlx-lm"


def quantization(config):
    settings = config.get("quantization") or config.get("quantization_config")
    if not settings:
        return "none"
    if "bits" not in settings:
        return settings.get("quant_method", "unknown")
    text = f"{settings['bits']}-bit"
    if "group_size" in settings:
        text += f", group size {settings['group_size']}"
    return text + (f", {settings['mode']}" if settings.get("mode") else "")


def env_var(alias):
    return "MODELLAB_MODEL_" + alias.upper().replace("-", "_")


def hub(host, request):
    """Run a Hub operation in the host's environment. Progress and errors go to the terminal."""
    python = host.root / "envs" / host.env / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"Model commands need the {host.env!r} environment. Run: modellab env sync {host.env}")
    # The Hub token stays where `hf auth login` put it, even when the model store moves.
    token = os.environ.get("HF_TOKEN_PATH") or str(
        Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "token")
    env = {**os.environ, "PYTHONPATH": str(HOME), "HF_HOME": str(host.hf_home), "HF_TOKEN_PATH": token,
           "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "0"}
    result = subprocess.run([python, "-m", "modellab.hub", json.dumps(request)],
                            cwd=HOME, env=env, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Hub {request['op']} of {request['repo']} failed (see above)")
    return json.loads(result.stdout.strip().splitlines()[-1])


def add(host, spec, alias=None, replace=False):
    repo, _, revision = spec.partition("@")
    alias = alias or default_alias(repo)
    if not _ALIAS.fullmatch(alias):
        raise SystemExit(f"Alias {alias!r}: use lowercase letters, digits and '-', for example --alias "
                         f"{default_alias(alias) or 'my-model'}")
    resolved = hub(host, {"op": "resolve", "repo": repo, "revision": revision or None})
    entry = {"repo": resolved["repo"], "revision": resolved["revision"], "kind": kind(resolved["config"]),
             "quantization": quantization(resolved["config"])}
    with _editing():
        models = load()
        old = models.get(alias)
        if old and (old["repo"], old["revision"]) != (entry["repo"], entry["revision"]) and not replace:
            raise SystemExit(f"{alias} already pins {old['repo']}@{old['revision'][:10]}. "
                             f"Use --replace to point it at {entry['repo']}@{entry['revision'][:10]}")
        models[alias] = entry
        save(models)
    return alias, entry, old


def pull(host, alias):
    entry = get(alias)
    result = hub(host, {"op": "pull", "repo": entry["repo"], "revision": entry["revision"]})
    record(host, alias, result["path"], f"{result['checked']} files verified against the Hub")
    return result


def record(host, alias, folder, how):
    """Mark a model as pulled on this host once its files are known to be right."""
    with runs.locked(host.root / "pulled.lock"):
        pulled = _pulled(host)
        pulled[_key(get(alias))] = {"path": str(folder), "how": how, "verified": time.time()}
        runs.write_json(host.root / "pulled.json", pulled)


def path(host, alias):
    """The model's verified local snapshot on this host, or None if it has not been pulled."""
    pulled = _pulled(host).get(_key(get(alias)))
    return pulled["path"] if pulled and Path(pulled["path"]).is_dir() else None


def _key(entry):
    return f"{entry['repo']}@{entry['revision']}"


def _pulled(host):
    return runs.read_json(host.root / "pulled.json", {})
