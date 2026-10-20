"""Configuration. Everything specific to one machine lives in the config file, not in code."""

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_ENV = "MODELLAB_CONFIG"
DEFAULT_CONFIG = Path("~/.config/modellab/config.toml")

MEMORY_DEFAULTS = {
    "reserve": "max(8G, 15%)",  # never given to jobs
    "floor": "10%",             # minimum available memory to admit a job or keep jobs running
    "cache_limit": "4G",        # MLX buffer cache per job, at most a quarter of its budget
    "over_budget": 1.1,         # kill a job whose footprint passes budget × this
    "poll": 0.5,                # seconds between guard readings
    "settle": 10,               # seconds to wait after a stop before choosing again
    "requeues": 3,              # stops for short memory or disk after which a resumable job is killed
    "disk_floor": "100%",       # free disk to keep on the swap volume (% of RAM, so swap can hold it all)
    "write_window": 1800,       # seconds over which a job's writing is judged when disk is short
    "stop_grace": 10,           # seconds between SIGTERM and SIGKILL for `modellab stop`
}
HOST_DEFAULTS = {"root": "~/modellab", "env": "mlx"}
_UNITS = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def parse_size(text, total):
    """Bytes from '40G', '512M', '15%' (of total), 'max(8G, 15%)' or 'min(4G, 25%)'."""
    text = str(text).replace(" ", "")
    if match := re.fullmatch(r"(max|min)\(([^,()]+),([^,()]+)\)", text):
        pick = max if match[1] == "max" else min
        return pick(parse_size(match[2], total), parse_size(match[3], total))
    if match := re.fullmatch(r"(\d+(?:\.\d+)?)%", text):
        return int(total * float(match[1]) / 100)
    if match := re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)(?:i?B)?", text, re.IGNORECASE):
        return int(float(match[1]) * _UNITS[match[2].upper()])
    raise ValueError(f"Not a size: {text!r}. Use forms like 40G, 512M, 15% or max(8G, 15%).")


def format_size(n):
    for unit in ("T", "G", "M", "K"):
        if abs(n) >= _UNITS[unit]:
            return f"{n / _UNITS[unit]:.1f}{unit}"
    return f"{n}B"


@dataclass(frozen=True)
class Host:
    name: str
    root: Path
    env: str
    hf_home: Path
    ssh: str | None
    launcher: str | None
    python: str | None  # interpreter for ModelLab and its environments on this host
    offline: bool       # the host cannot reach PyPI or the Hub; packages and models are pushed
    memory: dict

    def size(self, key, total):
        return parse_size(self.memory[key], total)

    def allowance(self, total):
        """Memory the host may give to jobs in total."""
        return total - self.size("reserve", total)

    def to_json(self):
        """Settings a run keeps, so later config edits do not change a running job's rules."""
        return {"name": self.name, "root": str(self.root), "env": self.env, "hf_home": str(self.hf_home),
                "ssh": self.ssh, "launcher": self.launcher, "python": self.python, "offline": self.offline,
                "memory": self.memory}

    @classmethod
    def from_json(cls, data):
        return cls(**{**data, "root": Path(data["root"]), "hf_home": Path(data["hf_home"])})


def config_path():
    return Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)).expanduser()


def load(path=None):
    """Hosts by name. Without a config file, the only host is `local` with default settings."""
    path = Path(path) if path else config_path()
    data = tomllib.loads(path.read_text()) if path.exists() else {}
    unknown = set(data) - {"hosts", "memory"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    memory = {**MEMORY_DEFAULTS, **data.get("memory", {})}
    hosts = data.get("hosts") or {"local": {}}
    return {name: _host(name, spec, memory, path) for name, spec in hosts.items()}


def _host(name, spec, memory, path):
    known = {"root", "env", "hf_home", "ssh", "launcher", "python", "offline", "memory"}
    if unknown := set(spec) - known:
        raise ValueError(f"{path}: host {name!r} has unknown keys {sorted(unknown)}")
    if unknown := set(spec.get("memory", {})) - set(MEMORY_DEFAULTS):
        raise ValueError(f"{path}: host {name!r} has unknown memory keys {sorted(unknown)}")
    if "ssh" in spec:
        # A remote host's paths are its own: nothing is expanded or defaulted from this Mac.
        if not Path(spec.get("root", "")).is_absolute():
            raise ValueError(f"{path}: remote host {name!r} needs an absolute root (no ~)")
        root = Path(spec["root"])
        hf_home = Path(spec.get("hf_home", root / "hf"))
        if not hf_home.is_absolute():
            raise ValueError(f"{path}: remote host {name!r} needs an absolute hf_home")
    else:
        root = Path(spec.get("root", HOST_DEFAULTS["root"])).expanduser()
        hf_home = Path(spec.get("hf_home", "~/.cache/huggingface")).expanduser()
    return Host(
        name=name,
        root=root,
        env=spec.get("env", HOST_DEFAULTS["env"]),
        hf_home=hf_home,
        ssh=spec.get("ssh"),
        launcher=spec.get("launcher"),
        python=spec.get("python"),
        offline=spec.get("offline", False),
        memory=_checked({**memory, **spec.get("memory", {})}, name, path),
    )


def _checked(memory, name, path):
    for key in ("poll", "write_window"):
        value = memory[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ValueError(f"{path}: host {name!r}: {key} must be a positive number of seconds, not {value!r}")
    return memory
