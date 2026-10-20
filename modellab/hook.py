"""Job hook. `modellab env sync` installs this file as `modellab_hook` in each environment,
with a .pth line that calls install() at interpreter start. It acts only inside a ModelLab
job (MODELLAB_BUDGET is set) and imports nothing from ModelLab, so it works on any host.

1. When mlx.core loads, set its memory and cache limits from the job budget, and clamp
   later calls to set_memory_limit, set_cache_limit and set_wired_limit. MLX-LM, MLX-VLM
   and mlx-lm-lora raise the wired limit on their own; this keeps them within the budget.
2. When mlx.core, mlx_lm or mlx_vlm loads, append its version and file to the run's
   libraries.jsonl, marking whether it came from inside the environment. A build loaded
   from anywhere else shows in the record.
3. Watch the pipe from the supervisor. If the supervisor dies, stop the job's process group.
"""

import importlib.abc
import json
import os
import signal
import stat
import sys
import threading


def install():
    budget = os.environ.get("MODELLAB_BUDGET")
    # Homebrew Python's sitecustomize can process this .pth a second time; install once.
    if budget is None or any(isinstance(finder, _AfterImport) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _AfterImport(int(budget), int(os.environ["MODELLAB_CACHE_LIMIT"])))
    _watch_supervisor()


def _watch_supervisor():
    # Only the job's first process watches; its children must not treat this fd number as the pipe.
    fd = os.environ.pop("MODELLAB_SUPERVISOR_FD", None)
    if fd is None:
        return
    fd = int(fd)
    try:
        is_pipe = stat.S_ISFIFO(os.fstat(fd).st_mode)
    except OSError:
        is_pipe = False
    if not is_pipe:
        # A wrapper closed the pipe before this interpreter started. The supervisor still
        # guards memory; only the stop-on-supervisor-death safety net is missing.
        print(f"modellab: MODELLAB_SUPERVISOR_FD={fd} is not the supervisor pipe; "
              "this job will not stop if its supervisor dies", file=sys.stderr, flush=True)
        return
    os.set_inheritable(fd, False)
    threading.Thread(target=_stop_when_supervisor_exits, args=(fd,), daemon=True,
                     name="modellab-supervisor-watch").start()


def _stop_when_supervisor_exits(fd):
    while os.read(fd, 1):  # the supervisor never writes; end of file means it is gone
        pass
    print("modellab: supervisor exited; stopping the job", file=sys.stderr, flush=True)
    os.killpg(os.getpgrp(), signal.SIGKILL)


def clamp(mx, budget, cache_limit):
    """Apply the job's limits to an imported mlx.core and cap every later change."""
    ceilings = {"set_memory_limit": budget, "set_cache_limit": cache_limit, "set_wired_limit": budget}
    for owner in (mx, getattr(mx, "metal", None)):
        for name, ceiling in ceilings.items():
            if owner is not None and (original := getattr(owner, name, None)):
                setattr(owner, name, _capped(original, ceiling))
    mx.set_memory_limit(budget)
    mx.set_cache_limit(cache_limit)


def _capped(original, ceiling):
    def set_limit(limit, *args, **kwargs):
        return original(min(int(limit), ceiling), *args, **kwargs)
    return set_limit


RECORDED = ("mlx.core", "mlx_lm", "mlx_vlm")


def record(name, module):
    """Append where a library was loaded from to $MODELLAB_RUN_DIR/libraries.jsonl. A failure
    here is reported on stderr and never breaks the job's import."""
    try:
        _record(name, module)
    except Exception as error:
        print(f"modellab: could not record where {name} was loaded from: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)


def _patches(name):
    """The reviewed patches env sync applied to this module's distribution (modellab_patches.json
    beside this hook in site-packages)."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modellab_patches.json")) as handle:
            applied = json.load(handle)
    except FileNotFoundError:
        return []
    top = name.split(".")[0]
    return [entry["patch"] for entry in applied if entry["distribution"].replace("-", "_") == top]


def _record(name, module):
    file = getattr(module, "__file__", None)
    file = file and os.path.realpath(file)
    version = getattr(module, "__version__", None)
    entry = {"module": name, "version": None if version is None else str(version), "file": file,
             "inside_env": bool(file) and file.startswith(os.path.realpath(sys.prefix) + os.sep),
             "patches": _patches(name), "pid": os.getpid()}
    run_dir = os.environ.get("MODELLAB_RUN_DIR")
    if not run_dir:
        print(f"modellab: MODELLAB_RUN_DIR is not set; {name} {entry['version']} from {file} is not recorded",
              file=sys.stderr, flush=True)
        return
    path = os.path.join(run_dir, "libraries.jsonl")
    with open(path, "a") as handle:  # one short line per write, so processes do not interleave
        handle.write(json.dumps(entry) + "\n")


class _AfterImport(importlib.abc.MetaPathFinder):
    """Act right after a recorded library executes, before any other module can use it:
    clamp mlx.core's limits, and record every recorded library."""

    def __init__(self, budget, cache_limit):
        self.budget = budget
        self.cache_limit = cache_limit

    def find_spec(self, name, path, target=None):
        if name not in RECORDED:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            if spec := finder.find_spec(name, path, target):
                break
        else:
            return None
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            if name == "mlx.core":
                clamp(module, self.budget, self.cache_limit)
            record(name, module)

        spec.loader.exec_module = exec_module
        return spec
