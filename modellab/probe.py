"""What this host lets ModelLab do. `modellab host init` runs this on the host itself."""

import fcntl
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import macos

MIB = 2**20


def run(root, disk_floor):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report = {}

    def check(name, action):
        try:
            report[name] = {"ok": True, "detail": action()}
        except Exception as error:  # every failure is reported by name; none is hidden
            report[name] = {"ok": False, "detail": f"{type(error).__name__}: {error}"}

    check("memory readings", _memory)
    check("disk readings", lambda: _disk(disk_floor))
    written = root / "probe-write.bin"
    child = subprocess.Popen([sys.executable, "-c",
                              "import os, sys, time\n"
                              "b = bytearray(64 * 2**20)\n"
                              "with open(sys.argv[1], 'wb') as f:\n"
                              "    f.write(os.urandom(16 * 2**20)); f.flush(); os.fsync(f.fileno())\n"
                              "time.sleep(30)\n", str(written)],
                             env={**os.environ, "MODELLAB_RUN_ID": "probe"}, start_new_session=True)
    try:
        time.sleep(1)
        check("process footprint", lambda: _footprint(child.pid))
        check("bytes written", lambda: _written(child.pid))
        check("run tag", lambda: _tag(child.pid))
        check("signals", lambda: _stop(child))
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        written.unlink(missing_ok=True)
    check("file locks", lambda: _flock(root / "probe.lock"))
    check("localhost port", _bind)
    return report


def _memory():
    pressure, available = macos.memory_state()
    return f"{macos.total_memory() // 2**30} GiB; pressure {pressure}; {available // 2**30} GiB available"


def _disk(disk_floor):
    volume = macos.swap_volume()
    free = macos.disk_free(volume)
    detail = f"{free // 2**30} GiB free on {volume}, floor {disk_floor // 2**30} GiB"
    if free < disk_floor:
        raise RuntimeError(f"{detail}: below the floor, so no job will start")
    return detail


def _footprint(pid):
    info = macos.process(pid)
    if not info or info.footprint < 60 * MIB:
        raise RuntimeError(f"read {info} for a child holding 64 MiB")
    return f"{info.footprint // MIB} MiB read for a child holding 64 MiB"


def _written(pid):
    info = macos.process(pid)
    if not info or info.written < 15 * MIB:
        raise RuntimeError(f"read {info} for a child that wrote 16 MiB")
    return f"{info.written // MIB} MiB read for a child that wrote 16 MiB"


def _tag(pid):
    if macos.environment_has(pid, "MODELLAB_RUN_ID=probe") is not True:
        raise RuntimeError("the child's MODELLAB_RUN_ID could not be read")
    return "found on a child process"


def _stop(child):
    os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=5)
    return "stopped a child's process group"


def _flock(path):
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return "taken and released"


def _bind():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        return f"listened on 127.0.0.1:{server.getsockname()[1]}"
