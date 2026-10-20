"""Process, memory and disk readings on macOS through libproc, sysctl and statvfs. None of
these need root."""

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import NamedTuple

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)

_RUSAGE_INFO_V2 = 2
_PROC_PGRP_ONLY = 2
_PROC_UID_ONLY = 4
_CTL_KERN = 1
_KERN_ARGMAX = 8
_KERN_PROCARGS2 = 49
# kern.memorystatus_vm_pressure_level reports dispatch levels; xnu maps "urgent" to warn.
_PRESSURE = {1: "normal", 2: "warn", 4: "critical"}


class _RusageInfoV2(ctypes.Structure):
    """struct rusage_info_v2 from <sys/resource.h>."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
        "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
        "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
        "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten")]


class Process(NamedTuple):
    footprint: int  # physical footprint in bytes, Metal buffers (MLX GPU memory) included
    start: int      # start time, which tells a process apart from a later one reusing its PID
    written: int    # bytes it has written to disk since it started


def process(pid):
    """A process's readings, or None once it is gone."""
    info = _RusageInfoV2()
    if _libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        return None
    return Process(info.phys_footprint, info.proc_start_abstime, info.diskio_byteswritten)


def disk_free(path):
    """Bytes available on the volume holding path, or its nearest existing parent. On APFS,
    volumes in one container share this space. Purgeable space, such as local snapshots,
    is not counted, so the reading errs low."""
    path = Path(path).absolute()
    while True:
        try:
            info = os.statvfs(path)
            return info.f_bavail * info.f_frsize
        except FileNotFoundError:
            if path == path.parent:
                raise
            path = path.parent


def swap_volume():
    """Where macOS keeps swap: a full disk here can hang the machine when memory runs short."""
    return "/System/Volumes/VM" if os.path.isdir("/System/Volumes/VM") else "/"


def children(pid):
    buffer = (ctypes.c_int * 4096)()
    count = _libproc.proc_listchildpids(pid, buffer, ctypes.sizeof(buffer))
    return list(buffer[:count]) if count > 0 else []


def group(pgid):
    return _listpids(_PROC_PGRP_ONLY, pgid)


def user_processes():
    """PIDs of every process of this user."""
    return _listpids(_PROC_UID_ONLY, os.getuid())


def _listpids(kind, argument):
    buffer = (ctypes.c_int * 16384)()
    size = _libproc.proc_listpids(kind, argument, buffer, ctypes.sizeof(buffer))
    return [pid for pid in buffer[: size // ctypes.sizeof(ctypes.c_int)] if pid > 0]


_procargs = None


def environment_has(pid, entry):
    """True if the process's environment holds entry ('NAME=value'), False if not, None if unreadable.

    macOS hides the environment of Apple platform binaries such as /bin/sleep; their
    readable data holds only the path and arguments, so they read as False.
    """
    global _procargs
    if _procargs is None:
        _procargs = ctypes.create_string_buffer(_sysctl_mib((_CTL_KERN, _KERN_ARGMAX), ctypes.c_int))
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(len(_procargs))
    if _libc.sysctl(mib, 3, _procargs, ctypes.byref(size), None, 0) != 0:
        return None
    return entry.encode() in _procargs.raw[4:size.value].split(b"\0")


def _sysctl_mib(mib, ctype):
    value = ctype()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    names = (ctypes.c_int * len(mib))(*mib)
    if _libc.sysctl(names, len(mib), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"sysctl {mib}: {os.strerror(error)}")
    return value.value


def _sysctl(name, ctype):
    value = ctype()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if _libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"sysctl {name}: {os.strerror(error)}")
    return value.value


def total_memory():
    return _sysctl("hw.memsize", ctypes.c_uint64)


def memory_state():
    """(pressure level, bytes available now). Available memory is xnu's own estimate, not free pages."""
    level = _sysctl("kern.memorystatus_vm_pressure_level", ctypes.c_int)
    if level not in _PRESSURE:
        raise OSError(f"Unexpected memory pressure level {level}")
    percent = _sysctl("kern.memorystatus_level", ctypes.c_int)
    return _PRESSURE[level], total_memory() * percent // 100
