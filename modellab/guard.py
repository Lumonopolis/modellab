"""Admission and guard decisions. No I/O here, so any reading can be tested.

ModelLab jobs yield first. The guard stops a job only when memory is actually short
(macOS reports critical pressure, or less than the floor is available) or disk space is
(less than the disk floor free where swap lives). The warn level alone decides nothing.
Resumable jobs yield to the others: queued jobs pass them, a job that does not fit may
stop them, and they are stopped first when memory is short.
"""

from collections import deque
from dataclasses import dataclass

from .config import format_size

WRITING = 2**20  # bytes per second: a job writing less (logs, a save now and then) cannot fill a disk soon
DAY = 86400      # seconds: a slower writer still counts once it would fill what is left within a day


@dataclass(frozen=True)
class Job:
    run_id: str
    budget: int
    footprint: int
    resumable: bool = False
    written: int = 0          # bytes the job has written to disk
    write_rate: float = 0.0   # bytes per second now: over the last `recent` seconds (Writes)
    window_rate: float = 0.0  # bytes per second over the last `window` seconds (Writes)
    window: float = 0.0       # that window, in seconds


class Writes:
    """A job's writing as bytes per second, now (over the last `recent` seconds) and over the
    last `window` seconds.

    Each rate divides the bytes by its whole span, not by the time they took to write, so one
    checkpoint save counts as what it adds over the window: on 2026-10-02 a 192 MB save, which
    the 10-second rate read as 19 MB/s, is 0.1 MB/s over 30 minutes. Time before the first
    reading counts as writing nothing; `since` is the byte count at that point (a restarted job
    is judged on what it writes after the restart).
    """

    def __init__(self, window, recent, since=0):
        self.window, self.recent, self.since = window, recent, since
        self.readings = deque()  # (time, bytes written so far); the oldest at or before the longer span's start

    def rates(self, now, written):
        """(rate now, rate over the window) after this reading."""
        if not self.readings:
            self.readings.append((now, self.since))
        self.readings.append((now, written))
        while len(self.readings) > 1 and self.readings[1][0] <= now - max(self.window, self.recent):
            self.readings.popleft()
        return (written - self._at(now - self.recent)) / self.recent, (written - self._at(now - self.window)) / self.window

    def _at(self, start):
        """Bytes written by the last reading at or before `start`, else by the oldest kept."""
        for when, written in reversed(self.readings):
            if when <= start:
                return written
        return self.readings[0][1]


def queued_ahead(created, resumable, queued):
    """How many other queued jobs a job waits behind. queued: (created, resumable) of each.

    Jobs start oldest first, except that no job waits for a resumable one, and a resumable
    job waits for every job that is not resumable.
    """
    if resumable:
        return sum(1 for when, other in queued if not other or when < created)
    return sum(1 for when, other in queued if not other and when < created)


def admission(budget, running, allowance, available, floor):
    """None if a job may start now; otherwise why it waits.

    running: Jobs already running. Their budgets must fit the allowance together with this
    one, and the part of their budgets they have not used yet must still fit in available memory.
    """
    held = sum(job.budget for job in running)
    if held + budget > allowance:
        return f"running jobs hold {format_size(held)} of {format_size(allowance)}"
    headroom = sum(max(0, job.budget - job.footprint) for job in running)
    if budget + headroom > available - floor:
        return (f"{format_size(available)} available; running jobs may still grow by "
                f"{format_size(headroom)} and {format_size(floor)} must stay free")
    return None


def make_room(budget, running, allowance, available, floor):
    """Resumable jobs to stop, largest budget first, so that a job that does not fit can start.

    None when stopping every resumable job would still leave too little room: then nothing
    is stopped. The memory a stopped job frees is taken to be its current footprint.
    """
    stopping = []
    for job in sorted((job for job in running if job.resumable),
                      key=lambda job: (job.budget, job.footprint), reverse=True):
        stopping.append(job)
        rest = [other for other in running if other not in stopping]
        freed = sum(stopped.footprint for stopped in stopping)
        if admission(budget, rest, allowance, available + freed, floor) is None:
            return [stopped.run_id for stopped in stopping]
    return None


def over_budget(job, factor):
    return job.footprint > job.budget * factor


def short_of_memory(pressure, available, floor):
    """Memory is actually short: critical pressure, or less than the floor available."""
    return pressure == "critical" or available < floor


def pressure_victim(jobs):
    """The job to stop when memory is short: the one furthest over its budget, otherwise a
    resumable job, otherwise any job; the largest first."""
    if not jobs:
        return None
    return max(jobs, key=lambda job: (max(job.footprint / job.budget, 1.0), job.resumable, job.footprint)).run_id


def disk_victim(jobs, free):
    """The job to stop when disk space is short: the one filling it fastest. Stopping a job
    frees none of the space its files hold; it only stops the filling, so a job that is not
    filling the disk is never chosen, and None means no ModelLab job is.

    A job is filling the disk when it writes at least WRITING, or fast enough to fill the
    `free` bytes left within a day, both now and over the write window. The window spares
    one checkpoint save; "now" spares a job that has stopped writing, since stopping it
    would stop nothing."""
    bar = min(WRITING, free / DAY)
    rates = {job.run_id: min(job.write_rate, job.window_rate) for job in jobs}
    filling = [job for job in jobs if rates[job.run_id] > 0 and rates[job.run_id] >= bar]
    if not filling:
        return None
    return max(filling, key=lambda job: job.window_rate).run_id
