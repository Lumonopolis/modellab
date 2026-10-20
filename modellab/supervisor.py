"""Run one job: wait for memory, start it, guard it, record the outcome.

`modellab run` starts this detached, without a shell, and hands it the run's lock already
held. The supervisor passes the lock on to the job, so the run counts as live while either
process exists. The job also gets the read end of a pipe; the job hook stops the job if the
supervisor dies. Every job process carries MODELLAB_RUN_ID, so children that leave the
process group or outlive their parent are still found.

A resumable job (`run --resumable`) that is stopped to give memory or disk back goes back in the
queue and starts again with the same command; it resumes from its own checkpoints.

The guard also keeps `disk_floor` free on the volume that holds swap: a full disk while
memory is short can hang the machine.
"""

import fcntl
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from . import HOME, config, guard, macos, models, runs, source
from .config import format_size

_PRESSURE_WAIT = 30  # seconds a pressure owner waits for its chosen job to go
_STALE_POLLS = 5     # a live.json older than this many polls belongs to a dead supervisor
_MAX_BACKOFF = 600   # seconds; the longest a requeued job waits after its stops for short memory or disk
_SHORT = ("memory short", "disk short")  # stops that count toward `requeues` and back off


class Stopped(Exception):
    """The supervisor itself received SIGTERM."""


class Supervisor:
    def __init__(self, path):
        self.path = Path(path)
        self.spec = runs.read_json(self.path / "run.json")
        self.host = config.Host.from_json(self.spec["host"])
        self.root = self.host.root
        self.memory = self.host.memory
        total = macos.total_memory()
        self.budget = self.spec["budget"]
        self.allowance = self.host.allowance(total)
        self.floor = self.host.size("floor", total)
        self.disk_floor = self.host.size("disk_floor", total)
        self.volume = macos.swap_volume()
        self.cache_limit = min(self.host.size("cache_limit", total), self.budget // 4)
        self.tag = f"MODELLAB_RUN_ID={self.spec['id']}"
        self.status = runs.read_json(self.path / "status.json")
        self.proc = None
        self.tracked = {}  # pid -> start time for every live process of the job
        self.checked = {}  # (pid, start time) -> whether it carries this run's tag
        self.peak = 0
        self.written = 0       # bytes the job's processes have written, those that ended included
        self.written_gone = 0  # what processes that have ended wrote
        self.last_written = {}  # pid -> bytes written at its last reading
        self.writes = None      # guard.Writes, new at each start
        self.noted = None       # the last line written to supervisor.log by note()
        self.owner = None  # (guard.lock handle, victim run folder, deadline, kind) while relieving
        self.resumable = bool(self.spec.get("resumable"))
        self.pressure = None  # the last macOS pressure level seen, logged when it changes
        self.asked_for_room = 0.0  # when this queued job last asked resumable jobs to make room

    def set(self, **fields):
        self.status.update(fields)
        runs.write_json(self.path / "status.json", self.status)

    def run(self, lock):
        signal.signal(signal.SIGTERM, _raise_stopped)
        self.set(supervisor_pid=os.getpid(), supervisor_start=macos.process(os.getpid()).start)
        try:
            while self.wait_for_admission():
                self.start(lock)
                if not self.guard():
                    break
        except Stopped:
            if self.proc:
                self.kill("the supervisor received SIGTERM")
            self.set(state="killed" if self.proc else "stopped", ended=time.time())
        except BaseException:
            if self.proc:
                self.terminate("supervisor error")
            self.set(state="error", error=traceback.format_exc(), ended=time.time())
            raise

    # Admission

    def wait_for_admission(self):
        while True:
            if (self.path / "stop").exists():
                when = "while requeued" if self.status.get("stops") else "before it started"
                self.set(state="stopped", reason=f"stopped {when}", ended=time.time())
                return False
            with runs.locked(self.root / "admit.lock"):
                reason = self.admission()
                if reason is None:
                    self.set(state="running", reason=None, started=self.status.get("started") or time.time())
                    return True
            if reason != self.status.get("reason"):
                self.set(reason=reason)
            time.sleep(1)

    def admission(self):
        """None when the job may start now; otherwise why it waits. Called under admit.lock.

        A job that is not resumable and does not fit asks running resumable jobs to make room.
        """
        others = [(path, runs.read_json(path / "run.json"), runs.read_json(path / "status.json"))
                  for path in runs.all_runs(self.root) if path != self.path and runs.is_live(path)]
        queued = [(spec["created"], bool(spec.get("resumable")))
                  for _, spec, status in others if status["state"] == "queued"]
        if ahead := guard.queued_ahead(self.spec["created"], self.resumable, queued):
            return f"behind {ahead} queued job(s)"
        # A settle time further ahead than one settle period comes from a clock change; ignore it.
        settle_until = runs.read_json(self.root / "guard.json", {"settle_until": 0})["settle_until"]
        if time.time() < settle_until <= time.time() + self.memory["settle"]:
            return "memory readings are settling after a stop"
        if resume := self.backoff():
            return f"stopped for short memory or disk; waits until {time.strftime('%H:%M:%S', time.localtime(resume))}"
        pressure, available = macos.memory_state()
        if pressure == "critical":
            return "memory pressure is critical"
        if short := self.disk_short():
            volume, free = short
            return f"{format_size(free)} free on {volume}; {format_size(self.disk_floor)} must stay free"
        fresh = time.time() - _STALE_POLLS * self.memory["poll"]
        running = []
        for path, spec, status in others:
            if status["state"] == "running":
                live = runs.read_json(path / "live.json", {})
                # Only a job whose supervisor still reports can act on a request to make room.
                yields = bool(spec.get("resumable")) and live.get("updated", 0) >= fresh
                running.append(guard.Job(path.name, spec["budget"], live.get("footprint", 0), yields))
        reason = guard.admission(self.budget, running, self.allowance, available, self.floor)
        if reason is None or self.resumable:
            return reason
        # Freed memory shows in the readings about a second late; asking again sooner would
        # stop more jobs than the first request needed.
        if time.time() < self.asked_for_room + self.memory["settle"]:
            return f"{reason}; waiting for resumable jobs to make room"
        victims = guard.make_room(self.budget, running, self.allowance, available, self.floor)
        if not victims:
            return reason
        request = f"stopped to make room for {self.spec['id']} ({self.spec['name']}); resumable jobs yield"
        asked = sum(self.ask_to_yield(runs.runs_dir(self.root) / run_id, request, "make room") for run_id in victims)
        self.asked_for_room = time.time()
        return f"{reason}; stopping {asked} resumable job(s) to make room"

    def backoff(self):
        """When a requeued job may start again after its last stop for short memory or disk, or
        None if it may start now: the settle time, doubled for each such stop (at most
        _MAX_BACKOFF), so a machine short for a while is not asked again at once."""
        short = [stop["at"] for stop in self.status.get("stops", []) if stop["kind"] in _SHORT]
        if not short:
            return None
        resume = short[-1] + min(self.memory["settle"] * 2 ** (len(short) - 1), _MAX_BACKOFF)
        return resume if time.time() < resume else None

    def disk_short(self):
        """(volume, bytes free) when the swap volume has less than the disk floor free, else
        None. A volume that cannot be read is logged and counts as not short."""
        try:
            free = macos.disk_free(self.volume)
        except OSError as error:
            self.note(f"cannot read free space on {self.volume}: {type(error).__name__}: {error}")
            return None
        return (self.volume, free) if free < self.disk_floor else None

    def ask_to_yield(self, victim, reason, kind):
        """Ask another running job's supervisor to stop its job; True if asked. Call under
        admit.lock, which a yielding supervisor also takes, so a request never outlives the
        stop it asked for."""
        if runs.read_json(victim / "status.json")["state"] != "running":
            return False
        runs.write_json(victim / "guard-stop", {"reason": reason, "kind": kind})
        return True

    # Start

    def start(self, lock):
        if self.status.get("stops"):
            self.record_restart()
        read, self.pipe = os.pipe()
        # Model paths come only from this run's --model choices, never from the caller's environment.
        inherited = {name: value for name, value in os.environ.items()
                     if name != "MODELLAB_MODEL" and not name.startswith("MODELLAB_MODEL_")}
        env = {**inherited, **self.job_env(read)}
        with open(self.path / "log", "ab") as log:
            self.proc = subprocess.Popen(
                self.spec["command"], cwd=self.spec["cwd"], env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, pass_fds=(read, lock), start_new_session=True)
        os.close(read)
        recent = max(self.memory["settle"], _STALE_POLLS * self.memory["poll"])
        self.writes = guard.Writes(self.memory["write_window"], recent, since=self.written_gone)
        self.set(job_pid=self.proc.pid)
        if info := macos.process(self.proc.pid):
            self.tracked[self.proc.pid] = info.start

    def record_restart(self):
        """A restart runs whatever code is in the working folder now; record it beside the
        first start's record (run.json commit and code/)."""
        restarts = self.status.get("restarts", [])
        entry = {"at": time.time(), "commit": None, "code": None}
        if self.spec.get("keeps_code"):
            folder = f"code-{len(restarts) + 2}"
            try:
                entry["commit"] = source.commit(self.spec["cwd"])
                if source.snapshot(self.spec["cwd"], self.path / folder, self.spec["command"]):
                    entry["code"] = folder
            except (Exception, SystemExit) as error:  # the job still restarts; the record says why it is short
                entry["error"] = f"{type(error).__name__}: {error}"
        self.set(restarts=[*restarts, entry])

    def job_env(self, read):
        folders = {"TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "UV_CACHE_DIR": "cache/uv",
                   "UV_PYTHON_INSTALL_DIR": "python"}
        env = {name: str(self.root / folder) for name, folder in folders.items()}
        for folder in env.values():
            Path(folder).mkdir(parents=True, exist_ok=True)
        env |= {
            "MODELLAB_RUN_ID": self.spec["id"],
            "MODELLAB_RUN_DIR": str(self.path),
            "MODELLAB_BUDGET": str(self.budget),
            "MODELLAB_CACHE_LIMIT": str(self.cache_limit),
            "MODELLAB_SUPERVISOR_FD": str(read),
            "HF_HOME": str(self.host.hf_home),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # So a job can use ModelLab's own code, such as modellab.recipe.
            "PYTHONPATH": os.pathsep.join(filter(None, [str(HOME), os.environ.get("PYTHONPATH")])),
        }
        chosen = self.spec.get("models", {})
        for alias, entry in chosen.items():
            env[models.env_var(alias)] = entry["path"]
        if len(chosen) == 1:
            env["MODELLAB_MODEL"] = next(iter(chosen.values()))["path"]
        if self.spec["env"]:
            env_dir = self.root / "envs" / self.spec["env"]
            env["VIRTUAL_ENV"] = str(env_dir)
            env["PATH"] = f"{env_dir / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
        return env

    # Guard

    def guard(self):
        """Watch the job until it ends. True when it was stopped to give memory or disk back and is
        queued again."""
        factor = self.memory["over_budget"]
        stop_deadline = None
        while True:
            exit_code = self.proc.poll()
            footprint = self.measure()
            if exit_code is not None:
                self.finish(exit_code)
                return False
            pressure, available = macos.memory_state()
            self.tend_ownership(pressure, available)
            self.note_pressure(pressure, available)
            job = guard.Job(self.spec["id"], self.budget, footprint, self.resumable)
            request = runs.read_json(self.path / "guard-stop", {})
            if guard.over_budget(job, factor):
                self.kill(f"footprint {format_size(footprint)} passed budget {format_size(self.budget)}")
            elif request:
                if self.give_way(request["reason"], request.get("kind", "memory short")):
                    return True
            elif self.relieve_pressure(pressure, available):
                return True
            else:
                stop_deadline = self.handle_stop(stop_deadline)
            time.sleep(self.memory["poll"])

    def note_pressure(self, pressure, available):
        """Log each change of the macOS pressure level. The warn level alone stops nothing,
        so this log is where it shows."""
        if pressure != self.pressure:
            self.note(f"pressure {pressure}, {format_size(available)} available")
            self.pressure = pressure

    def note(self, line):
        """Write a line to supervisor.log, once until a different line is written."""
        if line != self.noted:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}", flush=True)
            self.noted = line

    def discover(self):
        """Track every live process of the job: its process group, children of tracked
        processes, and any process of this user carrying the run's tag."""
        self.alive()
        found = set(macos.group(self.proc.pid))
        for pid in list(self.tracked):
            found.update(macos.children(pid))
        for pid in macos.user_processes():
            if pid in self.tracked or not (info := macos.process(pid)):
                continue
            key = (pid, info.start)
            if key not in self.checked:
                self.checked[key] = macos.environment_has(pid, self.tag) is True
            if self.checked[key]:
                found.add(pid)
        for pid in found - self.tracked.keys():
            if info := macos.process(pid):
                self.tracked[pid] = info.start
        if len(self.checked) > 20000:
            self.checked.clear()

    def alive(self):
        """Tracked processes that still exist, as {pid: macos.Process}. Drops the ones that are
        gone and keeps what they wrote (up to their last reading)."""
        alive = {}
        for pid, start in list(self.tracked.items()):
            info = macos.process(pid)
            if info and info.start == start:
                alive[pid] = info
                self.last_written[pid] = info.written
            else:
                del self.tracked[pid]
                self.written_gone += self.last_written.pop(pid, 0)
        return alive

    def measure(self):
        self.discover()
        alive = self.alive()
        footprint = sum(info.footprint for info in alive.values())
        self.peak = max(self.peak, footprint)
        self.written = self.written_gone + sum(info.written for info in alive.values())
        now, window = self.writes.rates(time.monotonic(), self.written)
        # write_rate keeps its meaning (the last few seconds) for supervisors still running older code.
        runs.write_json(self.path / "live.json", {
            "footprint": footprint, "budget": self.budget, "resumable": self.resumable, "written": self.written,
            "write_rate": now, "window_rate": window, "write_window": self.memory["write_window"],
            "pids": sorted(alive), "updated": time.time()})
        return footprint

    def kill(self, reason):
        self.set(guard_action=reason)
        self.terminate(reason)

    def give_way(self, reason, kind):
        """Stop the job to give memory or disk back. kind: "memory short", "disk short" or
        "make room".

        A resumable job goes back in the queue (returns True). After more than `requeues`
        stops for short memory or disk it is killed like any other job: it keeps running short itself.
        """
        if self.proc.poll() is not None:
            return False  # it ended on its own; the next pass records that
        if not self.resumable:
            self.kill(reason)
            return False
        stops = [*self.status.get("stops", []), {"at": time.time(), "kind": kind, "reason": reason}]
        short = sum(stop["kind"] in _SHORT for stop in stops)
        if short > self.memory["requeues"]:
            self.set(stops=stops)
            self.kill(f"{reason}; not requeued after {short} stops for short memory or disk")
            return False
        self.kill(reason)
        if self.proc.poll() is None or self.alive():
            return False  # something survived SIGKILL; never start a second copy beside it
        if self.proc.returncode != -signal.SIGKILL:  # it ended on its own just before the kill
            self.status.pop("guard_action", None)
            return False
        os.close(self.pipe)
        with runs.locked(self.root / "admit.lock"):
            (self.path / "guard-stop").unlink(missing_ok=True)
            (self.path / "live.json").unlink(missing_ok=True)
            for leftover in ("guard_action", "survivors", "survivors_reason"):
                self.status.pop(leftover, None)
            self.set(state="queued", reason=f"requeued: {reason}", stops=stops)
        with open(self.path / "log", "ab") as log:
            log.write(f"\nmodellab: {reason}; requeued\n".encode())
        self.proc = None
        self.tracked = {}
        if self.owner:  # a queued job must not keep other supervisors from relieving memory
            self.release_pressure(self.owner[0])
        return True

    def signal_all(self, sig):
        self.proc.poll()
        self.discover()
        for pid in self.alive():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def terminate(self, reason, wait=10):
        """SIGKILL every tracked process until none remains; record any that survive."""
        deadline = time.monotonic() + wait
        while True:
            self.signal_all(signal.SIGKILL)
            time.sleep(0.1)
            self.proc.poll()
            self.discover()
            if not (alive := self.alive()):
                return
            if time.monotonic() > deadline:
                self.set(survivors=sorted(alive), survivors_reason=reason)
                return

    def handle_stop(self, deadline):
        if not (self.path / "stop").exists():
            return None
        if deadline is None:
            self.signal_all(signal.SIGTERM)
            return time.monotonic() + self.memory["stop_grace"]
        if time.monotonic() >= deadline:
            self.kill("did not exit within the stop grace period")
        return deadline

    def relieve_pressure(self, pressure, available):
        """When memory or disk is short, one supervisor at a time stops one job. True when this
        job was the one stopped and is queued again.

        The owner holds guard.lock until its chosen job is gone, then records a settle
        time in guard.json that every supervisor respects, because macOS pressure readings
        lag. The settle time lives in a file because the owner may itself be the job it stops.
        """
        if self.owner:
            return False
        memory_short = guard.short_of_memory(pressure, available, self.floor)
        disk = None if memory_short else self.disk_short()
        if not memory_short and not disk:
            return False
        handle = open(self.root / "guard.lock", "a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        if time.time() < runs.read_json(self.root / "guard.json", {"settle_until": 0})["settle_until"]:
            handle.close()
            return False
        jobs = self.running_jobs()
        if memory_short:
            victim, kind = guard.pressure_victim(jobs), "memory short"
            reason = (f"memory short: {format_size(available)} available (floor {format_size(self.floor)}), "
                      f"pressure {pressure}; ModelLab jobs yield first")
        else:
            (volume, free), kind = disk, "disk short"
            victim = guard.disk_victim(jobs, free)
            reason = (f"disk short: {format_size(free)} free on {volume} (floor {format_size(self.disk_floor)}); "
                      f"ModelLab jobs yield first")
            if victim is None:
                self.note(f"{reason}; no ModelLab job is filling it, so none is stopped; new jobs wait")
            else:
                chosen = next(job for job in jobs if job.run_id == victim)
                span = f"{chosen.window // 60:.0f} min" if chosen.window % 60 == 0 else f"{chosen.window:.0f} s"
                reason += f"; it wrote {format_size(int(chosen.window_rate * chosen.window))} in the last {span}"
        if victim is None:
            handle.close()
            return False
        # At critical pressure wait only a few polls: by then the chosen job's supervisor has
        # acted, or it has stopped reporting and the next choice passes over it.
        wait = _STALE_POLLS * self.memory["poll"] if pressure == "critical" else _PRESSURE_WAIT
        if victim == self.spec["id"]:
            # Settle only once this job's memory is back: the owner releases after the stop.
            self.owner = (handle, self.path, time.monotonic() + wait, kind)
            return self.give_way(reason, kind)
        victim = runs.runs_dir(self.root) / victim
        with runs.locked(self.root / "admit.lock"):
            asked = self.ask_to_yield(victim, reason, kind)
        if not asked:  # it stopped meanwhile; nothing to settle after
            handle.close()
            return False
        self.owner = (handle, victim, time.monotonic() + wait, kind)
        return False

    def tend_ownership(self, pressure, available):
        """Release guard.lock once the chosen job has stopped or its wait has run out. Runs on
        every pass, whatever else happens to this job, so the lock is never held for long.
        When memory runs short during a disk stop, that stop also serves memory, since the job
        gives its memory back too; its wait shrinks to a few polls, then the settle time follows."""
        if not self.owner:
            return
        handle, victim, deadline, kind = self.owner
        if kind == "disk short" and guard.short_of_memory(pressure, available, self.floor):
            deadline = min(deadline, time.monotonic() + _STALE_POLLS * self.memory["poll"])
            self.owner = (handle, victim, deadline, "memory short")
        running = runs.is_live(victim) and runs.read_json(victim / "status.json")["state"] == "running"
        if running and time.monotonic() < deadline:
            return
        self.release_pressure(handle)

    def release_pressure(self, handle):
        runs.write_json(self.root / "guard.json", {"settle_until": time.time() + self.memory["settle"]})
        handle.close()
        self.owner = None

    def running_jobs(self):
        """Running jobs whose supervisors are reporting. A stale report cannot act on guard-stop."""
        fresh = time.time() - _STALE_POLLS * self.memory["poll"]
        jobs = []
        for path in runs.all_runs(self.root):
            if runs.read_json(path / "status.json")["state"] != "running" or not runs.is_live(path):
                continue
            live = runs.read_json(path / "live.json", {})
            if live and live["updated"] >= fresh:
                jobs.append(guard.Job(path.name, live["budget"], live["footprint"], live.get("resumable", False),
                                      live.get("written", 0), live.get("write_rate", 0.0),
                                      live.get("window_rate", 0.0), live.get("write_window", 0.0)))
        return jobs

    def finish(self, exit_code):
        self.discover()
        if self.alive():
            self.terminate("left running after the job's main process ended")
        if self.status.get("guard_action"):
            state = "killed"
        elif (self.path / "stop").exists():
            state = "stopped"
        else:
            state = "exited"
        self.set(state=state, exit_code=exit_code, peak=self.peak, written=self.written, ended=time.time())
        (self.path / "live.json").unlink(missing_ok=True)
        if self.owner:
            self.release_pressure(self.owner[0])


def _raise_stopped(signum, frame):
    raise Stopped()


def main(path, lock):
    Supervisor(path).run(lock)
