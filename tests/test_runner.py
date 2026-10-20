"""The runner end to end: real supervisors and jobs on this Mac, with small budgets."""

import fcntl
import functools
import json
import os
import signal
import subprocess
import sys
import time
import venv

import pytest

from conftest import CLI
from modellab import cli as commands, config, env, macos, runs, source, supervisor

M = 2**20

GROW = ("import time\n"
        "blocks = []\n"
        "for _ in range(32):\n"
        "    blocks.append(bytearray(32 * 2**20)); time.sleep(0.1)\n")


def gone(pid):
    return macos.process(pid) is None


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.1)


def cli(*args):
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True, timeout=120)


def test_run_prints_only_the_run_id_on_standard_output(lab):
    result = cli("run", "--budget", "256M", "--env", "none", "--", sys.executable, "-c", "pass")
    [run_id] = result.stdout.splitlines()
    assert (lab.root / "runs" / run_id / "run.json").exists() and result.stderr.split()[0] in runs.ACTIVE + ("exited",)
    assert lab.wait(lab.root / "runs" / run_id)[0] == "exited"


def test_a_job_runs_and_its_outcome_is_recorded(lab):
    path = lab.run(sys.executable, "-c", "print('hello')")
    state, status = lab.wait(path)
    assert (state, status["exit_code"]) == ("exited", 0)
    assert (path / "log").read_text() == "hello\n"
    assert not (path / "live.json").exists()


def test_a_job_that_grows_past_its_budget_is_killed(lab):
    path = lab.run(sys.executable, "-c", GROW)
    state, status = lab.wait(path)
    assert state == "killed" and "passed budget" in status["guard_action"]
    assert 256 * M < status["peak"] < 2 * 256 * M
    assert gone(status["job_pid"]) and "survivors" not in status


def test_a_child_that_leaves_the_process_group_is_still_guarded(lab):
    parent = ("import subprocess, sys, time\n"
              f"child = subprocess.Popen([sys.executable, '-c', {GROW!r}], start_new_session=True)\n"
              "print(child.pid, flush=True)\n"
              "time.sleep(60)\n")
    path = lab.run(sys.executable, "-c", parent)
    state, status = lab.wait(path)
    child = int((path / "log").read_text().split()[0])
    assert state == "killed" and "passed budget" in status["guard_action"]
    assert gone(child) and gone(status["job_pid"])


def test_a_detached_child_that_outlives_its_parent_is_found_by_its_tag(lab):
    parent = ("import subprocess, sys\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
              " start_new_session=True)\n"
              "print(child.pid, flush=True)\n")
    path = lab.run(sys.executable, "-c", parent)
    state, status = lab.wait(path)
    child = int((path / "log").read_text().split()[0])
    assert (state, status["exit_code"]) == ("exited", 0)
    assert gone(child)


def test_children_left_running_when_the_main_process_ends_are_killed(lab):
    parent = ("import subprocess, sys\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
              "print(child.pid, flush=True)\n")
    path = lab.run(sys.executable, "-c", parent)
    state, _ = lab.wait(path)
    assert state == "exited"
    assert gone(int((path / "log").read_text().split()[0]))


def test_jobs_whose_budgets_do_not_fit_together_run_one_after_the_other(lab):
    lab.configure(reserve="95%")
    budget = int(config.load()["local"].allowance(macos.total_memory()) * 0.6)
    sleep = (sys.executable, "-c", "import time; time.sleep(8)")  # margin for a loaded machine
    first = lab.run(*sleep, budget=str(budget))
    second = lab.run(*sleep, budget=str(budget))
    lab.wait(first, states=("running",))
    wait_for(lambda: "hold" in (runs.read_json(second / "status.json").get("reason") or ""))
    _, one = lab.wait(first)
    _, two = lab.wait(second)
    assert two["started"] >= one["ended"]


def test_a_queued_job_can_be_stopped_before_it_starts(lab):
    lab.configure(reserve="95%")
    budget = int(config.load()["local"].allowance(macos.total_memory()) * 0.6)
    first = lab.run(sys.executable, "-c", "import time; time.sleep(8)", budget=str(budget))  # margin under load
    second = lab.run(sys.executable, "-c", "print('never')", budget=str(budget))
    lab.wait(first, states=("running",))
    assert cli("stop", second.name).returncode == 0
    state, status = lab.wait(second)
    assert state == "stopped" and "job_pid" not in status
    lab.wait(first)


def test_stop_ends_a_job_with_sigterm(lab):
    path = lab.run(sys.executable, "-c", "import time; time.sleep(60)")
    lab.wait(path, states=("running",))
    assert cli("stop", path.name).returncode == 0
    state, status = lab.wait(path)
    assert (state, status["exit_code"]) == ("stopped", -15)


def test_stop_kills_a_job_that_ignores_sigterm(lab):
    lab.configure(stop_grace=1)
    ignore = ("import signal, time\n"
              "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
              "print('ready', flush=True)\n"
              "time.sleep(60)\n")
    path = lab.run(sys.executable, "-c", ignore)
    wait_for(lambda: (path / "log").exists() and "ready" in (path / "log").read_text())
    (path / "stop").touch()
    state, status = lab.wait(path)
    assert (state, status["exit_code"]) == ("killed", -9)
    assert "grace period" in status["guard_action"]


def test_a_guard_stop_request_kills_the_job(lab):
    path = lab.run(sys.executable, "-c", "import time; time.sleep(60)")
    lab.wait(path, states=("running",))
    runs.write_json(path / "guard-stop", {"reason": "chosen to relieve memory"})
    state, status = lab.wait(path)
    assert (state, status["guard_action"]) == ("killed", "chosen to relieve memory")


def test_stray_entries_in_runs_are_ignored(lab):
    (lab.root / "runs").mkdir(parents=True)
    (lab.root / "runs" / ".DS_Store").write_text("")
    path = lab.run(sys.executable, "-c", "print('fine')")
    assert lab.wait(path)[0] == "exited"
    assert cli("ps").returncode == 0


def test_sigterm_to_the_supervisor_kills_its_job(lab):
    path = lab.run(sys.executable, "-c", "import time; time.sleep(60)")
    lab.wait(path, states=("running",))
    status = runs.read_json(path / "status.json")
    os.kill(status["supervisor_pid"], signal.SIGTERM)
    state, status = lab.wait(path)
    assert state == "killed" and "SIGTERM" in status["guard_action"]
    assert gone(status["job_pid"])


def test_a_job_without_the_hook_is_reported_unguarded_when_its_supervisor_dies(lab):
    path = lab.run(sys.executable, "-c", "import time; time.sleep(60)")
    lab.wait(path, states=("running",))
    status = runs.read_json(path / "status.json")
    os.kill(status["supervisor_pid"], signal.SIGKILL)
    try:
        wait_for(lambda: runs.state(path)[0] == "unguarded")
        refused = cli("stop", path.name)
        assert refused.returncode != 0 and "no supervisor" in refused.stderr
    finally:
        os.killpg(status["job_pid"], signal.SIGKILL)


def test_a_python_job_stops_when_its_supervisor_dies(lab):
    env_dir = lab.root / "envs" / "plain"
    venv.create(env_dir, symlinks=True)  # linked, as uv makes them: a copied interpreter can miss its libpython
    env.install_hook(env_dir)
    path = lab.run("python", "-c", "import time; time.sleep(60)", env="plain")
    lab.wait(path, states=("running",))
    time.sleep(1)
    status = runs.read_json(path / "status.json")
    os.kill(status["supervisor_pid"], signal.SIGKILL)
    lab.wait(path, states=("lost",), timeout=5)
    assert gone(status["job_pid"])
    assert "supervisor exited" in (path / "log").read_text()


def test_the_hook_still_installs_when_the_supervisor_pipe_is_missing(lab):
    env_dir = lab.root / "envs" / "plain"
    venv.create(env_dir, symlinks=True)
    env.install_hook(env_dir)
    probe = "import sys; print(sum(type(f).__name__ == '_AfterImport' for f in sys.meta_path))"
    result = subprocess.run(
        [env_dir / "bin" / "python", "-c", probe], capture_output=True, text=True,
        env={**os.environ, "MODELLAB_BUDGET": str(2**30), "MODELLAB_CACHE_LIMIT": str(2**28),
             "MODELLAB_SUPERVISOR_FD": "99"})
    assert result.stdout.strip() == "1"
    assert "not the supervisor pipe" in result.stderr


def test_a_bad_budget_is_refused_with_a_message(lab):
    for budget, message in (("lots", "Not a size"), ("1M", "below the minimum"), ("100T", "allowance")):
        result = cli("run", "--budget", budget, "--env", "none", "--", "true")
        assert result.returncode != 0 and message in result.stderr, result.stderr


def test_a_host_whose_reserve_leaves_jobs_nothing_says_how_to_fix_it(lab):
    lab.configure(reserve="100%")
    result = cli("run", "--budget", "256M", "--env", "none", "--", "true")
    assert result.returncode != 0 and "Set a smaller reserve" in result.stderr, result.stderr


def test_critical_pressure_stops_the_job(lab, monkeypatch):
    path = runs.new_run(lab.root, {
        "host": config.load()["local"].to_json(), "name": "t", "command": ["sleep", "30"],
        "cwd": "/", "env": None, "budget": 256 * M})
    lock = os.open(path / "lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(lock, fcntl.LOCK_EX)
    job = supervisor.Supervisor(path)
    job.set(supervisor_pid=os.getpid(), supervisor_start=macos.process(os.getpid())[1], state="running")
    job.start(lock)
    monkeypatch.setattr(macos, "memory_state", lambda: ("critical", 3 * 2**30))
    job.guard()
    status = runs.read_json(path / "status.json")
    assert status["state"] == "killed" and "pressure critical" in status["guard_action"]
    assert "ModelLab jobs yield first" in status["guard_action"]
    assert gone(status["job_pid"])
    os.close(lock)


def test_critical_pressure_stops_only_the_chosen_job(lab, monkeypatch):
    host = config.load()["local"].to_json()
    large = runs.new_run(lab.root, {"host": host, "name": "large", "command": [], "cwd": "/", "env": None,
                                    "budget": 8 * 2**30})
    runs.write_json(large / "status.json", {"state": "running"})
    runs.write_json(large / "live.json", {"budget": 8 * 2**30, "footprint": 6 * 2**30,
                                          "updated": time.time() + 3600})  # reports throughout the test
    large_lock = open(large / "lock", "a")
    fcntl.flock(large_lock, fcntl.LOCK_EX)
    small = runs.new_run(lab.root, {"host": host, "name": "small", "command": ["sleep", "2"], "cwd": "/",
                                    "env": None, "budget": 256 * M})
    lock = os.open(small / "lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(lock, fcntl.LOCK_EX)
    job = supervisor.Supervisor(small)
    job.set(supervisor_pid=os.getpid(), supervisor_start=macos.process(os.getpid())[1], state="running")
    job.start(lock)
    monkeypatch.setattr(macos, "memory_state", lambda: ("critical", 2**40))
    job.guard()
    assert runs.read_json(small / "status.json")["state"] == "exited"  # it ran to its end
    assert "pressure critical" in runs.read_json(large / "guard-stop")["reason"]
    os.close(lock)
    large_lock.close()


def test_when_memory_is_short_one_supervisor_stops_the_largest_job_then_settles(lab, monkeypatch):
    host = config.load()["local"].to_json()
    locks = []

    def running(name, footprint, updated=None):
        path = runs.new_run(lab.root, {"host": host, "name": name, "command": [], "cwd": "/",
                                       "env": None, "budget": 8 * 2**30})
        runs.write_json(path / "status.json", {"state": "running"})
        runs.write_json(path / "live.json", {"budget": 8 * 2**30, "footprint": footprint,
                                             "updated": updated or time.time()})
        lock = open(path / "lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX)
        locks.append(lock)
        return path

    stale = running("stale", 7 * 2**30, updated=time.time() - 60)  # its supervisor stopped reporting
    small, large = running("small", 1 * 2**30), running("large", 6 * 2**30)
    me = running("me", 2 * 2**30)
    short = ("normal", 2**20)  # below any floor
    monkeypatch.setattr(macos, "memory_state", lambda: short)
    owner = supervisor.Supervisor(me)
    owner.relieve_pressure(*short)
    request = runs.read_json(large / "guard-stop")
    assert request["kind"] == "memory short" and request["reason"].startswith("memory short:")
    assert not (small / "guard-stop").exists() and not (stale / "guard-stop").exists()
    locks[2].close()  # the large job goes
    owner.tend_ownership(*short)  # every pass checks whether the chosen job has stopped
    assert owner.owner is None
    assert runs.read_json(lab.root / "guard.json")["settle_until"] > time.time()
    supervisor.Supervisor(small).relieve_pressure(*short)  # settling: nobody is chosen
    assert not (me / "guard-stop").exists()


def queued_and_running(lab, specs):
    """Run folders in given states, with their locks held as if their supervisors lived."""
    host, made, locks = config.load()["local"].to_json(), {}, []
    for name, state, budget, footprint, resumable in specs:
        path = runs.new_run(lab.root, {"host": host, "name": name, "command": [], "cwd": "/", "env": None,
                                       "budget": budget, "resumable": resumable})
        runs.write_json(path / "status.json", {"state": state})
        if state == "running":
            runs.write_json(path / "live.json", {"budget": budget, "footprint": footprint,
                                                 "resumable": resumable, "updated": time.time() + 3600})
        lock = open(path / "lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX)
        locks.append(lock)
        made[name] = path
    return made, locks


def test_a_job_asks_for_room_once_per_settle_time(lab, monkeypatch):
    small_allowance(lab)
    made, _held = queued_and_running(lab, [("r1", "running", 512 * M, 400 * M, True),
                                           ("r2", "running", 256 * M, 200 * M, True),
                                           ("new", "queued", 512 * M, 0, False)])
    # Room for the new job only once r1's memory is back, and readings that lag behind the stop:
    # without the wait, the second check would stop r2 as well.
    floor = config.parse_size(config.MEMORY_DEFAULTS["floor"], macos.total_memory())
    monkeypatch.setattr(macos, "memory_state", lambda: ("normal", floor + 350 * M))
    new = supervisor.Supervisor(made["new"])
    assert "stopping 1 resumable job(s)" in new.admission()
    assert (made["r1"] / "guard-stop").exists() and not (made["r2"] / "guard-stop").exists()
    runs.write_json(made["r1"] / "status.json", {"state": "queued"})  # r1 yielded; readings lag
    (made["r1"] / "guard-stop").unlink()
    assert "waiting for resumable jobs to make room" in new.admission()
    assert not (made["r2"] / "guard-stop").exists()


def test_admission_waits_for_the_settle_time_and_backs_off_after_short_memory(lab, monkeypatch):
    made, _held = queued_and_running(lab, [("job", "queued", 256 * M, 0, True)])
    monkeypatch.setattr(macos, "memory_state", lambda: ("normal", 2**40))
    job = supervisor.Supervisor(made["job"])
    assert job.admission() is None
    runs.write_json(lab.root / "guard.json", {"settle_until": time.time() + 5})
    assert "settling" in job.admission()
    runs.write_json(lab.root / "guard.json", {"settle_until": time.time() + 86400})
    assert job.admission() is None  # further ahead than one settle period: a clock change, ignored
    runs.write_json(lab.root / "guard.json", {"settle_until": 0})
    job.status["stops"] = [{"at": time.time() - 15, "kind": "memory short", "reason": "t"}] * 2
    assert "stopped for short memory or disk; waits until" in job.admission()  # 2nd stop: 10 s × 2 = 20 s
    job.status["stops"] = [{"at": time.time() - 601, "kind": "memory short", "reason": "t"}] * 20
    assert job.admission() is None  # the wait never grows past ten minutes
    job.status["stops"] = [{"at": time.time() - 15, "kind": "make room", "reason": "t"}] * 2
    assert job.admission() is None  # making room does not back off


def test_a_restart_starts_even_when_its_code_cannot_be_recorded(lab, monkeypatch):
    made, _held = queued_and_running(lab, [("job", "queued", 256 * M, 0, True)])
    spec = runs.read_json(made["job"] / "run.json")
    runs.write_json(made["job"] / "run.json", {**spec, "keeps_code": True})

    def broken(*args):
        raise SystemExit("git diff failed")
    monkeypatch.setattr(source, "snapshot", broken)
    job = supervisor.Supervisor(made["job"])
    job.record_restart()
    [restart] = runs.read_json(made["job"] / "status.json")["restarts"]
    assert restart["code"] is None and "git diff failed" in restart["error"]


def test_the_pressure_owner_lets_go_once_its_choice_stops_or_its_wait_runs_out(lab):
    made, _held = queued_and_running(lab, [("victim", "running", 8 * 2**30, 6 * 2**30, False),
                                           ("owner", "running", 256 * M, 0, False)])
    owner = supervisor.Supervisor(made["owner"])

    def own(deadline):
        handle = open(lab.root / "guard.lock", "a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner.owner = (handle, made["victim"], deadline, "memory short")

    def free():
        with open(lab.root / "guard.lock", "a") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                return False

    calm = ("normal", 2**40)
    own(time.monotonic() + 60)
    owner.tend_ownership(*calm)
    assert owner.owner and not free()          # the choice still runs: keep waiting
    runs.write_json(made["victim"] / "status.json", {"state": "queued"})
    owner.tend_ownership(*calm)
    assert owner.owner is None and free()      # it stopped
    runs.write_json(made["victim"] / "status.json", {"state": "running"})
    own(time.monotonic() - 1)
    owner.tend_ownership(*calm)
    assert owner.owner is None and free()      # its wait ran out
    runs.write_json(made["victim"] / "status.json", {"state": "exited"})
    assert not owner.ask_to_yield(made["victim"], "t", "memory short")
    assert not (made["victim"] / "guard-stop").exists()


def test_admission_waits_while_disk_is_short(lab, monkeypatch):
    made, _held = queued_and_running(lab, [("job", "queued", 256 * M, 0, False)])
    monkeypatch.setattr(macos, "memory_state", lambda: ("normal", 2**40))
    monkeypatch.setattr(macos, "disk_free", lambda path: 2**20)
    reason = supervisor.Supervisor(made["job"]).admission()
    assert "free on" in reason and "must stay free" in reason


def writing(made, rates):
    """Each job writing at `rate` now and over its write window."""
    for name, rate in rates.items():
        live = runs.read_json(made[name] / "live.json")
        runs.write_json(made[name] / "live.json", {**live, "write_rate": rate, "window_rate": rate, "write_window": 1800})


def test_when_disk_is_short_the_job_writing_fastest_is_stopped(lab, monkeypatch):
    made, _held = queued_and_running(lab, [("writer", "running", 1 * 2**30, 100 * M, False),
                                           ("training", "running", 1 * 2**30, 900 * M, True),
                                           ("me", "running", 256 * M, 10 * M, False)])
    writing(made, {"writer": 300 * M, "training": 0.0, "me": 0.0})
    monkeypatch.setattr(macos, "disk_free", lambda path: 2**20)
    owner = supervisor.Supervisor(made["me"])
    assert owner.relieve_pressure("normal", 2**40) is False  # memory is fine; disk is not
    request = runs.read_json(made["writer"] / "guard-stop")
    assert request["kind"] == "disk short" and request["reason"].startswith("disk short:")
    assert not (made["training"] / "guard-stop").exists()  # stopping an idle job frees nothing


def test_when_disk_is_short_and_nothing_fills_it_nothing_is_stopped(lab, monkeypatch, capsys):
    made, _held = queued_and_running(lab, [("idle", "running", 1 * 2**30, 900 * M, True),
                                           ("me", "running", 256 * M, 10 * M, False)])
    writing(made, {"idle": 0.0, "me": 0.0})
    monkeypatch.setattr(macos, "disk_free", lambda path: 2**20)
    owner = supervisor.Supervisor(made["me"])
    assert owner.relieve_pressure("normal", 2**40) is False
    assert not (made["idle"] / "guard-stop").exists() and owner.owner is None
    assert "no ModelLab job is filling it" in capsys.readouterr().out
    assert not (lab.root / "guard.json").exists()  # nothing stopped, so nothing to settle after


def test_memory_comes_before_disk_and_a_pending_disk_stop_gives_way(lab, monkeypatch):
    made, _held = queued_and_running(lab, [("writer", "running", 1 * 2**30, 100 * M, False),
                                           ("big", "running", 4 * 2**30, 3 * 2**30, False),
                                           ("me", "running", 256 * M, 10 * M, False)])
    writing(made, {"writer": 300 * M, "big": 0.0, "me": 0.0})
    monkeypatch.setattr(macos, "disk_free", lambda path: 2**20)
    owner = supervisor.Supervisor(made["me"])
    owner.relieve_pressure("critical", 2**40)  # both short: memory decides
    assert runs.read_json(made["big"] / "guard-stop")["kind"] == "memory short"
    assert not (made["writer"] / "guard-stop").exists()
    owner.release_pressure(owner.owner[0])
    runs.write_json(lab.root / "guard.json", {"settle_until": 0})  # skip the settle time for the test
    (made["big"] / "guard-stop").unlink()
    owner.relieve_pressure("normal", 2**40)   # disk only: the writer is asked
    assert owner.owner[3] == "disk short" and (made["writer"] / "guard-stop").exists()
    owner.tend_ownership("critical", 2**40)   # memory turns critical while it waits
    _, _, deadline, kind = owner.owner         # the pending stop now serves memory too
    assert kind == "memory short" and deadline <= time.monotonic() + 5 * owner.memory["poll"]
    owner.relieve_pressure("critical", 2**40)
    assert not (made["big"] / "guard-stop").exists()  # no second stop while the first is pending
    runs.write_json(made["writer"] / "status.json", {"state": "queued"})
    owner.tend_ownership("critical", 2**40)
    assert owner.owner is None and runs.read_json(lab.root / "guard.json")["settle_until"] > time.time()


def test_an_unreadable_swap_volume_is_logged_and_does_not_stop_anything(lab, monkeypatch, capsys):
    made, _held = queued_and_running(lab, [("job", "queued", 256 * M, 0, False)])

    def unreadable(path):
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(macos, "memory_state", lambda: ("normal", 2**40))
    monkeypatch.setattr(macos, "disk_free", unreadable)
    job = supervisor.Supervisor(made["job"])
    assert job.admission() is None
    assert "cannot read free space" in capsys.readouterr().out


def test_disk_stops_count_toward_the_requeue_limit(lab):
    lab.configure(requeues=0)
    path = lab.run(sys.executable, "-c", TWICE, extra=("--resumable",))
    started(lab, path)
    runs.write_json(path / "guard-stop", {"reason": "disk short: test", "kind": "disk short"})
    state, status = lab.wait(path)
    assert state == "killed" and "not requeued after 1 stops for short memory or disk" in status["guard_action"]


# Rewrites one 8 MiB block about ten times a second: a steady writer that does not fill the disk.
REWRITE = ("import os, sys, time\n"
           "with open(sys.argv[1], 'wb') as f:\n"
           "    while True:\n"
           "        f.seek(0); f.write(os.urandom(8 * 2**20)); f.flush(); os.fsync(f.fileno()); time.sleep(0.1)\n")


def test_a_job_writing_when_disk_runs_short_is_stopped(lab, monkeypatch, tmp_path):
    lab.configure(write_window=60, disk_floor="200G")
    host = config.load()["local"].to_json()
    path = runs.new_run(lab.root, {"host": host, "name": "t", "command": [sys.executable, "-c", REWRITE,
                                                                         str(tmp_path / "block.bin")],
                                   "cwd": "/", "env": None, "budget": 512 * M})
    lock = os.open(path / "lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(lock, fcntl.LOCK_EX)
    job = supervisor.Supervisor(path)
    job.set(supervisor_pid=os.getpid(), supervisor_start=macos.process(os.getpid()).start, state="running")
    job.start(lock)
    monkeypatch.setattr(macos, "memory_state", lambda: ("normal", 2**40))
    monkeypatch.setattr(macos, "disk_free", lambda path: 100 * 2**30)
    job.guard()
    status = runs.read_json(path / "status.json")
    assert status["state"] == "killed" and status["guard_action"].startswith("disk short:")
    assert "it wrote" in status["guard_action"] and "in the last 1 min" in status["guard_action"]
    assert status["written"] >= 8 * M
    assert gone(status["job_pid"])
    os.close(lock)


# Saves one 32 MiB checkpoint, then goes on without writing: a training job between saves.
SAVE = ("import os, sys, time\n"
        "time.sleep(0.6)\n"
        "with open(sys.argv[1], 'wb') as f:\n"
        "    f.write(os.urandom(32 * 2**20)); f.flush(); os.fsync(f.fileno())\n"
        "time.sleep(3)\n")


@pytest.mark.parametrize("window, state", [(60, "exited"), (2, "killed")])
def test_one_save_while_disk_is_short_is_judged_over_the_write_window(lab, monkeypatch, tmp_path, window, state):
    # Over a minute, 32 MiB is about 0.5 MiB/s: below the bar. Over two seconds it is 16 MiB/s.
    lab.configure(write_window=window, disk_floor="200G")
    host = config.load()["local"].to_json()
    path = runs.new_run(lab.root, {"host": host, "name": "t", "command": [sys.executable, "-c", SAVE,
                                                                         str(tmp_path / "adapter.bin")],
                                   "cwd": "/", "env": None, "budget": 512 * M})
    lock = os.open(path / "lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(lock, fcntl.LOCK_EX)
    job = supervisor.Supervisor(path)
    job.set(supervisor_pid=os.getpid(), supervisor_start=macos.process(os.getpid()).start, state="running")
    job.start(lock)
    monkeypatch.setattr(macos, "memory_state", lambda: ("normal", 2**40))
    monkeypatch.setattr(macos, "disk_free", lambda path: 100 * 2**30)
    job.guard()
    status = runs.read_json(path / "status.json")
    assert status["state"] == state and status["written"] >= 32 * M
    os.close(lock)


def test_a_restarted_job_is_judged_on_what_it_writes_after_the_restart(lab, tmp_path):
    lab.configure(write_window=60)
    host = config.load()["local"].to_json()
    path = runs.new_run(lab.root, {"host": host, "name": "t", "command": [sys.executable, "-c", SAVE,
                                                                         str(tmp_path / "adapter.bin")],
                                   "cwd": "/", "env": None, "budget": 512 * M})
    lock = os.open(path / "lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(lock, fcntl.LOCK_EX)
    job = supervisor.Supervisor(path)
    job.set(supervisor_pid=os.getpid(), supervisor_start=macos.process(os.getpid()).start, state="running")
    job.start(lock)
    wait_for(lambda: job.measure() is not None and job.written >= 32 * M)
    assert runs.read_json(path / "live.json")["window_rate"] >= 32 * M / 60
    job.terminate("test")
    job.start(lock)  # the same command again: it waits 0.6 s before it writes
    job.measure()
    live = runs.read_json(path / "live.json")
    assert live["written"] >= 32 * M and live["window_rate"] == 0 and live["write_rate"] == 0
    job.terminate("test")
    os.close(lock)


def test_ps_says_when_disk_is_short(lab):
    lab.configure(disk_floor="1000T")
    assert "Disk short:" in cli("ps").stdout
    lab.configure(disk_floor="1K")
    assert "Disk short:" not in cli("ps").stdout


def test_an_idle_job_runs_on_when_disk_runs_short(lab, monkeypatch):
    host = config.load()["local"].to_json()
    path = runs.new_run(lab.root, {"host": host, "name": "t", "command": ["sleep", "2"], "cwd": "/",
                                   "env": None, "budget": 256 * M})
    lock = os.open(path / "lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(lock, fcntl.LOCK_EX)
    job = supervisor.Supervisor(path)
    job.set(supervisor_pid=os.getpid(), supervisor_start=macos.process(os.getpid()).start, state="running")
    job.start(lock)
    monkeypatch.setattr(macos, "memory_state", lambda: ("normal", 2**40))
    monkeypatch.setattr(macos, "disk_free", lambda path: 2**20)
    job.guard()
    assert runs.read_json(path / "status.json")["state"] == "exited"
    os.close(lock)


def test_a_job_counts_what_its_processes_wrote_including_ended_ones(lab):
    script = ("import subprocess, sys, time\n"
              "writer = ('import os, sys, time\\n'\n"
              "          'with open(sys.argv[1], \"wb\") as f:\\n'\n"
              "          '    f.write(os.urandom(24 * 2**20)); f.flush(); os.fsync(f.fileno())\\n'\n"
              "          'time.sleep(1)\\n')\n"
              "subprocess.run([sys.executable, '-c', writer, 'child.bin'], check=True)\n"
              "time.sleep(1)\n")
    path = lab.run(sys.executable, "-c", script, extra=("--cwd", str(lab.root)))
    state, status = lab.wait(path)
    assert state == "exited", (path / "log").read_text()
    assert status["written"] >= 24 * M  # the child had ended before the job did


def test_warn_pressure_with_memory_to_spare_stops_nothing(lab):
    path = runs.new_run(lab.root, {"host": config.load()["local"].to_json(), "name": "t", "command": [],
                                   "cwd": "/", "env": None, "budget": 8 * 2**30})
    runs.write_json(path / "status.json", {"state": "running"})
    runs.write_json(path / "live.json", {"budget": 8 * 2**30, "footprint": 6 * 2**30, "updated": time.time()})
    assert not supervisor.Supervisor(path).relieve_pressure("warn", 2**40)
    assert not (path / "guard-stop").exists() and not (lab.root / "guard.lock").exists()


@pytest.mark.parametrize("pressure, available, waits_for", [
    ("critical", 2**40, "pressure is critical"), ("normal", 0, "available"), ("warn", 2**40, None)])
def test_admission_waits_only_while_memory_is_short(lab, monkeypatch, pressure, available, waits_for):
    path = runs.new_run(lab.root, {
        "host": config.load()["local"].to_json(), "name": "t", "command": ["true"],
        "cwd": "/", "env": None, "budget": 256 * M})
    monkeypatch.setattr(macos, "memory_state", lambda: (pressure, available))
    reason = supervisor.Supervisor(path).admission()
    assert reason is None if waits_for is None else waits_for in reason


# A resumable job that counts its starts in its run folder, and finishes on its second start.
TWICE = ("import os, pathlib, time\n"
         "starts = pathlib.Path(os.environ['MODELLAB_RUN_DIR'], 'out', 'starts')\n"
         "count = len(starts.read_text()) + 1 if starts.exists() else 1\n"
         "starts.write_text('x' * count)\n"
         "time.sleep(60 if count == 1 else 0)\n")


def started(lab, path):
    """Wait until the TWICE job has counted its first start, so a stop cannot come before it."""
    lab.wait(path, states=("running",))
    wait_for(lambda: (path / "out" / "starts").exists())


def test_a_resumable_job_stopped_for_memory_is_requeued_and_starts_again(lab):
    lab.configure(settle=1)  # the requeued job waits settle × 2^0 before it starts again
    path = lab.run(sys.executable, "-c", TWICE, extra=("--resumable",))
    started(lab, path)
    runs.write_json(path / "guard-stop", {"reason": "memory short: test", "kind": "memory short"})
    state, status = lab.wait(path)
    assert (state, status["exit_code"]) == ("exited", 0)
    assert (path / "out" / "starts").read_text() == "xx"
    assert [stop["kind"] for stop in status["stops"]] == ["memory short"]
    assert "guard_action" not in status and not (path / "guard-stop").exists()
    assert "memory short: test; requeued" in (path / "log").read_text()
    assert "yielded 1x" in cli("ps").stdout


def test_a_resumable_job_that_keeps_running_short_is_killed_after_its_requeues(lab):
    lab.configure(requeues=0)
    path = lab.run(sys.executable, "-c", TWICE, extra=("--resumable",))
    started(lab, path)
    runs.write_json(path / "guard-stop", {"reason": "memory short: test", "kind": "memory short"})
    state, status = lab.wait(path)
    assert state == "killed" and "not requeued after 1 stops" in status["guard_action"]
    assert (path / "out" / "starts").read_text() == "x"


def small_allowance(lab, **memory):
    """Configure an allowance of about 1 GiB, so a few small jobs cannot all fit."""
    allowance_percent = 100 * 2**30 / macos.total_memory()
    lab.configure(reserve=f"{100 - allowance_percent:.4f}%", **memory)


def test_a_job_that_does_not_fit_makes_a_resumable_job_yield_and_runs_first(lab):
    small_allowance(lab, requeues=0)  # making room is not a memory-short stop: it never uses up requeues
    training = lab.run(sys.executable, "-c", TWICE, budget="768M", name="training", extra=("--resumable",))
    started(lab, training)
    small = lab.run(sys.executable, "-c", "import time; time.sleep(1)", budget="512M", name="small")
    assert lab.wait(small)[0] == "exited"
    state, status = lab.wait(training)
    assert (state, (training / "out" / "starts").read_text()) == ("exited", "xx")
    [stop] = status["stops"]
    assert stop["kind"] == "make room" and small.name in stop["reason"]
    small_status = runs.read_json(small / "status.json")
    assert small_status["started"] < status["ended"]  # the small job ran while training waited


def test_a_user_stop_ends_a_requeued_job(lab):
    small_allowance(lab)
    training = lab.run(sys.executable, "-c", TWICE, budget="768M", name="training", extra=("--resumable",))
    started(lab, training)
    small = lab.run(sys.executable, "-c", "import time; time.sleep(5)", budget="512M", name="small")
    lab.wait(small, states=("running",))
    wait_for(lambda: runs.read_json(training / "status.json")["state"] == "queued")
    assert cli("stop", training.name).returncode == 0
    state, status = lab.wait(training)
    assert (state, status["reason"]) == ("stopped", "stopped while requeued")
    assert lab.wait(small)[0] == "exited"


def test_a_resumable_job_over_its_budget_is_killed_for_good(lab):
    path = lab.run(sys.executable, "-c", GROW, extra=("--resumable",))
    state, status = lab.wait(path)
    assert state == "killed" and "passed budget" in status["guard_action"] and "stops" not in status


def test_a_restart_records_the_code_it_starts_from(lab, tmp_path):
    lab.configure(settle=1)
    repo = tmp_path / "project"
    repo.mkdir()
    git = functools.partial(subprocess.run, cwd=repo, check=True, capture_output=True)
    git(["git", "init", "-q"])
    (repo / "job.py").write_text(TWICE)
    git(["git", "add", "."])
    git(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "start"])
    launched = subprocess.run([sys.executable, CLI, "run", "--budget", "256M", "--env", "none", "--resumable", "--",
                               sys.executable, "job.py"], cwd=repo, capture_output=True, text=True, check=True)
    path = lab.root / "runs" / launched.stdout.split()[0]
    started(lab, path)
    assert not (path / "code").exists()  # clean at the first start
    (repo / "job.py").write_text(TWICE + "# edited while the job ran\n")
    runs.write_json(path / "guard-stop", {"reason": "memory short: test", "kind": "memory short"})
    state, status = lab.wait(path)
    assert state == "exited"
    [restart] = status["restarts"]
    assert restart["commit"].endswith("+dirty") and restart["code"] == "code-2"
    assert "edited while the job ran" in (path / "code-2" / "diff.patch").read_text()


def test_a_user_stop_ends_a_resumable_job_for_good(lab):
    path = lab.run(sys.executable, "-c", TWICE, extra=("--resumable",))
    started(lab, path)
    (path / "stop").touch()
    state, status = lab.wait(path)
    assert state == "stopped" and "stops" not in status


def test_the_hook_records_where_mlx_libraries_came_from(lab, tmp_path):
    env_dir = lab.root / "envs" / "plain"
    venv.create(env_dir, symlinks=True)
    env.install_hook(env_dir)
    [site] = env_dir.glob("lib/python*/site-packages")
    (site / "mlx_vlm").mkdir()
    (site / "mlx_vlm" / "__init__.py").write_text("__version__ = (9, 9)\n")  # not a string
    private = tmp_path / "private-build"
    (private / "mlx_lm").mkdir(parents=True)
    (private / "mlx_lm" / "__init__.py").write_text("__version__ = '9.9-private'\n")
    path = lab.run("python", "-c", f"import sys; sys.path.insert(0, {str(private)!r}); import mlx_lm, mlx_vlm",
                   env="plain")
    assert lab.wait(path)[0] == "exited", (path / "log").read_text()
    found = {entry["module"]: entry for entry in runs.libraries(path)}
    assert (found["mlx_vlm"]["version"], found["mlx_vlm"]["inside_env"]) == ("(9, 9)", True)
    assert (found["mlx_lm"]["version"], found["mlx_lm"]["inside_env"]) == ("9.9-private", False)
    assert "mlx_lm loaded from outside the environment" in cli("ps").stdout
    with open(path / "libraries.jsonl", "a") as handle:
        handle.write('{"module": "mlx.co')  # a job killed mid-write
    listing = cli("ps")
    assert listing.returncode == 0 and "libraries.jsonl has unreadable lines" in listing.stdout


def test_a_run_keeps_the_uncommitted_code_it_ran(lab, tmp_path):
    repo = tmp_path / "project"
    (repo / "data").mkdir(parents=True)
    git = functools.partial(subprocess.run, cwd=repo, check=True, capture_output=True)
    git(["git", "init", "-q"])
    (repo / ".gitignore").write_text("data/\n")
    (repo / "train.py").write_text("print('committed')\n")
    git(["git", "add", "."])
    git(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "start"])
    (repo / "train.py").write_text("print('changed')\n")
    (repo / "new_trainer.py").write_text("print('new')\n")
    (repo / "data" / "secret.env").write_text("TOKEN=1\n")
    (repo / "big.bin").write_bytes(b"0" * (source.FILE_LIMIT + 1))
    result = subprocess.run([sys.executable, CLI, "run", "--budget", "256M", "--env", "none", "--",
                             sys.executable, "train.py"], cwd=repo, capture_output=True, text=True, check=True)
    path = lab.root / "runs" / result.stdout.split()[0]
    spec = runs.read_json(path / "run.json")
    assert spec["commit"].endswith("+dirty") and spec["resumable"] is False
    manifest = runs.read_json(path / "code" / "manifest.json")
    assert "print('changed')" in (path / "code" / "diff.patch").read_text()
    assert [entry["path"] for entry in manifest["untracked"]] == ["new_trainer.py"]
    assert (path / "code" / "untracked" / "new_trainer.py").read_text() == "print('new')\n"
    assert [(entry["path"], entry["reason"]) for entry in manifest["skipped"]] == [("big.bin", "larger than 1.0M")]
    assert not (path / "code" / "untracked" / "data").exists()  # ignored files are never copied
    assert lab.wait(path)[0] == "exited"


def test_a_clean_checkout_keeps_no_code_copy(lab, tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    git = functools.partial(subprocess.run, cwd=repo, check=True, capture_output=True)
    git(["git", "init", "-q"])
    (repo / "train.py").write_text("print('committed')\n")
    git(["git", "add", "."])
    git(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "start"])
    result = subprocess.run([sys.executable, CLI, "run", "--budget", "256M", "--env", "none", "--",
                             sys.executable, "train.py"], cwd=repo, capture_output=True, text=True, check=True)
    path = lab.root / "runs" / result.stdout.split()[0]
    assert not (path / "code").exists() and not runs.read_json(path / "run.json")["commit"].endswith("+dirty")
    assert lab.wait(path)[0] == "exited"


def test_wait_reports_each_run_as_it_ends_and_fails_if_any_did(lab):
    good = lab.run(sys.executable, "-c", "import time; time.sleep(1)", name="good")
    bad = lab.run(sys.executable, "-c", "import sys; sys.exit(3)", name="bad")
    result = cli("wait", good.name, bad.name)
    assert result.returncode == 1
    header, *rows = result.stdout.splitlines()
    ended = {row.split()[0]: row for row in rows}  # in the order they ended, which admission decides
    assert header.startswith("RUN") and len(rows) == 2
    assert "exit 3" in ended[bad.name] and "exit 0" in ended[good.name]
    assert cli("wait", good.name).returncode == 0  # a run that has ended answers at once


def test_wait_counts_a_requeued_job_as_still_running(lab):
    lab.configure(settle=1)
    path = lab.run(sys.executable, "-c", TWICE, extra=("--resumable",))
    started(lab, path)
    waiting = subprocess.Popen([sys.executable, CLI, "wait", "--json", path.name], stdout=subprocess.PIPE, text=True)
    try:
        runs.write_json(path / "guard-stop", {"reason": "memory short: test", "kind": "memory short"})
        output, _ = waiting.communicate(timeout=30)
    finally:
        waiting.kill()
    [record] = [json.loads(line) for line in output.splitlines()]
    assert waiting.returncode == 0 and (path / "out" / "starts").read_text() == "xx"
    assert (record["id"], record["state"], record["exit_code"], record["yields"]) == (path.name, "exited", 0, 1)


def test_wait_gives_up_at_its_timeout_and_leaves_the_run_alone(lab):
    path = lab.run(sys.executable, "-c", "import time; time.sleep(60)")
    lab.wait(path, states=("running",))
    result = cli("wait", "--timeout", "0.5", path.name)
    assert result.returncode == 124 and f"Still active after 0.5 s: {path.name}" in result.stderr
    assert runs.state(path)[0] == "running"
    cli("stop", path.name)
    assert lab.wait(path)[0] == "stopped"


def test_wait_tells_a_mistaken_run_from_a_failed_one(lab):
    result = cli("wait", "20260101-000000-0000")
    assert result.returncode == 2 and "No run '20260101-000000-0000'" in result.stderr
    assert cli("wait", "--host", "nowhere", "20260101-000000-0000").returncode == 2


def test_ps_keeps_an_older_live_run_and_gives_json_records(lab):
    server = lab.run(sys.executable, "-c", "import time; time.sleep(60)", name="long")
    lab.wait(server, states=("running",))
    wait_for(lambda: (server / "live.json").exists())
    later = [f"29990101-0000{n:02d}-0000" for n in range(20)]
    for run_id in later:  # 20 finished runs after it: the latest 20 alone would hide it
        runs.write_json(lab.root / "runs" / run_id / "run.json",
                        {"id": run_id, "created": 1, "name": "done", "budget": 256 * M})
        runs.write_json(lab.root / "runs" / run_id / "status.json",
                        {"state": "exited", "exit_code": 0, "peak": M, "started": 1, "ended": 2})
    records = [json.loads(line) for line in cli("ps", "--json").stdout.splitlines()]
    assert [record["id"] for record in records] == [server.name, *later]
    assert records[0]["state"] == "running" and records[0]["footprint"] > 0 and records[0]["dir"] == str(server)
    assert (records[1]["state"], records[1]["exit_code"], records[1]["footprint"]) == ("exited", 0, None)
    assert server.name in cli("ps").stdout
    cli("stop", server.name)
    lab.wait(server)
    assert server.name not in cli("ps").stdout and server.name in cli("ps", "--all").stdout


RECORD = {"id": "20261020-120000-abcd", "name": "serve qwen", "state": "running", "exit_code": None, "reason": None,
          "guard_action": None, "error": None, "budget": 12 * 2**30, "footprint": 9 * 2**30, "peak": None,
          "created": 1.0, "started": None, "ended": None, "resumable": False, "yields": 0, "dir": "/r",
          "serve": {"model": "qwen", "url": "http://127.0.0.1:50123/v1"}, "outside_env": [],
          "unreadable_libraries": False}


def test_ps_rows_keep_the_text_programs_already_read():
    # Programs find a server by a line holding "running", "serve <alias>" and the URL; keep that text.
    assert commands._ps_row(RECORD) == ("20261020-120000-abcd  running    12.0G    9.0G       -   0m00s  "
                                        "serve qwen / http://127.0.0.1:50123/v1")
    ended = {**RECORD, "name": "train", "state": "exited", "exit_code": 0, "footprint": None, "peak": 2**30,
             "started": 100.0, "ended": 3700.0, "yields": 2, "serve": None, "outside_env": ["mlx_lm"]}
    assert commands._ps_row(ended) == ("20261020-120000-abcd  exited     12.0G       -    1.0G   1h00m  train / "
                                       "exit 0; yielded 2x; mlx_lm loaded from outside the environment")
