"""Configuration, macOS readings and guard decisions."""

import os
import signal
import subprocess
import sys
import time

import pytest

from modellab import config, guard, macos, probe

G = 2**30


def test_sizes():
    total = 100 * G
    assert config.parse_size("40G", total) == 40 * G
    assert config.parse_size("512MiB", total) == 512 * 2**20
    assert config.parse_size("15%", total) == 15 * G
    assert config.parse_size("max(8G, 15%)", total) == 15 * G
    assert config.parse_size("min(4G, 15%)", total) == 4 * G
    with pytest.raises(ValueError, match="Not a size"):
        config.parse_size("lots", total)


def test_without_a_config_file_there_is_one_local_host(tmp_path):
    hosts = config.load(tmp_path / "absent.toml")
    assert list(hosts) == ["local"]
    assert hosts["local"].allowance(100 * G) == 85 * G  # reserve max(8G, 15%)
    assert hosts["local"].size("disk_floor", 100 * G) == 100 * G  # free disk for all of memory as swap


def test_host_settings_override_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[memory]\nfloor = "5%"\n[hosts.small]\nroot = "/r"\nmemory = {reserve = "2G"}\n')
    host = config.load(path)["small"]
    assert (host.allowance(16 * G), host.size("floor", 100 * G)) == (14 * G, 5 * G)


@pytest.mark.parametrize("text, message", [
    ('[mystery]\n', "unknown sections"),
    ('[hosts.a]\nrooot = "/r"\n', "unknown keys"),
    ('[hosts.a]\nmemory = {reserv = "1G"}\n', "unknown memory keys"),
    ('[memory]\nwrite_window = 0\n', "write_window must be a positive number"),
    ('[memory]\nwrite_window = -60\n', "write_window must be a positive number"),
    ('[memory]\nwrite_window = "30m"\n', "write_window must be a positive number"),
    ('[memory]\nwrite_window = true\n', "write_window must be a positive number"),
    ('[memory]\nwrite_window = nan\n', "write_window must be a positive number"),
    ('[hosts.a]\nmemory = {write_window = 0}\n', "write_window must be a positive number"),
    ('[memory]\npoll = 0\n', "poll must be a positive number"),
])
def test_config_mistakes_are_rejected(tmp_path, text, message):
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        config.load(path)


def test_admission_counts_budgets_and_the_room_running_jobs_may_still_grow_into():
    running = [guard.Job("a", 20 * G, 5 * G)]
    assert guard.admission(10 * G, running, 40 * G, 60 * G, 5 * G) is None
    assert "hold" in guard.admission(30 * G, running, 40 * G, 60 * G, 5 * G)
    assert "grow" in guard.admission(10 * G, running, 40 * G, 28 * G, 5 * G)  # 10G + 15G > 28G - 5G
    assert "available" in guard.admission(10 * G, [], 40 * G, 14 * G, 5 * G)


def test_pressure_victim_prefers_over_budget_then_resumable_then_the_largest():
    small_over = guard.Job("a", 1 * G, int(1.05 * G))
    large = guard.Job("b", 40 * G, 30 * G)
    larger = guard.Job("c", 60 * G, 50 * G)
    small_resumable = guard.Job("d", 8 * G, 4 * G, resumable=True)
    assert guard.pressure_victim([small_over, large, larger, small_resumable]) == "a"
    assert guard.pressure_victim([large, larger, small_resumable]) == "d"
    assert guard.pressure_victim([large, larger]) == "c"
    assert guard.pressure_victim([]) is None


def test_disk_victim_is_the_fastest_writer_and_never_an_idle_job():
    M = 2**20

    def job(name, now, window, **kw):
        return guard.Job(name, 8 * G, 1 * G, write_rate=now, window_rate=window, window=1800, **kw)

    fast = job("a", 200 * M, 200 * M, written=1 * G)
    slow = job("b", 5 * M, 5 * M, resumable=True, written=90 * G)
    idle = job("c", 0.0, 0.0, resumable=True, written=500 * G)
    logging = job("d", guard.WRITING / 10, guard.WRITING / 10)
    saved = job("e", 0.0, 2 * G / 1800)       # saved 2 GiB a while ago and stopped writing
    older = guard.Job("f", 8 * G, 1 * G, write_rate=200 * M)  # an older supervisor publishes no window rate
    assert guard.disk_victim([fast, slow, idle, logging, saved, older], 100 * G) == "a"
    assert guard.disk_victim([slow, idle], 100 * G) == "b"
    assert guard.disk_victim([idle, logging, saved, older], 100 * G) is None  # stopping them would stop nothing
    # With little left, a slower writer counts once it would fill the rest within a day.
    assert guard.disk_victim([idle, logging], 1 * G) == "d"
    assert guard.disk_victim([idle, saved], 0) is None
    assert guard.disk_victim([], 100 * G) is None


def test_writes_are_counted_from_the_start_or_restart():
    writes = guard.Writes(60, 10, since=5 * G)  # a restart after 5 GiB
    assert writes.rates(0.0, 5 * G) == (0.0, 0.0)
    assert writes.rates(1.0, 5 * G + 600) == (60.0, 10.0)


def test_the_rate_now_holds_when_the_window_is_shorter():
    writes = guard.Writes(5, 10)
    rates = [writes.rates(i * 0.5, i * 0.5 * 2**20) for i in range(60)]  # 1 MiB/s steady
    assert rates[-1] == (2**20, 2**20)


def replay(readings, free, window=1800):
    """The disk victim chosen at each (time, bytes written so far) reading of one job."""
    writes = guard.Writes(window, 10)
    chosen = []
    for now, written in readings:
        rate_now, rate_window = writes.rates(now, written)
        job = guard.Job("job", 64 * G, 60 * G, written=written, write_rate=rate_now, window_rate=rate_window,
                        window=window)
        chosen.append((now, guard.disk_victim([job], free)))
    return chosen


def test_one_checkpoint_save_does_not_make_a_job_the_disk_victim():
    # 2026-10-02: a training run wrote only its log for 13 hours, then saved its adapter (two
    # 96 MB files) while another project filled the disk to 123.8 GiB free. Readings every 0.5 s.
    quiet = [(i * 0.5, i * 4) for i in range(13 * 3600 * 2)]
    end = quiet[-1][1]
    after = [(13 * 3600 + i * 0.5, end + 191_818_780) for i in range(1, 2 * 3600)]
    assert {victim for _, victim in replay(quiet + after, 123.8 * G)} == {None}
    # The ten-second rate used until then took the save for the disk filling and stopped the job.
    assert "job" in {victim for _, victim in replay(quiet + after, 123.8 * G, window=10)}


def first_chosen(readings, free):
    return next(now for now, victim in replay(readings, free) if victim)


def test_steady_writing_and_stacked_checkpoints_still_make_a_job_the_disk_victim():
    steady = [(i * 0.5, i * 150 * 10**6) for i in range(120)]  # 300 MB/s
    assert first_chosen(steady, 123.8 * G) <= 10
    stacked = [(i * 0.5, (i // 60) * 192 * 10**6) for i in range(2 * 3600)]  # 192 MB every 30 s
    assert first_chosen(stacked, 123.8 * G) <= 6 * 60


def test_a_large_save_counts_only_while_it_is_written():
    # 1 KB/s of log, and one 2 GiB save at 100 s: above the bar over the window, but the job
    # stops writing at 100.5 s, and stopping it after that would stop nothing.
    readings = [(i * 0.5, i * 500 + (2 * G if i >= 200 else 0)) for i in range(2 * 3600)]
    chosen = [now for now, victim in replay(readings, 123.8 * G) if victim]
    assert chosen and min(chosen) == 100.0 and max(chosen) < 110.0


def test_with_little_free_one_save_counts():
    save = [(i * 0.5, 0 if i < 10 else 192 * 10**6) for i in range(40)]
    chosen = [now for now, victim in replay(save, 5 * G) if victim]
    assert min(chosen) == 5.0 and max(chosen) < 15.0  # it would fill 5 GiB within a day at its rate


def test_memory_is_short_only_at_critical_pressure_or_below_the_floor():
    assert not guard.short_of_memory("normal", 20 * G, 10 * G)
    assert not guard.short_of_memory("warn", 20 * G, 10 * G)  # warn with room to spare stops nothing
    assert guard.short_of_memory("critical", 20 * G, 10 * G)
    assert guard.short_of_memory("normal", 5 * G, 10 * G)
    assert guard.short_of_memory("warn", 5 * G, 10 * G)


def test_queued_jobs_pass_resumable_ones():
    older, newer = 1.0, 2.0
    assert guard.queued_ahead(newer, False, [(older, False)]) == 1
    assert guard.queued_ahead(newer, False, [(older, True)]) == 0     # a resumable job never holds others up
    assert guard.queued_ahead(older, True, [(newer, False)]) == 1     # it waits even for newer jobs
    assert guard.queued_ahead(newer, True, [(older, True)]) == 1      # resumable jobs keep their order
    assert guard.queued_ahead(older, True, [(newer, True)]) == 0


def test_making_room_stops_resumable_jobs_largest_first_and_only_when_that_is_enough():
    training = guard.Job("t", 60 * G, 50 * G, resumable=True)
    small_training = guard.Job("s", 10 * G, 8 * G, resumable=True)
    server = guard.Job("v", 20 * G, 15 * G)
    running = [training, small_training, server]
    allowance, floor = 100 * G, 10 * G
    assert guard.admission(20 * G, running, allowance, 30 * G, floor) is not None
    assert guard.make_room(20 * G, running, allowance, 30 * G, floor) == ["t"]
    assert guard.make_room(85 * G, running, allowance, 30 * G, floor) is None  # even both would not do
    assert guard.make_room(20 * G, [server], allowance, 5 * G, floor) is None  # nothing resumable to stop


def test_process_readings_follow_a_child():
    child = subprocess.Popen([sys.executable, "-c",
                              "import subprocess, sys, time\n"
                              "b = bytearray(200 * 2**20)\n"
                              "subprocess.Popen(['sleep', '5'])\n"
                              "time.sleep(5)"], start_new_session=True)
    try:
        time.sleep(1)
        footprint, start, _ = macos.process(child.pid)
        assert footprint > 200 * 2**20
        assert macos.process(child.pid).start == start
        [grandchild] = macos.children(child.pid)
        assert set(macos.group(child.pid)) == {child.pid, grandchild}
    finally:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    assert macos.process(child.pid) is None


def test_the_run_tag_is_read_from_a_process_environment():
    tagged = {**os.environ, "MODELLAB_RUN_ID": "r1"}
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], env=tagged)
    try:
        time.sleep(0.5)
        assert macos.environment_has(child.pid, "MODELLAB_RUN_ID=r1") is True
        assert macos.environment_has(child.pid, "MODELLAB_RUN_ID=r2") is False
        assert child.pid in macos.user_processes()
    finally:
        child.kill()
        child.wait()


def test_bytes_written_are_read_from_a_process(tmp_path):
    child = subprocess.Popen([sys.executable, "-c",
                              "import os, sys, time\n"
                              "with open(sys.argv[1], 'wb') as f:\n"
                              "    f.write(os.urandom(32 * 2**20)); f.flush(); os.fsync(f.fileno())\n"
                              "time.sleep(5)\n", str(tmp_path / "out.bin")])
    try:
        deadline = time.monotonic() + 5
        while macos.process(child.pid).written < 32 * 2**20:
            assert time.monotonic() < deadline, macos.process(child.pid)
            time.sleep(0.1)
    finally:
        child.kill()
        child.wait()


def test_disk_readings(tmp_path):
    root, missing = macos.disk_free("/"), macos.disk_free("/a/folder/that/does/not/exist")
    assert root > 0 and abs(root - missing) < 2**30  # the same volume; other writers move it a little
    assert macos.disk_free("relative/missing") > 0
    assert os.path.isdir(macos.swap_volume())
    report = probe.run(tmp_path, disk_floor=2**60)
    assert not report["disk readings"]["ok"] and "below the floor" in report["disk readings"]["detail"]


def test_memory_state():
    pressure, available = macos.memory_state()
    assert pressure in {"normal", "warn", "critical"}
    assert 0 < available <= macos.total_memory()
