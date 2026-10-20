"""modellab: run guarded jobs and manage environments on configured Macs."""

import argparse
import functools
import fcntl
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from . import HOME, config, env, gold, macos, models, probe, remote, revision, runs, source
from .config import format_size

Parser = functools.partial(argparse.ArgumentParser, allow_abbrev=False)  # "--hos" must not mean "--host"
USAGE = 2  # a mistake in the command, such as an unknown run: argparse's own status for one
TIMED_OUT = 124  # `wait --timeout` ran out, as timeout(1) exits
SSH_FAILED = 255  # ssh's own exit status when the connection fails or drops
REMOTE_MARGIN = 30  # seconds past `wait --timeout` for SSH to connect and the host to answer


def main(argv=None):
    parser = Parser(prog="modellab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)

    run = commands.add_parser("run", help="run a command as a guarded job")
    run.add_argument("--host", default="local")
    run.add_argument("--budget", help="memory the job may use, e.g. 40G (default: what can be admitted now)")
    run.add_argument("--env", help="environment to run in, or 'none' (default: the host's)")
    run.add_argument("--name")
    run.add_argument("--cwd", help="working directory (default: the current one)")
    run.add_argument("--sync", action="append", default=[], metavar="PATH",
                     help="remote hosts: copy PATH into <root>/work before the job starts (the default cwd there)")
    run.add_argument("--resumable", action="store_true",
                     help="the job resumes from its own checkpoints: ModelLab may stop it to make room for "
                          "other jobs or when memory is short, and starts it again when there is room")
    run.add_argument("--commit", help=argparse.SUPPRESS)
    run.add_argument("--model", action="append", default=[], metavar="ALIAS",
                     help="a pulled model; its folder is passed as $MODELLAB_MODEL_<ALIAS> "
                          "(and $MODELLAB_MODEL when only one is given)")
    run.add_argument("argv", nargs=argparse.REMAINDER, metavar="-- command ...")
    run.set_defaults(handler=cmd_run)

    serve = commands.add_parser("serve", help="serve a pulled model on 127.0.0.1 as a guarded job")
    serve.add_argument("alias")
    serve.add_argument("--host", default="local")
    serve.add_argument("--budget", help="memory the server may use (default: what can be admitted now)")
    serve.add_argument("--port", type=int, help="default: a free port")
    serve.add_argument("--adapter", help="a LoRA adapter folder to load with the model")
    serve.add_argument("--max-num-seqs", type=int, default=8, help="vision models: concurrent sequences")
    serve.add_argument("--top-logprobs-k", type=int, default=20, help="vision models: log-probabilities returned")
    serve.add_argument("--wait", type=float, default=60, help="seconds to wait for the server to answer")
    serve.set_defaults(handler=cmd_serve)

    ps = commands.add_parser("ps", help="list runs")
    ps.add_argument("--host", default="local")
    ps.add_argument("--all", action="store_true", help="all runs, not only the latest 20 and those still active")
    ps.add_argument("--json", action="store_true", help="one JSON record per run")
    ps.set_defaults(handler=cmd_ps)

    wait = commands.add_parser("wait", help="wait until runs end; exit 0 only if each exited 0")
    wait.add_argument("run_ids", nargs="+", metavar="run_id")
    wait.add_argument("--host", default="local")
    wait.add_argument("--timeout", type=float, help=f"seconds; exit {TIMED_OUT} if a run is still active then")
    wait.add_argument("--json", action="store_true", help="each run's record as it ends, as `ps --json` gives it")
    wait.set_defaults(handler=cmd_wait)

    logs = commands.add_parser("logs", help="print a run's output")
    logs.add_argument("run_id")
    logs.add_argument("--host", default="local")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.set_defaults(handler=cmd_logs)

    stop = commands.add_parser("stop", help="stop a run (SIGTERM, then SIGKILL after a grace period)")
    stop.add_argument("run_id")
    stop.add_argument("--host", default="local")
    stop.set_defaults(handler=cmd_stop)

    env_parser = commands.add_parser("env", help="manage environments")
    env_commands = env_parser.add_subparsers(dest="env_command", required=True, parser_class=Parser)
    sync = env_commands.add_parser("sync", help="install an environment's lock on a host")
    sync.add_argument("name", nargs="?", default="mlx")
    sync.add_argument("--host", default="local")
    sync.set_defaults(handler=cmd_env_sync)

    models_parser = commands.add_parser("models", help="pin, pull and locate models")
    model_commands = models_parser.add_subparsers(dest="models_command", required=True, parser_class=Parser)
    add = model_commands.add_parser("add", help="pin a Hub model to one commit: REPO[@REVISION]")
    add.add_argument("spec")
    add.add_argument("--alias")
    add.add_argument("--replace", action="store_true", help="repoint an existing alias")
    pull = model_commands.add_parser("pull", help="download and verify models (default: all)")
    pull.add_argument("aliases", nargs="*")
    where = model_commands.add_parser("path", help="print a pulled model's folder")
    where.add_argument("alias")
    listing = model_commands.add_parser("list", help="list pinned models and whether they are pulled")
    add.set_defaults(handler=cmd_models_add)
    for parser_, handler in ((pull, cmd_models_pull), (where, cmd_models_path), (listing, cmd_models_list)):
        parser_.add_argument("--host", default="local")
        parser_.set_defaults(handler=handler)

    fetch = commands.add_parser("fetch", help="copy a remote run's folder (log, status, out/) to this Mac")
    fetch.add_argument("run_id")
    fetch.add_argument("--host", required=True)
    fetch.add_argument("--to", help="destination folder (default: <local root>/fetched/<host>/<run-id>)")
    fetch.set_defaults(handler=cmd_fetch)

    gold_parser = commands.add_parser("gold", help="check and freeze gold sets (docs/gold-sets.md)")
    gold_commands = gold_parser.add_subparsers(dest="gold_command", required=True, parser_class=Parser)
    gold_check = gold_commands.add_parser("check", help="report every problem in a gold set")
    gold_check.add_argument("set", help="a gold-set folder, or <folder>@<version>")
    gold_check.add_argument("--outputs", action="append", default=[], metavar="JSONL",
                            help="outputs joined to the items by id, checked with them")
    gold_check.set_defaults(handler=cmd_gold_check)
    gold_freeze = gold_commands.add_parser("freeze", help="keep the set as it is now, as a version with its own id")
    gold_freeze.add_argument("set")
    gold_freeze.set_defaults(handler=cmd_gold_freeze)

    look = commands.add_parser("look", help="render a gold set, with outputs, as sheets an agent can see whole")
    look.add_argument("set", help="a gold-set folder, or <folder>@<version>")
    look.add_argument("--outputs", action="append", default=[], metavar="JSONL",
                      help="outputs joined to the items by id; their media paths are relative to the file")
    look.add_argument("--sample", type=int, default=24, help="how many items (default 24)")
    look.add_argument("--seed", type=int, help="for the random sample (default: a new one, recorded)")
    which = look.add_mutually_exclusive_group()
    which.add_argument("--lowest", metavar="FIELD", help="the items with the lowest number in FIELD")
    which.add_argument("--highest", metavar="FIELD", help="the items with the highest number in FIELD")
    look.add_argument("--size", default="1600x1600", help="sheet size in pixels (default 1600x1600)")
    look.add_argument("--to", help="a new folder for the sheets (default: ./look-<set>-<time>)")
    look.add_argument("--blind", action="store_true",
                      help="a judge's view: targets withheld, shuffled panes lettered per item, the key beside the folder")
    look.set_defaults(handler=cmd_look)

    colour = commands.add_parser("colour", help="how different pictures of the same framing look (CIEDE2000)")
    colour.add_argument("pictures", nargs="+", metavar="FIRST SECOND", help="pairs of pictures, compared in order")
    colour.add_argument("--json", action="store_true", help="one JSON report per pair")
    colour.set_defaults(handler=cmd_colour)

    report = commands.add_parser("report", help="an evaluation's items first, then its numbers (an Inspect AI log)")
    report.add_argument("log", help="an Inspect AI log (.eval)")
    report.add_argument("--set", required=True, help="the gold set its sample ids belong to")
    report.add_argument("--scorer", help="which scorer to report (default: the log's first)")
    report.add_argument("--sample", type=int, default=24, help="items per sheet set (default 24)")
    report.add_argument("--seed", type=int, help="for the sample and the intervals (default: a new one, recorded)")
    report.add_argument("--to", help="a new folder (default: ./report-<log>)")
    report.set_defaults(handler=cmd_report)

    label = commands.add_parser("label", help="answer a gold set's questions in a page on this Mac")
    label.add_argument("set", help="a gold-set folder (answers go to its labels.jsonl)")
    label.add_argument("--labeller", default=os.environ.get("USER", "me"), help="who is answering (default: $USER)")
    label.add_argument("--port", type=int, default=0, help="default: a free port")
    label.add_argument("--no-open", action="store_true", help="print the address without opening a browser")
    label.set_defaults(handler=cmd_label)

    host_parser = commands.add_parser("host", help="prepare hosts")
    host_commands = host_parser.add_subparsers(dest="host_command", required=True, parser_class=Parser)
    init = host_commands.add_parser("init", help="set up ModelLab on a host and report what the host allows")
    init.add_argument("name")
    init.set_defaults(handler=cmd_host_init)

    probe_parser = commands.add_parser("_probe")
    probe_parser.set_defaults(handler=cmd_probe)
    record = commands.add_parser("_record-pulled")
    record.add_argument("alias")
    record.add_argument("path")
    record.add_argument("how")
    record.set_defaults(handler=cmd_record_pulled)

    supervise = commands.add_parser("_supervise")
    supervise.add_argument("path")
    supervise.add_argument("lock", type=int)
    supervise.set_defaults(handler=cmd_supervise)

    args = parser.parse_args(argv)
    args.argv_given = list(sys.argv[1:] if argv is None else argv)
    return args.handler(args) or 0


def host_named(name):
    hosts = config.load()
    if name not in hosts:
        raise SystemExit(f"Unknown host {name!r}. Configured: {', '.join(hosts)} ({config.config_path()})")
    return hosts[name]


def local_host(name="local"):
    host = host_named(name)
    if host.ssh:
        raise SystemExit(f"Host {name!r} is remote; this command runs on this Mac only")
    return host


def forwarded(args):
    """The command as given, without its --host option (words after `--` are the job's own)."""
    given, kept, skip = args.argv_given, [], False
    for index, word in enumerate(given):
        if word == "--":
            return kept + given[index:]
        if skip:
            skip = False
        elif word == "--host":
            skip = True
        elif not word.startswith("--host="):
            kept.append(word)
    return kept


def forward(host, args, **kwargs):
    """Run the same command on a remote host."""
    return remote.invoke(host, forwarded(args), **kwargs)


MIN_BUDGET = 64 * 2**20


def cmd_run(args):
    host = host_named(args.host)
    command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not command:
        raise SystemExit("Give the command after --, e.g. modellab run -- python train.py")
    name = args.name or Path(command[0]).name
    if host.ssh:
        remote.push_code(host)
        work = remote.prepare_work(host, args.sync)
        cwd = args.cwd if args.cwd and args.cwd.startswith("/") else f"{work}/{args.cwd}" if args.cwd else work
        argv = ["run", "--name", name, "--cwd", cwd, "--commit", source.commit(Path.cwd()) or "none"]
        for option, value in (("--budget", args.budget), ("--env", args.env)):
            if value:
                argv += [option, value]
        if args.resumable:
            argv.append("--resumable")
        for alias in args.model:
            argv += ["--model", alias]
        return remote.invoke(host, [*argv, "--", *command]).returncode
    if args.sync:
        raise SystemExit("--sync copies inputs to a remote host; this host is local")
    path = submit(host, command, args.budget, args.env, args.model, name, Path(args.cwd or os.getcwd()).resolve(),
                  commit=args.commit, resumable=args.resumable)
    report_start(path)


def submit(host, command, budget_text, env_arg, aliases, name, cwd, extra=None, commit=None, resumable=False,
           keep_code=True):
    """Check a job against the host, write its run folder and start its supervisor.

    keep_code: record the Git commit at cwd and keep any uncommitted code in the run folder.
    """
    total = macos.total_memory()
    allowance = host.allowance(total)
    if allowance < MIN_BUDGET:
        raise SystemExit(f"{host.name} leaves jobs {format_size(max(allowance, 0))}: its reserve is "
                         f"{format_size(total - allowance)} of {format_size(total)}. Set a smaller reserve "
                         f"under [memory] in the configuration (README: Give a budget).")
    _, available = macos.memory_state()
    admissible = min(allowance, available - host.size("floor", total))
    try:
        budget = config.parse_size(budget_text, total) if budget_text else admissible
    except ValueError as error:
        raise SystemExit(str(error))
    if budget < MIN_BUDGET:
        raise SystemExit(f"Budget {format_size(budget)} is below the minimum {format_size(MIN_BUDGET)}"
                         + ("" if budget_text else "; this host has too little memory available now"))
    if budget > allowance:
        raise SystemExit(f"Budget {format_size(budget)} exceeds {host.name}'s allowance "
                         f"{format_size(allowance)} (memory minus reserve)")
    env_name = None if env_arg == "none" else env_arg or host.env
    if env_name:
        env_dir = host.root / "envs" / env_name
        if not (env_dir / "bin" / "python").exists() or not list(env_dir.glob("lib/python*/site-packages/modellab_hook.pth")):
            raise SystemExit(f"Environment {env_name!r} is not installed on {host.name}. "
                             f"Run: modellab env sync {env_name}")
    chosen = {}
    for alias in aliases:
        entry = models.get(alias)
        if (folder := models.path(host, alias)) is None:
            raise SystemExit(f"Model {alias!r} is not pulled on {host.name}. Run: modellab models pull {alias}")
        chosen[alias] = {**entry, "path": folder}
    (host.root / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=host.root / "tmp") as staging:
        # Taken before the run folder exists, so a failure here leaves no half-made run behind.
        keeps_code = keep_code and commit is None  # a remote run's --commit comes from the controlling Mac
        code = source.snapshot(cwd, Path(staging) / "code", command) if keeps_code else None
        if keeps_code:
            commit = source.commit(cwd)
        path = runs.new_run(host.root, {
            "host": host.to_json(), "name": name, "command": command, "cwd": str(cwd), "env": env_name,
            "budget": budget, "resumable": resumable, "models": chosen, "commit": commit, "keeps_code": keeps_code,
            "modellab": revision(), **(extra or {})})
        if code:
            shutil.move(Path(staging) / "code", path / "code")
    # Hold the run's lock from the start and hand it to the supervisor, so the run is live throughout.
    lock = os.open(path / "lock", os.O_WRONLY | os.O_CREAT, 0o644)
    fcntl.flock(lock, fcntl.LOCK_EX)
    with open(path / "supervisor.log", "ab") as log:
        subprocess.Popen([sys.executable, str(HOME / "bin" / "modellab"), "_supervise", str(path), str(lock)],
                         cwd=path, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         pass_fds=(lock,), start_new_session=True)
    os.close(lock)
    return path


def report_start(path):
    """Print the run id alone on standard output, so `id=$(modellab run ...)` works, and the
    first state on standard error."""
    print(path.name, flush=True)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = runs.read_json(path / "status.json")
        state = status["state"]
        if state == "error":
            raise SystemExit(f"error: {_last_line(status['error'])} (see {path / 'status.json'})")
        if state != "queued" or status.get("reason") != "starting":
            print(state + (f": {status['reason']}" if status.get("reason") else ""), file=sys.stderr)
            return state
        time.sleep(0.1)
    print(f"queued; the supervisor has not reported yet (see {path / 'supervisor.log'})", file=sys.stderr)
    return "queued"


SERVERS = {"mlx-lm": "mlx_lm.server", "mlx-vlm": "mlx_vlm.server"}


def cmd_serve(args):
    """Serve a pulled model on 127.0.0.1 as a guarded job."""
    host = host_named(args.host)
    if host.ssh:
        remote.push_code(host)
        result = forward(host, args, capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        for line in result.stdout.splitlines():
            if line.startswith("ready: "):
                port = line.rsplit(":", 1)[1].split("/")[0]
                print(f"On this Mac: ssh -N -L {port}:127.0.0.1:{port} {host.ssh}")
        return result.returncode
    entry = models.get(args.alias)
    folder = models.path(host, args.alias)
    if folder is None:
        raise SystemExit(f"Model {args.alias!r} is not pulled on {host.name}. Run: modellab models pull {args.alias}")
    port = args.port or free_port()
    command = ["python", "-m", SERVERS[entry["kind"]], "--model", folder, "--host", "127.0.0.1", "--port", str(port)]
    if entry["kind"] == "mlx-vlm":
        command += ["--max-num-seqs", str(args.max_num_seqs), "--top-logprobs-k", str(args.top_logprobs_k)]
    if args.adapter:
        command += ["--adapter-path", str(Path(args.adapter).resolve())]
    url = f"http://127.0.0.1:{port}/v1"
    path = submit(host, command, args.budget, None, [args.alias], f"serve {args.alias}", HOME,
                  extra={"serve": {"model": args.alias, "url": url}}, keep_code=False)
    if report_start(path) != "running":
        return
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline and runs.state(path)[0] == "running":
        try:
            urllib.request.urlopen(f"{url}/models", timeout=2).close()
            print(f"ready: {url}")
            return
        except OSError:
            time.sleep(0.5)
    state = runs.state(path)[0]
    print(f"{state}: {url} is not answering yet (see modellab logs {path.name})")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _last_line(text):
    return text.strip().splitlines()[-1] if text.strip() else ""


def cmd_ps(args):
    host = host_named(args.host)
    if host.ssh:
        return forward_reading(host, args)
    paths = runs.all_runs(host.root)
    if not args.all:  # a long-lived job, such as a server, never drops off the list
        paths = [path for path in paths[:-20] if runs.state(path)[0] in LIVE] + paths[-20:]
    if short := _disk_short(host):
        print(short, file=sys.stderr if args.json else sys.stdout)
    if not args.json:
        print(PS_HEADER)
    for path in paths:
        record = run_record(path)
        print(json.dumps(record) if args.json else _ps_row(record))


LIVE = ("queued", "running", "unguarded")  # states in which a run can still change
PS_HEADER = f"{'RUN':<21} {'STATE':<8} {'BUDGET':>7} {'NOW':>7} {'PEAK':>7} {'TIME':>7}  NAME / NOTE"


def run_record(path):
    """A run as `ps --json` and `wait --json` give it: sizes in bytes, times in seconds since
    the epoch, and null for what does not apply. `serve` is {"model", "url"} for a server."""
    spec = runs.read_json(path / "run.json")
    state, status = runs.state(path)
    live = runs.read_json(path / "live.json", {}) if state in ("running", "unguarded") else {}
    loaded = runs.libraries(path)
    return {"id": path.name, "name": spec["name"], "state": state, "exit_code": status.get("exit_code"),
            "reason": status.get("reason"), "guard_action": status.get("guard_action"),
            "error": status.get("error"), "budget": spec["budget"], "footprint": live.get("footprint"),
            "peak": status.get("peak"), "created": spec["created"], "started": status.get("started"),
            "ended": status.get("ended"), "resumable": bool(spec.get("resumable")),
            "yields": len(status.get("stops", [])), "dir": str(path),
            "serve": {"model": spec["serve"]["model"], "url": spec["serve"]["url"]} if "serve" in spec else None,
            "outside_env": sorted({entry["module"] for entry in loaded if entry["inside_env"] is False}),
            "unreadable_libraries": any(entry["inside_env"] is None for entry in loaded)}


def _ps_row(record):
    now = format_size(record["footprint"]) if record["footprint"] is not None else "-"
    peak = format_size(record["peak"]) if record["peak"] is not None else "-"
    started = record["started"]
    seconds = (record["ended"] or time.time()) - started if started else 0
    note = record["guard_action"] or record["reason"] or _last_line(record["error"] or "")
    if record["state"] == "running" and record["serve"]:
        note = record["serve"]["url"]
    if record["exit_code"] is not None:
        note = f"exit {record['exit_code']}" + (f"; {note}" if note else "")
    if record["yields"]:
        note += f"; yielded {record['yields']}x"
    if record["outside_env"]:
        note += f"; {', '.join(record['outside_env'])} loaded from outside the environment"
    if record["unreadable_libraries"]:
        note += "; libraries.jsonl has unreadable lines"
    note = note.lstrip("; ")
    return (f"{record['id']:<21} {record['state']:<8} {format_size(record['budget']):>7} {now:>7} {peak:>7} "
            f"{_duration(seconds):>7}  {record['name']}" + (f" / {note}" if note else ""))


def cmd_wait(args):
    """Wait until each run has ended. A requeued resumable job has not ended. Stopping `wait`
    never touches the runs."""
    try:  # an unknown host or run is a mistake in the command, not a run that failed
        host = host_named(args.host)
        waiting = {} if host.ssh else {run_id: runs.find(host.root, run_id) for run_id in args.run_ids}
    except SystemExit as error:
        print(error, file=sys.stderr)
        return USAGE
    if host.ssh:  # bounded here too: a link that drops silently must not outlast --timeout
        return forward_reading(host, args, None if args.timeout is None else args.timeout + REMOTE_MARGIN)
    deadline = None if args.timeout is None else time.monotonic() + args.timeout
    failed, shown = False, False
    while True:
        for run_id, path in list(waiting.items()):
            if runs.state(path)[0] in LIVE:
                continue
            record = run_record(path)
            if not args.json and not shown:
                print(PS_HEADER)
                shown = True
            print(json.dumps(record) if args.json else _ps_row(record), flush=True)
            failed |= (record["state"], record["exit_code"]) != ("exited", 0)
            del waiting[run_id]
        if not waiting:
            return 1 if failed else 0
        if deadline is not None and time.monotonic() >= deadline:
            print(f"Still active after {args.timeout:g} s: {' '.join(waiting)}", file=sys.stderr)
            return TIMED_OUT
        time.sleep(1 if deadline is None else max(0, min(1, deadline - time.monotonic())))


def forward_reading(host, args, timeout=None):
    """Forward a command that reads run records, and say what an SSH failure, or a host whose
    ModelLab code is older than the command, means."""
    try:
        result = forward(host, args, stderr=subprocess.PIPE, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as expired:  # what ssh said so far, such as why it cannot connect
        sys.stderr.write((expired.stderr or b"").decode(errors="replace"))
        print(f"No answer from {host.name} within {timeout:g} s; runs there are not affected.", file=sys.stderr)
        return TIMED_OUT
    sys.stderr.write(result.stderr)
    if result.returncode == SSH_FAILED:
        print(f"SSH to {host.name} failed or dropped; runs there are not affected. Try the same command again.",
              file=sys.stderr)
    elif "invalid choice" in result.stderr or "unrecognized arguments" in result.stderr:
        print(f"{host.name} has ModelLab code from before this command; `modellab host init {host.name}` "
              "updates it.", file=sys.stderr)
    return result.returncode


def _disk_short(host):
    """A line for `ps` while the swap volume is below the disk floor, as the guard reads it."""
    volume, floor = macos.swap_volume(), disk_floor(host)
    try:
        free = macos.disk_free(volume)
    except OSError as error:
        return f"Cannot read free space on {volume}: {type(error).__name__}: {error}"
    if free >= floor:
        return None
    return (f"Disk short: {format_size(free)} free on {volume}, floor {format_size(floor)}. New jobs wait; "
            f"a running job is stopped only if it is filling the disk.")


def _duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def cmd_logs(args):
    host = host_named(args.host)
    if host.ssh:
        return forward(host, args).returncode
    path = runs.find(host.root, args.run_id)
    log = path / "log"
    position = 0
    while True:
        if log.exists():
            with open(log, "rb") as handle:
                handle.seek(position)
                chunk = handle.read()
                position += len(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
        if not args.follow or runs.state(path)[0] not in runs.ACTIVE:
            return
        time.sleep(0.5)


def cmd_stop(args):
    host = host_named(args.host)
    if host.ssh:
        return forward(host, args).returncode
    path = runs.find(host.root, args.run_id)
    state, status = runs.state(path)
    if state == "unguarded":
        raise SystemExit(f"{args.run_id} has no supervisor, so nothing can act on a stop request. "
                         f"Its job's process group is {status['job_pid']}: kill -TERM -{status['job_pid']}")
    if state not in runs.ACTIVE:
        raise SystemExit(f"{args.run_id} is not active ({state})")
    (path / "stop").touch()
    print(f"stop requested for {args.run_id}")


def cmd_env_sync(args):
    host = host_named(args.host)
    if host.ssh:
        return remote.env_sync(host, local_host(), args.name)
    print(env.sync(host, args.name))


def cmd_models_add(args):
    alias, entry, old = models.add(local_host(), args.spec, args.alias, args.replace)
    was = f" (was {old['repo']}@{old['revision'][:10]})" if old and old != entry else ""
    print(f"{alias}: {entry['repo']}@{entry['revision']} ({entry['kind']}, {entry['quantization']}){was}")


def cmd_models_pull(args):
    host = host_named(args.host)
    aliases = args.aliases or list(models.load())
    if not aliases:
        raise SystemExit("No models pinned yet. Pin one with: modellab models add <org/name>")
    if host.ssh:
        for alias, target in remote.models_pull(host, local_host(), aliases).items():
            print(f"{alias}: {target} on {host.name}")
        return
    for alias in aliases:
        result = models.pull(host, alias)
        print(f"{alias}: {result['checked']} files verified in {result['path']}")


def cmd_models_path(args):
    host = host_named(args.host)
    if host.ssh:
        remote.push_lock(host)
        return forward(host, args).returncode
    folder = models.path(host, args.alias)
    if folder is None:
        raise SystemExit(f"{args.alias} is not pulled. Run: modellab models pull {args.alias}")
    print(folder)


def cmd_models_list(args):
    host = host_named(args.host)
    if host.ssh:
        remote.push_lock(host)
        return forward(host, args).returncode
    pinned = models.load()
    if not pinned:
        print(f"No models pinned in {models.lock_path()}. Pin one with: modellab models add <org/name>")
    for alias, entry in pinned.items():
        pulled = "pulled" if models.path(host, alias) else "not pulled"
        print(f"{alias:<24} {entry['repo']}@{entry['revision'][:10]}  {entry['kind']}, "
              f"{entry['quantization']}  [{pulled}]")


RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")


def cmd_fetch(args):
    if not RUN_ID.fullmatch(args.run_id):
        raise SystemExit(f"Not a run id: {args.run_id!r}")
    host = host_named(args.host)
    if not host.ssh:
        raise SystemExit(f"{args.run_id} is on this Mac already: {runs.find(host.root, args.run_id)}")
    # Under ModelLab's own folder, never the current one: that may be a repository's checkout.
    folder = remote.fetch(host, args.run_id, args.to or local_host().root / "fetched" / args.host / args.run_id)
    print(folder)
    out = folder / "out"
    if not out.is_dir() or not any(out.iterdir()):
        cwd = json.loads((folder / "run.json").read_text()).get("cwd")
        print(f"out/ is empty: the job wrote no results to $MODELLAB_RUN_DIR/out, which is all fetch copies. "
              f"Anything it wrote elsewhere is still on {args.host}, such as in its working folder {cwd}.", file=sys.stderr)


def cmd_gold_check(args):
    problems = gold.check(args.set, args.outputs)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"{args.set}: no problems")


def cmd_gold_freeze(args):
    try:
        version = gold.freeze(args.set)
    except gold.Invalid as error:
        raise SystemExit(f"Not frozen: {error}")
    print(f"{args.set}@{version}")


def cmd_look(args):
    try:
        width, height = (int(side) for side in args.size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size {args.size!r}: give WIDTHxHEIGHT, e.g. 1600x1600")
    if min(width, height) < 400 or args.sample < 1:
        raise SystemExit("--size needs at least 400 px a side, and --sample at least 1")
    folder, _, version = args.set.partition("@")
    to = Path(args.to or f"look-{Path(folder).name}-{time.strftime('%Y%m%d-%H%M%S')}").resolve()
    if to.exists():
        raise SystemExit(f"{to} exists; give a new folder with --to")
    request = {"set": str(Path(folder).resolve()) + (f"@{version}" if version else ""),
               "outputs": [str(Path(path).resolve()) for path in args.outputs], "sample": args.sample,
               "seed": args.seed if args.seed is not None else random.randrange(2**31), "lowest": args.lowest,
               "highest": args.highest, "size": [width, height], "to": str(to), "blind": args.blind}
    result = in_env(local_host(), "modellab.look", request)
    print(f"{result['items']} items in {result['to']}: {', '.join(result['sheets']) or 'no image panes'}; sheets.md")
    if "key" in result:
        print(f"Key (keep it from the judge): {result['key']}")


def cmd_report(args):
    from . import report
    log_path = Path(args.log).resolve()
    to = Path(args.to or f"report-{log_path.stem}").resolve()
    if to.exists():
        raise SystemExit(f"{to} exists; give a new folder with --to")
    if args.sample < 1:
        raise SystemExit("--sample needs at least 1")
    folder, files = gold.open_set(args.set)
    known = {item["id"]: item for item in gold.read_items(files / "items.jsonl")}
    if clash := sorted({field for item in known.values() for field in item} & set(report.FIELDS)):
        raise SystemExit(f"The gold set has fields {clash}, which the report's outputs would replace")
    host = local_host()
    log = in_env(host, "modellab.inspect_log", {"log": str(log_path)}, env="eval")
    if not log["rows"]:
        raise SystemExit(f"{log_path} holds no items (was it saved with --no-log-samples?)")
    if unknown := sorted({row["id"] for row in log["rows"]} - set(known)):
        raise SystemExit(f"{len(unknown)} items of the log are not in {args.set}, e.g. {unknown[:5]}. "
                         "Give each Sample the gold set's item id (Sample(id=...)).")
    scorer = args.scorer or next(iter(log["scorers"]), None)
    if scorer is None:
        raise SystemExit(f"{log_path} has no scores to report")
    items, failed = report.per_item(log["rows"], scorer)
    if not items:
        raise SystemExit(f"No item of {log_path} was scored; {len(failed)} failed to run")
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    figures = report.numbers(items, seed)
    recorded = str(folder.resolve()) + (f"@{files.name}" if files != folder else "")
    shown = min(args.sample, len(items))
    sheets = [("sample", {}, f"a random sample of {shown} of the {len(items)} items scored"),
              ("lowest", {"lowest": "eval_score"}, f"the {shown} items that scored lowest, misses first")]
    partial = to.with_name(f".{to.name}.partial-{os.getpid()}")  # renamed only once the report is whole
    partial.mkdir(parents=True)
    try:
        outputs = report.outputs(partial / "outputs.jsonl", items)
        common = {"set": recorded, "outputs": [str(outputs)], "ids": [item["id"] for item in items],
                  "sample": args.sample, "seed": seed, "lowest": None, "highest": None, "size": [1600, 1600],
                  "blind": False}
        for name, extra, _ in sheets:
            in_env(host, "modellab.look", {**common, **extra, "to": str(partial / name)})
        report.write(partial, {**log, "path": str(log_path), "set": recorded}, scorer, figures, failed,
                     [(name, how) for name, _, how in sheets])
        os.replace(partial, to)
        for index in to.glob("*/sheets.md"):  # they name the outputs file by its path, which moved
            index.write_text(index.read_text().replace(str(partial), str(to)))
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    verdict = {True: "better than trivial", False: "NO BETTER THAN TRIVIAL",
               None: "no constant-answer baseline"}[figures["better_than_trivial"]]
    print(f"{to / 'report.md'}: {figures['result']:.3f} over {figures['n']} items, {verdict}"
          + (f"; {len(failed)} failed to run" if failed else ""))


def cmd_colour(args):
    if len(args.pictures) % 2:
        raise SystemExit("Give pictures in pairs: FIRST SECOND [FIRST SECOND ...]")
    pictures = [str(Path(path).resolve()) for path in args.pictures]
    reports = in_env(local_host(), "modellab.colour", {"pairs": [pictures[i:i + 2] for i in range(0, len(pictures), 2)]})
    for report in reports:
        if args.json:
            print(json.dumps(report))
        elif "error" in report:
            print(f"{report['first']} / {report['second']}: {report['error']}")
        else:
            print(f"{report['first']} / {report['second']}: CIEDE2000 mean {report['mean']:.2f}, "
                  f"95th percentile {report['p95']:.2f} (CIE76 mean {report['cie76_mean']:.2f})")
    return 1 if any("error" in report for report in reports) else 0


def in_env(host, module, request, env=None):
    """Run a ModelLab module in an environment (the host's, unless named) with one JSON request;
    its last output line is the JSON result. Errors go to the terminal."""
    env = env or host.env
    python = host.root / "envs" / env / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"This command needs the {env!r} environment. Run: modellab env sync {env}")
    result = subprocess.run([python, "-m", module, json.dumps(request)], cwd=HOME, stdout=subprocess.PIPE,
                            text=True, env={**os.environ, "PYTHONPATH": str(HOME)})
    if result.returncode != 0:
        raise SystemExit(f"{module} failed (see above)")
    return json.loads(result.stdout.strip().splitlines()[-1])


def cmd_label(args):
    from . import label
    if "@" in args.set:
        raise SystemExit("A frozen version does not change; label the set's folder, then freeze it again")
    if not gold.ID.fullmatch(args.labeller):
        raise SystemExit(f"--labeller {args.labeller!r}: use letters, digits, '_', '.' or '-'")

    def ready(url):
        print(f"Labelling {args.set} as {args.labeller} at {url} (Ctrl-C to stop)", flush=True)
        if not args.no_open:
            subprocess.run(["open", url], check=False)
    try:
        label.serve(args.set, args.labeller, args.port, ready)
    except gold.Invalid as error:
        raise SystemExit(f"Cannot label {args.set}: {error}")
    except KeyboardInterrupt:
        print("stopped")


def cmd_host_init(args):
    host = host_named(args.name)
    report = remote.init(host, local_host()) if host.ssh else probe.run(host.root, disk_floor(host))
    for check, result in report.items():
        print(f"{'ok ' if result['ok'] else 'NO '} {check}: {result['detail']}")
    if not all(result["ok"] for result in report.values()):
        return 1


def cmd_probe(args):
    host = local_host()
    print(json.dumps(probe.run(host.root, disk_floor(host))))


def disk_floor(host):
    return host.size("disk_floor", macos.total_memory())


def cmd_record_pulled(args):
    models.record(local_host(), args.alias, args.path, args.how)


def cmd_supervise(args):
    from . import supervisor
    supervisor.main(args.path, args.lock)
