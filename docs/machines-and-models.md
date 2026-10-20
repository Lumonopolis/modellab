# Machines and models layer

Status: built, and in use on two Macs (an M4 Max and an M5, 128 GB each) since
2026-09-29.

## Goal

A person or an agent on any configured Mac can:

1. run code in a known, locked Python environment;
2. resolve a model alias to exact files;
3. run a job on any configured machine: detached, logged, queued by memory, and
   unable to take the machine down;
4. serve a model on a local, OpenAI-compatible endpoint.

## Use

```sh
modellab run --host lab2 --budget 40G --sync scripts -- python scripts/train.py
modellab run --resumable --budget 90G -- python train.py --resume-from-latest
modellab ps --host lab2
modellab logs <run-id> --host lab2
modellab wait <run-id> --host lab2
modellab stop <run-id> --host lab2
modellab fetch <run-id> --host lab2
modellab serve qwen3-vl-8b-4bit --budget 12G
modellab models pull qwen3-vl-8b-4bit --host lab2
modellab env sync --host lab2
```

## Template and instance

| Where | Holds | Tracked |
|---|---|---|
| This repository | Code, environment definitions, docs, `config.example.toml` | Yes; forkable |
| `~/.config/modellab/config.toml` | Hosts, roots, memory policy, store paths | No |
| `~/.config/modellab/models.lock` | This installation's pinned models, next to its config | No |
| `~/.ssh/config` | Addresses, keys, host-key aliases | No |

Network details stay in SSH configuration. ModelLab refers to hosts by SSH alias.

```toml
[hosts.local]
root = "~/modellab"                 # all ModelLab state on this host
hf_home = "~/.cache/huggingface"

[hosts.lab2]
ssh = "<ssh-alias>"
root = "<remote-root>/modellab"
launcher = "<remote-root>/run"      # optional prefix, such as a sandbox
hf_home = "<remote-root>/modellab/hf"

[memory]                            # defaults; any host may override
reserve = "max(8G, 15%)"            # never given to jobs
floor = "10%"                       # minimum available memory to admit or keep running
requeues = 3                        # stops for short memory or disk before a resumable job is killed
disk_floor = "100%"                 # free disk to keep where swap lives, as a share of RAM
write_window = 1800                 # seconds over which a job's writing is judged when disk is short
```

## Hosts

- Local and remote hosts run the same code. This Mac pushes the ModelLab code,
  environment definitions, `models.lock`, a `uv` binary and the host's own
  config (`<root>/config.toml`, where the host is `local`) to `<root>` with
  `rsync`, then runs the host's own `modellab` over SSH. A remote host needs no
  network access beyond that SSH link.
- Nothing is written outside `root` except the model store (`hf_home`): no
  Homebrew packages, no launchd agents, no sysctl changes. Jobs get `TMPDIR`,
  `XDG_CACHE_HOME`, `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` inside `root`,
  and ModelLab's own code first on `PYTHONPATH`, for `modellab.recipe`
  (`docs/training.md`); `python -I` or `-E` ignores it.
- Remote commands arrive as one quoted command (`shlex.join`), because sshd and
  any forced command re-split them. A host `launcher` prefixes commands and
  `rsync --rsync-path`; a host that forces its own launcher needs none.
- Code never assumes a shell, `ps` or `top`.
- `modellab host init <host>` pushes everything and runs a probe on the host:
  memory and disk readings, a child's footprint and bytes written, the run tag,
  signals, file locks and a localhost port. It reports each result and changes no sandbox profile.
- An `offline` host installs environments from this Mac's package cache
  (`env sync --host` pushes it; `uv sync --offline` with the host's `python`).
- `models pull --host` copies only the pinned revision of a snapshot verified
  on this Mac, as plain files (`rsync --copy-links`), into
  `<hf_home>/hub/<repo>/snapshots/<commit>` on the host: the Hub cache can link
  blobs to a store shared across repositories. The host then records it as
  pulled. A remote host's `hf_home` defaults to `<root>/hf`, never to this
  Mac's paths.
- `run --host` gives each run a fresh `<root>/work/<tag>` folder, copies its
  `--sync` inputs there (unchanged files hard-link to the previous run's copy),
  and uses it as the default working folder; a relative `--cwd` is taken inside
  it. `run.json` records this Mac's commit and the ModelLab revision that ran.
  `ps`, `logs`, `wait`, `stop`, `serve`, `models path` and `models list` run
  on the host; `fetch` copies a run's folder back, to `<local root>/fetched/<host>/<id>`
  unless `--to` says otherwise (never into the current folder).
- Work folders accumulate; nothing prunes them yet.

Verified on 2026-09-29 with a second Mac (an M5, 128 GiB, reached over SSH
inside a sandbox with no internet access): every probe check passed, the
environment installed offline, a pulled model ran through `run --model` and
`fetch`, and a job growing on the GPU past its budget was killed.

## Environments

- `envs/` holds a `pyproject.toml` and a `uv.lock` for each environment.
  `mlx`: MLX, MLX-LM (git commit pin when needed), MLX-VLM,
  `huggingface_hub`, Pillow, OpenCV, and the ModelLab job hook (below).
  `eval`: Inspect AI, kept apart because it releases almost daily
  (`docs/evaluation.md`). Optional packages go in dependency groups.
- `modellab env sync [--host H]` runs `uv sync --frozen` with
  `UV_PROJECT_ENVIRONMENT=<root>/envs/mlx`, then applies the environment's reviewed patches
  (`envs/<env>/patches/`) to the exact versions they fix and records them for the job hook
  (`docs/stack.md`, Patches).
- A project that needs another package adds it here. One shared definition
  replaces per-project environments.

## Models

- `models.lock`, next to the config file, maps each alias to repository, commit, kind
  (`mlx-lm`, `mlx-vlm`, ...) and quantization. `MODELLAB_MODELS_LOCK` points
  elsewhere.
- `models add <repo>[@revision] --alias A` resolves the canonical repository
  name and the full commit, and records them. Aliases use lowercase letters,
  digits and `-`. Repointing an existing alias needs `--replace`.
- `models pull [A ...]` downloads with `snapshot_download` at that commit and
  checks the files against the Hub's recorded hashes (`verify_repo_checksums`).
  Only then does `<root>/pulled.json` record the snapshot. A partial download
  never counts as pulled. Remote hosts receive a local snapshot by `rsync`
  (above).
- `models path A` and `models list` read `pulled.json`; they need no network.
- `run --model A` passes the snapshot folder as `$MODELLAB_MODEL_<A>` (and
  `$MODELLAB_MODEL` when only one model is given) and records the alias,
  repository and commit in `run.json`. Jobs never inherit model variables from
  the caller. The folder belongs to the shared cache: jobs read it and must not
  save into it.
- Hub operations run inside the ModelLab environment, which has
  `huggingface_hub`. Jobs run with `HF_HUB_OFFLINE=1` and
  `HF_HUB_DISABLE_TELEMETRY=1`.

## Jobs and memory safety

`modellab run` writes `<root>/runs/<id>/run.json` (command, host, environment,
budget, whether the job is resumable, project Git commit, and the host settings
it runs under) and starts a detached supervisor without a shell. The default
budget is what could be admitted now: the smaller of the allowance
(RAM − reserve) and available memory minus the floor. Every guard stays on.

ModelLab jobs yield first. Apart from killing a job that passes its own budget,
the guard stops jobs only when memory or disk space is actually short, and only
ModelLab's own jobs. It never blocks or stops other work on the machine. Among
ModelLab jobs, resumable ones yield to the others.

A job started with `--resumable` promises to resume from its own checkpoints.
When it is stopped to give memory or disk back, it goes back in the queue and starts
again with the same command when there is room. Pausing would not help: an MLX
job keeps its GPU memory while paused.

### Admission

- `modellab run` takes a `flock` on the run folder and hands it to the
  supervisor, which passes it to the job. A crashed or restarted host leaves no
  stale entries: its locks are gone.
- Under `<root>/admit.lock`, a queued job starts when:
  - no job ahead of it is queued. Jobs start oldest first, except that no job
    waits for a resumable one, and a resumable job waits for every job that is
    not resumable;
  - no stop happened in the last `settle` seconds (`guard.json`), and
    pressure is not critical;
  - the volume that holds swap has at least `disk_floor` free (below);
  - a requeued job has waited `settle` × 2^(n−1) seconds (at most 10 minutes)
    since its n-th stop for short memory or disk;
  - its budget fits the allowance minus the budgets of runs whose locks are held;
  - its budget, plus what running jobs may still grow into, fits the memory
    available now (`kern.memorystatus_level`) minus the floor. This counts
    memory used outside ModelLab.
- A job that is not resumable and does not fit asks running resumable jobs to
  make room, largest budget first, but only if stopping them would be enough.
  They go back in the queue behind it. It asks at most once per `settle`
  seconds, because freed memory shows in the readings about a second late.
- `ps` shows why a job is waiting, and starts with a line while the disk is short.

### Inside the job

A `.pth` hook in the environment acts only when `MODELLAB_BUDGET` is set:

- When `mlx.core` loads, it sets the memory limit to the budget and the cache
  limit to the smaller of 4 GiB and a quarter of the budget. It wraps MLX's
  `set_memory_limit`, `set_cache_limit` and `set_wired_limit`, so later calls,
  including MLX-LM's own, are clamped to those values (wired: the budget).
- When `mlx.core`, `mlx_lm` or `mlx_vlm` loads, it appends the version and the
  file to `libraries.jsonl` in the run folder, marking whether the file lies
  inside the environment and naming any reviewed patch `env sync` applied to it.
  `ps` flags a run that loaded one from anywhere else, such as a private build.
- It watches a pipe from the supervisor. If the supervisor dies, the job kills
  its own process group.
- The hook works only for Python jobs in a ModelLab environment. Other jobs are
  still guarded from outside while their supervisor lives.

### Guard

Twice a second the supervisor reads the physical footprint (`proc_pid_rusage`)
of every live process of the job. MLX GPU buffers count in this footprint
(verified 2026-09-29 on an M4 Max, macOS 26.4.1, MLX 0.30.3, also with the wired
limit raised). It finds the job's processes three ways: its process group,
children of tracked processes, and every process of the user whose environment
carries `MODELLAB_RUN_ID=<run>`. The tag finds children that leave the group and
outlive their parent. macOS hides the environment of Apple platform binaries,
so only those can escape the tag.

- Footprint above budget × 1.1: kill the job's processes (`SIGKILL`). The job
  is not requeued, resumable or not.
- Memory short, meaning pressure critical or available memory below the floor:
  one supervisor acts, chosen by `flock` on `<root>/guard.lock`. It stops one
  job among those whose supervisors still report: the one furthest over its
  budget, otherwise a resumable job, otherwise any job; the largest first. It
  waits until that job has stopped (at most 30 s, or five polls at critical
  pressure), then records a settle time in `guard.json` that every supervisor
  respects before stopping another. The owner checks this on every poll,
  whatever happens to its own job, so the lock is never held for long.
- Disk short, meaning less than `disk_floor` free on the volume that holds
  swap (`/System/Volumes/VM`; the ModelLab root and the model store share its
  space unless they are on another disk). A full disk while memory is short can
  hang the machine; the default floor equals RAM, so swap can always hold all
  of memory. The same one supervisor stops one job: the one filling the disk
  fastest. Stopping a job frees none of the space its files hold, so only a
  job that is filling the disk is ever stopped. If none is, none is stopped,
  new jobs wait, and the supervisor logs it.
  - A job is filling the disk when it writes at least 1 MiB/s, both now (the
    last `settle` seconds, at least five polls) and on average over the last
    `write_window` seconds (30 minutes). With less than 84 GiB free, the bar
    falls to the rate that would fill the free space within a day.
  - The window spares one checkpoint save: a 192 MB save is 0.1 MB/s over 30
    minutes. A save counts only if it is at least 1.76 GiB (window × 1 MiB/s),
    or free ÷ 48 below 84 GiB free, and then only while it is written and for
    `settle` seconds after: once a job has stopped writing, stopping it would
    stop nothing. Steady writing, or checkpoints saved every few seconds,
    counts within seconds or minutes.
  - Each start, including a requeued job's restart, begins a new window.
  - The stop's reason says what the job wrote in its window.
  - On 2026-10-02 the earlier rule (the last 10 seconds only) stopped a
    13-hour training run one second after it saved a 192 MB adapter, while
    another program filled the disk.
  - Unlike memory stops, disk stops do not prefer resumable jobs: only the
    writer's stop helps. If memory runs short while a disk stop waits for its
    job, that stop counts as the memory stop too (the job frees its memory as
    well): its wait shrinks to five polls and the settle time follows.
  - Another disk is not watched: a full data disk breaks jobs, not the
    machine, and pruning keeps it in check.
- A job's bytes written come from `proc_pid_rusage` for each of its processes,
  including ones that have ended (up to their last reading). They miss
  processes that live less than one poll, and they count data later
  overwritten or deleted. The reading of free space leaves out purgeable
  space, such as local snapshots, so it errs low. An unreadable volume is
  logged and counts as not short.
- macOS's warn level alone stops nothing. On 2026-09-30 it fired with 43–61 GiB
  available and stopped within-budget jobs. Each supervisor logs every change of
  pressure level to `supervisor.log`.
- A stopped resumable job is requeued, unless it has already been stopped for
  short memory or disk more than `requeues` times: then it is killed.
- When the job's main process ends, kill whatever it left running.
- After every kill, confirm that no tracked process remains. Record the result.
  A resumable job whose processes survive `SIGKILL` is not requeued, so two
  copies never run.

`modellab stop` writes a request file that the supervisor reads. The supervisor
sends `SIGTERM`, then `SIGKILL` after a grace period. `SIGTERM` to a supervisor
kills its job. Supervisors only signal their own job's processes. A job whose
supervisor died shows as `unguarded` in `ps`.

`modellab wait RUN...` returns when each run has ended and prints its `ps` row
(`--json`: its record). A requeued resumable job has not ended. Stopping `wait`
never touches a run. So an agent starts a long job with `modellab run`, not as
a background task of its own, which a time limit or sleep can end, and waits
with `modellab wait`, again if that wait is cut off. Exit status:

- 0: every run exited 0.
- 1: a run ended otherwise: a non-zero exit, stopped, killed, `error` or
  `lost`. A job whose supervisor died (`unguarded`) ends as `lost`, without
  an exit code, even if the job itself succeeded.
- 2: a mistake in the command, such as an unknown run or host, or a remote
  host whose ModelLab code predates `wait` (`modellab host init <host>`
  updates it).
- 124: `--timeout` ran out first. On a remote host it also bounds a silent SSH
  link, with 30 seconds' grace.
- 255: SSH failed or dropped; the runs go on. How soon SSH notices a dropped
  link depends on its keepalive settings (`ServerAliveInterval`).

### Records

Status, exit code, peak footprint, bytes written, any guard action and every
stop of a resumable job (`stops`) go to `<root>/runs/<id>/status.json`. Output goes to
`log`. `ps` lists the latest 20 runs and every older run still queued, running or
`unguarded`, such as a server; `--all` lists every run. `ps --json` prints one
JSON object per line, one per run: sizes in bytes, times in seconds since the
epoch, `null` where a field does not apply. `started` is the first start, kept
across requeues; a negative `exit_code` is the signal that ended the job; `dir`
is the run's folder on its host; `error` is the full error text. The disk line
goes to standard error. Jobs write results to `$MODELLAB_RUN_DIR/out`, which
`fetch` copies back; when `out/` is empty, `fetch` says so and names the job's
working folder, where anything it wrote elsewhere stays.

- `commit` is the Git commit at the job's working folder, with `+dirty` when
  anything under that folder is uncommitted. The run then keeps the uncommitted
  code in `code/`: `diff.patch` for tracked files and `untracked/` for files Git
  does not ignore, listed in `manifest.json`. Files the command names are copied
  first, then the files nearest to them. A file over 1 MiB, or anything past
  16 MiB in all (diff included), is listed but not copied; an unreadable file is
  listed with the error. Files Git ignores are never copied; an untracked file
  that Git does not ignore can be, so keep data and secrets ignored. A remote
  run needs none of this: its `--sync` inputs stay in its own work folder on
  the host.
- Each restart of a resumable job records the commit it starts from in
  `restarts`, and any uncommitted code in `code-<n>/`: a restart runs whatever
  code the working folder holds by then. If that record fails, the job still
  restarts and the entry holds the error.
- A binary change is listed in the manifest but not kept.
- `modellab` is the last commit that changed ModelLab's code, with `+dirty` when
  that code differs from it.
- `libraries.jsonl` records the MLX libraries the job actually loaded (above).

## Serving

`modellab serve A` is a guarded job that runs `mlx_lm.server` or
`mlx_vlm.server` according to the model's kind. It always passes
`--host 127.0.0.1`; vision servers also get `--max-num-seqs` and
`--top-logprobs-k`. The port comes from `--port` or the operating system. It
waits until the server answers and prints its URL; `ps` shows the URL while it
runs, and `stop` ends it. A program finds a server by its alias in `ps --json`:
the record whose `state` is `running` and whose `serve.model` is the alias has
the address in `serve.url`. Queued and ended servers keep `serve` too.
`--adapter` loads a LoRA adapter. For a remote host, forward the port with
`ssh -N -L <port>:127.0.0.1:<port> <host>`.

## Build order

Each step is one reviewed change.

1. Configuration, supervisor, admission, guard, job hook, local `env sync`;
   `run`, `ps`, `logs`, `stop`. Built.
2. Models: `models add`, `pull`, `path`, `list`; `run --model`. Built.
3. Serving and remote hosts: `serve`, `host init`, remote `env sync`,
   `models pull --host`, `run --host --sync`, `fetch`, `--host` for every
   command. Built.
4. Watching runs from programs and agents: `wait`, `ps --json`, and older live
   runs kept on `ps`. Built.

## Verification

- Unit: admission arithmetic, queue order around resumable jobs, making room,
  locks under concurrent processes, guard decisions with injected readings
  (warn with memory to spare stops nothing; critical stops only the chosen job;
  short disk stops the fastest writer and never an idle job; a replay of
  2026-10-02 shows one checkpoint save no longer counts, while steady writing
  and stacked checkpoints still do),
  descendant tracking, configuration parsing.
- Local integration:
  - an MLX job allocating on the GPU past a 1 GiB budget is killed near
    1.1 GiB, and no process remains;
  - a job whose child calls `setsid` is still tracked and killed;
  - MLX-LM's own `set_wired_limit` call is clamped;
  - two jobs that do not fit together run one after the other;
  - a resumable job stopped for memory is requeued and finishes on its second
    start; after `requeues` it is killed; a user stop ends it for good;
  - a job that does not fit makes a running resumable job yield, runs first,
    and the resumable job then finishes;
  - the hook records a library loaded from outside the environment;
  - a job's bytes written include those of a child that has already ended;
    a job writing steadily is stopped when disk runs short, one that saves a
    checkpoint once runs on, and a job waits to start while disk is short;
    `ps` says when it is;
  - a run from a dirty checkout keeps its diff and untracked code, and skips
    ignored and oversized files;
  - killing the supervisor makes the job exit;
  - `wait` prints a row for each run and fails if one did, counts a requeued
    job as still running, leaves a run alone at its timeout, and exits 2 for an
    unknown run or host; through a stand-in `ssh`, it explains status 255,
    names `host init` for older code, and ends at its timeout on a silent link;
  - `ps` keeps an older live run listed and gives JSON records; a fixed record
    pins the row text that programs read for a server's URL.
- Not tested on a real machine: critical pressure. Driving a Mac that is in use
  into critical pressure risks the crash the guard exists to prevent; the
  decision is tested with injected readings instead.
- Remote integration, on the M5: `host init` capability report, a remote job
  returned with `fetch` (`MODELLAB_TEST_REMOTE=<host>`), and the guard killing
  a GPU job there.
- Proof by use: 102 runs of one research project's training, sampling and
  evaluation on one Mac, 2026-10-02 to 10-19: 92 exited cleanly, 9 were stopped
  on purpose, 1 was killed for passing its budget, and resumable jobs were
  requeued 9 times to make room for others.
- An independent code review before each change.
