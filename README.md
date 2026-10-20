# ModelLab

A small, forkable workbench for local model work on Apple Silicon Macs: run
models, collect labelled judgments, evaluate, and fine-tune, on a Mac that
stays usable for everything else. It is built on maintained upstream tools
(MLX, MLX-LM, MLX-VLM, Inspect AI, uv). Everything specific to one machine,
model or project is configuration.

## What it does

- **Jobs guarded so they do not take the Mac down.** `modellab run` starts a
  job only when its memory budget fits, watches its footprint (GPU memory
  included) from outside, and caps MLX's memory and wired limits from inside,
  which MLX-LM, MLX-VLM and mlx-lm-lora otherwise raise to the maximum. When memory or disk
  runs short, ModelLab stops its own jobs, never other work. What the guard
  cannot see is listed in `docs/machines-and-models.md`.
- **Jobs that yield and come back.** A `--resumable` job steps aside for other
  jobs and for memory pressure, and restarts from its checkpoints when there is
  room. The training recipe resumes exactly: adapter, optimizer state, data
  position and random numbers, so a stopped run ends with the same weights, to
  the last bit, as one never stopped.
- **Jobs that agents can watch.** Jobs run detached. `modellab wait` returns
  when they end, with an exit status a script can trust, and `ps --json` gives
  the latest runs and every live one as records.
- **Comparable results.** One locked environment, models pinned to an exact
  Hub commit and verified, and run records that keep the command, the code and
  the MLX libraries actually loaded.
- **Gold sets, labelling and evaluation.** Versioned gold sets, a keyboard-first
  labelling page, sheets that show items beside their targets, and reports on
  Inspect AI logs that put the items before the numbers.
- **Serving.** A pinned model on a local OpenAI-compatible endpoint, as a
  guarded job.
- **A second Mac.** The same commands on another Mac over SSH, including one
  in a sandbox with no internet access.

## Install

Needs macOS on Apple Silicon, Python 3.11 or later, and
[uv](https://docs.astral.sh/uv/); `modellab env sync` installs the Python
3.14 its environments use. ModelLab runs from its checkout: clone it, then put
`bin/modellab` on your `PATH` (here through `~/.local/bin`, which must be on
`PATH`).

```sh
git clone https://github.com/Lumonopolis/modellab.git ~/src/modellab
mkdir -p ~/.local/bin && ln -s ~/src/modellab/bin/modellab ~/.local/bin/modellab
```

Keep the checkout apart from ModelLab's state folder (`~/modellab` by
default). Without a configuration file, ModelLab uses this Mac alone with the
defaults in `config.example.toml`; copy it to `~/.config/modellab/config.toml`
to change them or to add a second Mac.

## Use

```sh
modellab env sync                                    # install the mlx environment on this Mac
modellab models add mlx-community/SmolLM2-135M-Instruct-8bit --alias smol
modellab models pull smol                            # download and verify against the Hub
modellab run --budget 40G -- python train.py         # prints the run id
modellab run --resumable --budget 90G -- python train.py   # yields memory, restarts later
modellab run --model smol -- python -c 'import os; print(os.environ["MODELLAB_MODEL"])'
modellab wait <run-id>                               # until it ends; exit 0 only if it exited 0
modellab ps                                          # latest 20 and live runs; --json, --all
modellab logs <run-id> -f
modellab stop <run-id>
modellab serve smol --budget 4G                      # OpenAI-compatible, on 127.0.0.1
modellab gold check path/to/set                      # a gold set: task.toml, items.jsonl, media
modellab gold freeze path/to/set                     # keep it as a version: path/to/set@<id>
modellab look path/to/set --outputs run/out/outputs.jsonl   # sheets: items, outputs, targets
modellab look path/to/set --blind                    # a judge's view: no targets, shuffled panes
modellab label path/to/set                           # answer its questions in a page on this Mac
modellab colour render.jpg target.jpg                # how different two pictures look (CIEDE2000)
modellab report logs/x.eval --set path/to/set        # an Inspect AI log: items first, then numbers

# A second Mac, configured as host "lab2" (see config.example.toml)
modellab host init lab2                              # set up and report what the host allows
modellab env sync --host lab2                        # push packages; install offline if needed
modellab models pull smol --host lab2                # copy a verified model over the link
modellab run --host lab2 --sync ./job --model smol -- python job/train.py
modellab fetch <run-id> --host lab2                  # log, status and out/ back to this Mac
```

A training script uses the recipe from inside a job (`from modellab import
recipe`): a check that the sampler's prompts match what training saw, a timed
fit step, checkpoints, exact resume, a small run by default, and a look at the
replies when it ends.

## Documentation

- `docs/machines-and-models.md`: environments, models, jobs and the memory
  guard, serving, remote hosts.
- `docs/training.md`: the training recipe.
- `docs/gold-sets.md`: gold sets, sheets and the labelling page.
- `docs/evaluation.md`: evaluation reports.
- `docs/on-the-data.md`: why every view shows items beside their targets.
- `docs/stack.md`: each tool choice and its reason, known upstream limits, and
  patches.
- `AGENTS.md`: what ModelLab owns and its design rules, for people and agents
  changing it.

## Status

In use on two Macs (an M4 Max and an M5, 128 GB each) since late September
2026. Commands and records may still change.

Tests:

```sh
uv run --no-project --python 3.14 --with pytest --with numpy --with pillow python -m pytest tests
```

Tests that need an installed environment, a model, the Hub or a second Mac are
skipped unless `MODELLAB_TEST_MLX_ENV`, `MODELLAB_TEST_EVAL_ENV`,
`MODELLAB_TEST_VLM`, `MODELLAB_TEST_HUB` or `MODELLAB_TEST_REMOTE` is set.

## License

MIT. See `LICENSE`.
