# ModelLab

Small forkable workbench for local model work on Apple Silicon Macs.

If you use agents to drive a variety of small local model experiments,
sometimes in parallel, on a Mac that also handles other types of occasionally
compute-intensive work, this may be useful. Built on maintained upstream tools
(MLX, MLX-LM, MLX-VLM, Inspect AI, uv).

## How it helps

- **Crash guard:** `modellab run` starts a job only when its memory budget fits,
  watches its footprint (GPU memory included) from outside, and caps MLX's memory
  and wired limits from inside, which MLX-LM, MLX-VLM and mlx-lm-lora otherwise
  raise to the maximum. When memory or disk runs short, ModelLab stops its own jobs,
  never other work. Its limits are in `docs/machines-and-models.md`, Limits.
- **Flexible, resumable jobs:** A `--resumable` job steps aside for other
  jobs and for memory pressure, and restarts from its checkpoints when there is
  room. The training recipe for vision-language models resumes exactly:
  adapter, optimizer state, data position and random numbers, so a stopped run
  ends with the same weights, to the last bit, as one never stopped.
- **Agent visibility:** Jobs run detached. `modellab wait` returns
  when they end, with an exit status a script can trust, and `ps --json` gives
  the latest runs and every live one as records.
- **Traceable results:** Each model name points to one exact version of its files on
  Hugging Face, so it cannot change under you, and each run records its command, its
  code version and the MLX libraries it loaded.
- **Evals and labelling tools:** A keyboard-first labelling page, gold set versioning,
  sheets that show items beside their targets, and reports on Inspect AI logs.
- **Remote Macs:** Manage jobs with the same commands on another Mac connected over SSH,
  including one in a sandbox with no internet access.

> [!IMPORTANT]
> **Remind agents to run training through ModelLab, not directly.** The guard
> protects only jobs started with `modellab run` or `modellab serve`; a script
> an agent runs on its own gets no protection and can still freeze the Mac. For
> example, in your project's `AGENTS.md` or `CLAUDE.md`:
>
> ```text
> Start every MLX job through ModelLab, never directly: `modellab run --budget <size> -- <command>`
> for training and evaluation, `modellab serve <model>` for a local endpoint. Wait for a job
> with `modellab wait <run-id>`; list jobs with `modellab ps --json`.
> ```

## Install

Needs macOS on Apple Silicon, Python 3.11 or later, and
[uv](https://docs.astral.sh/uv/) (`brew install uv`); `modellab env sync`
installs the Python 3.14 its environments use. ModelLab runs from its
checkout: clone it, then put `bin/modellab` on your `PATH`.

```sh
git clone https://github.com/Lumonopolis/modellab.git ~/src/modellab
mkdir -p ~/.local/bin && ln -s ~/src/modellab/bin/modellab ~/.local/bin/modellab
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zprofile ;; esac
```

The last line adds `~/.local/bin` to your `PATH` if it is not there yet; open a
new Terminal window after it.

ModelLab keeps its state (runs, environments, caches) in `~/modellab`, apart
from the checkout, and models in the Hugging Face cache
(`~/.cache/huggingface`). Without a configuration file it uses this Mac alone
with the defaults in `config.example.toml`. To change them, or to add a second
Mac, copy it: `mkdir -p ~/.config/modellab && cp config.example.toml
~/.config/modellab/config.toml`.

## Commands

```sh
modellab env sync                                    # once: install the locked MLX environment
modellab models add mlx-community/SmolLM2-135M-Instruct-8bit --alias smol   # pin its current version on Hugging Face
modellab models pull smol                            # download and verify
modellab models list                                 # pinned models and whether they are pulled
modellab run --name train --budget 8G -- python train.py   # prints the run id
modellab run --resumable --budget 8G -- python train.py    # yields memory, restarts later
modellab run --model smol -- python -c 'import os; print(os.environ["MODELLAB_MODEL"])'
modellab wait <run-id>                               # until it ends; exit 0 only if it exited 0
modellab ps                                          # latest 20 and live runs; --json, --all
modellab logs <run-id> -f
modellab stop <run-id>
modellab serve smol --budget 2G                      # OpenAI-compatible, on 127.0.0.1
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

A training script for a vision-language model uses the recipe from inside a
job (`from modellab import recipe`): a check that the sampler's prompts match
what training saw, a timed fit step, checkpoints, exact resume, a small run by
default, and a look at the replies when it ends. Text models train with
MLX-LM's own trainer, as in the demo below, which resumes from the adapter's
weights only.

Two rules for every job:

- **Give a budget.** `--budget` is the memory the job may use, GPU memory
  included. A job waits until that much is free, and is stopped, not the Mac,
  if it uses more: start low, look at its peak, and raise the budget if it was
  stopped. Two numbers show the peak: `PEAK` in `modellab ps`, the whole job as
  the guard saw it, and the peak memory in MLX-LM's log; go by the larger.
  Without a budget, a job may take all the memory ModelLab can give, and other
  jobs wait. ModelLab gives jobs at most the Mac's memory minus a reserve (by
  default the larger of 8 GB and 15%): about 8 GB on a 16 GB Mac, 24 GB on
  32 GB, 54 GB on 64 GB, 109 GB on 128 GB. The demo below peaks under 1 GB, so
  `--budget 1G` works where 4G is refused. An 8 GB Mac has nothing to give
  with the default reserve; `reserve = "5G"` under `[memory]` in your
  configuration gives jobs 3 GB and leaves less for everything else.
- **Variables that exist only inside the job, such as `$MODELLAB_MODEL`, belong
  in your script, not on the command line.** Your shell expands `$` before
  ModelLab sees the command, and ModelLab runs the command without a shell. The
  examples read them with `os.environ`.

## Documentation

- `docs/machines-and-models.md`: environments, models, jobs and the memory
  guard, serving, remote hosts.
- `docs/training.md`: the training recipe for vision-language models.
- `docs/gold-sets.md`: gold sets, sheets and the labelling page.
- `docs/evaluation.md`: evaluation reports.
- `docs/on-the-data.md`: why every view shows items beside their targets.
- `docs/stack.md`: each tool choice and its reason, known upstream limits, and
  patches.

Tests:

```sh
uv run --no-project --python 3.14 --with pytest --with numpy --with pillow python -m pytest tests
```

Tests that need an installed environment, a model, Hugging Face or a second Mac are
skipped unless `MODELLAB_TEST_MLX_ENV`, `MODELLAB_TEST_EVAL_ENV`,
`MODELLAB_TEST_VLM`, `MODELLAB_TEST_HUB` or `MODELLAB_TEST_REMOTE` is set.

## Beginner demo (tiny test run)

This trains a LoRA adapter (a small add-on to a model's weights) for a
135M-parameter text model on ten questions about a made-up café, then asks it
one. It takes a few minutes the first time, most of it downloading, and less
than 1 GB of memory.

```sh
cd ~/src/modellab
modellab env sync                                    # once: install the MLX environment (about 1 GB)
modellab models add mlx-community/SmolLM2-135M-Instruct-8bit --alias smol
modellab models pull smol                            # download and verify; no Hugging Face token needed
modellab run --name first-finetune --budget 4G --model smol -- python examples/finetune_text.py
modellab wait <run-id>                               # the id the last command printed
modellab logs <run-id>                               # MLX-LM's report, and where the adapter went
modellab run --budget 2G --model smol -- python examples/ask.py "Where is the Copper Kettle?" \
    ~/modellab/runs/<run-id>/out/adapters
modellab wait <ask-id> && modellab logs <ask-id>     # <ask-id>: the id that command printed
```

Without the adapter path, `ask.py` gets the untrained model's guess; with it,
the answer from the training data. In the training log, the loss is how far
the model's answers are from the examples, lower is better; the validation
loss is measured on the examples in `valid.jsonl`, which training does not
see. An adapter trained on ten lines repeats them and invents the rest: ask it
something the data does not say and it answers confidently anyway.

`models add` may warn about unauthenticated Hugging Face requests; public
models need no token. Messages in the logs from Python or MLX-LM marked as
warnings are not errors unless the run's exit code is not 0.

To train on your own data, copy `examples/data`, write your own questions and
answers in `train.jsonl` and `valid.jsonl` (one chat per line, in the same
shape), and add the folder's path after `examples/finetune_text.py`. If `run`
refuses the 4G budget or keeps the job waiting, use `--budget 1G`: the demo
peaks under 1 GB. The two rules for every job, under Commands, say more.

## License

MIT. See `LICENSE`.
