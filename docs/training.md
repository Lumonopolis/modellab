# Training recipes

Status, 2026-10-02: step 3 of `docs/on-the-data.md`, in pieces.

1. Built:
   - the prompt check, which runs before every training run;
   - the fit step;
   - jobs import ModelLab.
2. Built:
   - checkpoints kept as the run goes (the latest and the best by the goal's
     measure);
   - exact resume from the latest checkpoint.
3. Built:
   - the small-sample default;
   - the look every run ends with: a health line on the replies, the adapter
     against the untrained model, and sheets, on items the project declares.

## Adopt or build

- **Adopted:** MLX-VLM's trainer, pinned in `envs/mlx`. It trains LoRA, masks
  the prompt (`train_on_completions`) and saves every N steps. Its CLI resumes
  from an adapter's weights only, not the optimizer's state or the step.
- **Missing there:**
  - a fit step;
  - a way to keep only some checkpoints (it keeps every one);
  - any check that the prompts a sampler sends match what training saw.
- **Elsewhere:** TRL warns when a generation prompt is not a token prefix of the
  full conversation, then trains anyway ([issue 7449], 2026-09-29: Gemma 4;
  Qwen3.5 and other templates also trigger it). It never sees the sampler's own
  prompts.
- **Built here:** a thin layer over MLX-VLM's trainer, `modellab.recipe`.

[issue 7449]: https://github.com/huggingface/trl/issues/7449

## Using the recipe

A project's training script runs as a ModelLab job. ModelLab's code is on the
job's `PYTHONPATH`. The script loads the model and builds its dataset: MLX-VLM's
`VisionDataset`, or a subclass that adds to it (such as weights for the loss).
It then calls `recipe.train`:

```python
import json, os
from mlx_vlm import load
from mlx_vlm.trainer.datasets import VisionDataset
from modellab import recipe
from sampler import prompts          # the sampler's own prompt builder

model, processor = load(os.environ["MODELLAB_MODEL"])
dataset = VisionDataset(items, model.config.__dict__, processor, train_on_completions=True)
look = recipe.Look(items=calibration, declared="data/look-ids.txt", allowed="data/allowed-ids.txt",
                   parse=json.loads)
recipe.train(model, processor, dataset, lambda item: prompts(processor, model.config, item),
             "adapters/real", look=look, epochs=2, full=full, fit=fit)   # full, fit: the script's own flags
```

`items` are MLX-VLM examples: `{"id", "images": [paths], "messages": [...]}`.
Give each an `id`: problems name examples by it, or else as "example 412".
`prompts` returns every prompt text the sampler sends for an example. Use the
sampler's own function, not a copy: the check is only as good as that. A
check script of its own can do the same: import the training script's dataset
and the sampler's own prompt functions.

## The prompt check

Before anything else, `train` calls `recipe.check_prompts`. For every training
example and every prompt the sampler sends for it, it checks three things:

1. The prompt's tokens are the first tokens of the example, as the trainer makes
   them. The prompt's tokens are made the way MLX-VLM's `generate` and
   `stream_generate` make them, from the prompt text and the example's images
   (`_prepare_generation_inputs`). The example's tokens come from the dataset's
   own `process`, which is what the trainer's batches use. `batch_generate` and
   the server are not modelled.
2. The prompt reaches the first token the trainer learns from, where the
   completion mask starts.
3. The prompt leaves some of the example for the model to write.

`train` also fails any example longer than `max_seq_length`. MLX-VLM cuts such an
example's reply short, or skips the example when its image tokens do not fit,
with only a warning in the log. On Qwen3-VL and Qwen3.6 one 12-megapixel photo
is about 11,750 tokens.

Otherwise the model continues from tokens it never trained on. On 2026-10-02,
one project's sampler ended its prompt with `{` where training had the single
token `{"`, and the adapter answered with unquoted keys. The check finds that,
and also:

- a different system prompt or thinking-block setup between training and
  sampling;
- special tokens made differently: MLX-VLM builds training and generation inputs
  with different functions;
- a `resize_shape` given to generation and not to training, or the reverse
  (`VisionDataset` ignores `image_resize_shape` for Qwen models);
- a prefill or continuation that splits a token training kept whole: `{` against
  `{"`, or `},` against `},"`;
- a chat-template change between library versions;
- MLX-VLM placing the start of the completion mask after where generation starts.

On any mismatch it stops with every problem listed: which example, which prompt,
the token where they differ, and the tokens before it. The summary goes to
`out/prompt-check.json` in the run. A sampler calls the same function before it
generates; a model loaded with `load(path, lazy=True)` is enough, since no
weights are read. Time: about 0.06 s per example with small pictures, and 4
minutes (0.28 s each) for 866 two-photo examples on a busy Mac.

Proof by use, 2026-10-02: on those 866 examples, the project's old sampler
values (`{` and `,`) failed all 1,732 prompts, each at the right token, and the
fixed values (`{"` and `,"`) passed all 1,732.

It does not check:

- prompts the project does not declare;
- images other than the example's own: the sampler side uses the example's
  image files;
- loss weights the project adds on top of the completion mask;
- sampling settings such as temperature.

## The fit step

`train(..., fit=True)` runs the prompt check, then one training step on the
example with the most tokens, timed by the recipe's own clock. MLX-VLM's
`It/sec` is wrong for a short last report. It reports the seconds and the peak
memory, to the job's log and to `out/fit.json`. Its adapter goes to a temporary
folder and is deleted, so it can never be taken for a trained one. With a
`measure`, it runs that once after the step, so the peak includes generation
during training. Compare the peak with each host's allowance (RAM minus
`reserve`) before a long run.

## Training

- `train` adds LoRA adapters (rank 16, alpha 32 by default) to the language
  model's linear layers, as MLX-VLM's CLI does. On a mixture-of-experts model
  such as Qwen3.6-35B-A3B that includes the router and shared-expert gates, not
  the routed experts. It runs MLX-VLM's trainer with batch size 1, the dataset's
  completion mask, and the project's `loss` when given.
- The learning rate is constant, as in MLX-VLM's CLI: its trainer does not use
  `TrainingArgs.warmup_steps`.
- The steps are rounded up to a whole number of `accumulate` steps. Otherwise
  MLX-VLM drops the examples of a last, partial update.
- `seed` fixes which examples each segment trains on (a fresh shuffle for each
  pass), MLX-VLM's shuffle within the segment, dropout and the adapters'
  starting values.
- The result goes to `out/train.json`: the steps, the step it resumed from, the
  seconds and peak memory of this start, the latest and best checkpoints, and
  every measure.

## Checkpoints and resuming

The recipe runs MLX-VLM's trainer in segments of `save_every` steps (default:
one pass; a multiple of `accumulate`). After each segment it saves a checkpoint
to `<out>/checkpoints/step-NNNNNNN/`: the adapter, its config and the
optimizer's state.

- **Kept:** the latest checkpoint, and the best by `measure(model)`. Others
  are deleted as the run goes. Without `measure`, only the latest is kept.
- **The measure:** a number, higher is better, computed on the goal's target,
  such as a gate score on calibration items. It runs in evaluation mode, after
  the checkpoint is saved. A NaN is recorded as no measure. A tie keeps the
  earlier checkpoint. A measure that fails, or a stop during it, ends the run
  with its checkpoint saved; running again measures that checkpoint first, so
  no training and no measure is lost.
- **State:** `<out>/recipe-state.json` names the latest and best checkpoints,
  every measure, and the run's fingerprint. It is replaced atomically once a
  checkpoint is complete, so a stop at any moment leaves a checkpoint to resume
  from. This holds against a killed job, not against power loss: nothing is
  synced to disk.
- **Adapter:** `<out>/adapters.safetensors` (and `adapter_config.json`) is the
  latest adapter. It is replaced atomically after the state names its
  checkpoint. Load the best from its checkpoint folder.
- **Resuming:** run the same script again. If `<out>` holds the same run, the
  recipe loads the latest checkpoint's adapter and optimizer state and continues
  with the same examples and random numbers. First it finishes the latest
  checkpoint, in case the stop came before that was done: it copies the adapter
  into `<out>` and measures it if it has no measure yet. A run stopped after a
  checkpoint and resumed ends with the same weights, to the last bit, as one
  that was never stopped. This includes dropout, gradient accumulation across
  the stop, and a short last segment. So `modellab run --resumable` loses at
  most `save_every` steps. A finished run, run again, trains nothing more.
- **The same run:** the fingerprint covers every array the trainer is given
  (tokens, masks, pixels, and anything a dataset subclass adds), the base
  model's weights by name, shape and type (a 4-bit and an 8-bit build differ),
  and the settings. It does not cover the loss function's code, or weight
  values: another build of the same architecture and quantization passes. If
  `<out>` holds a different run, the recipe stops and names what differs; it
  never mixes two runs.
- **Disk:** each checkpoint writes the adapter twice and the optimizer state,
  about three times the adapter's size. When the disk is short, the guard counts
  a job as filling it from about 1.76 GiB in 30 minutes, so checkpoints that
  come faster than that can get the job stopped.
- Other files in `<out>` are left alone; only `checkpoints/step-NNNNNNN` folders
  are ever deleted.

## Notes for samplers

- On 2026-10-02 MLX-VLM's `stream_generate` held back the text of compact JSON
  until the end of generation (`docs/stack.md`, Known limits). A sampler that
  stops on a pattern in the streamed text stops late or never. Decode the
  collected token ids instead.

## Small first

By default `train` runs small: `small` examples (8, chosen by `seed`), one pass,
one checkpoint, into `<out>/small`, and then the look. The prompt check still
covers every example. So the whole loop, from the first example to replies on
the goal's items, runs before any long work. `full=True` trains on everything,
into `<out>`. A small run and a full run never share a folder.

## The look at the end of a run

Every run ends with a look, on items the project declares (`recipe.Look`).
`train` refuses to start without one; `look=None` runs without it, on purpose.

- **Items:** MLX-VLM items with `id`, `images` and `messages`; an assistant
  message is the target. Use calibration items, never test items.
- **Declared ids:** `declared` lists the ids a look may use (a list, or a file
  with one id per line; blank lines and lines starting with `#` are skipped).
  Any other item, an item without an id, or an id given twice is refused before
  training starts, so a test item cannot slip into a look.
- **Allowed ids:** `allowed` lists the ids the look may show. Other declared
  items appear only as counts in the health line: never by id, reply or
  picture, in any file or the log. A project whose private items must never be
  shown leaves them off this list.
- **Checked first:** before training, the recipe also reads the allowed list,
  makes every item's prompt and finds every picture, so the look cannot fail
  after a long run for want of them.
- **Replies:** greedy, up to `max_tokens`, from the sampler's first prompt for
  the item (or `Look.prompt`), with the same generation settings as the prompt
  check (`generation`, such as `resize_shape`). The default `max_tokens` is 1.5
  times the longest reply in training; without a completion mask the recipe
  cannot tell, and `max_tokens` must be given. The recipe collects token ids,
  not MLX-VLM's streamed text. The reply given to `parse` is the prompt's
  prefill (what follows the chat template's own generation prompt) and what the
  model wrote.

It writes `<out>/look/` (`<out>/small/look/` for a small run) and prints two
lines to the job's log:

1. **Health:** how many replies parse (`Look.parse`), how many stopped at
   `max_tokens`, the reply lengths, and the first three tokens of the first
   allowed replies. It also counts prompts that do not start with the chat
   template's generation prompt: their replies are parsed without the prefill,
   which can make them look unparseable. On 2026-10-02 both of that project's
   faults would have shown here within minutes: unquoted keys as 0% parsing,
   and a sampler that never stopped as every reply at `max_tokens`.
2. **Against the untrained model:** before the first step a LoRA adapter
   changes nothing (its second matrix starts at zero). So the recipe keeps the
   untrained model's replies to the first `compare` allowed items (8) then, in
   `<out>/look-base.json`, and compares the adapter's replies at the end: how
   many differ, and the earliest token where they do. An adapter that did not
   train or load shows as 0 of 8. If `max_tokens` changed between the two, or
   no look item is allowed, it says so instead.

The sheets show the allowed items:

- `sheet-NN.png`, drawn with `modellab look`'s `draw_sheets`: the item's
  pictures and, when `Look.picture` returns one, a picture the project made from
  the reply (a render, say). A picture that fails is noted in `sheets.md`.
- `sheets.md`: per item, where it stopped, why it does not parse, and the
  target beside the reply. When `parse` gives a dictionary for both, they are
  shown part by part, with each part that differs marked; otherwise both are
  shown in full in code fences.
- `replies.jsonl` (each allowed reply, with its token ids) and `look.json` (the
  summary, also in the job's `out/train.json`).

A run that is resumed, or a finished run run again, looks again at its end,
against the untrained replies it kept.
