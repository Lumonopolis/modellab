# Evaluation and reports

Status, 2026-10-01: built. Step 2 of `docs/on-the-data.md`.

Inspect AI runs evaluations; ModelLab adds what makes a result readable: the items first,
then the numbers, with what a trivial answerer would have scored on the same items.

## Running an evaluation

Inspect AI lives in its own environment, pinned exactly (it is released almost daily and
its lock holds 86 packages): `modellab env sync eval`. Models are reached through
`modellab serve`, which prints an OpenAI-compatible address:

```sh
modellab serve qwen3-vl-8b-4bit                       # ready: http://127.0.0.1:<port>/v1
LOCAL_API_KEY=none <root>/envs/eval/bin/inspect eval tasks.py@my_task \
  --model openai-api/local/<model path from /v1/models> --model-base-url http://127.0.0.1:<port>/v1 \
  --log-dir logs --no-log-images
```

- Sample ids are the gold set's item ids, so results join to items.
- Images are given by path. `--no-log-images` (or `log_images=False`) keeps them out of the
  log; the log names the files instead.
- Name the task (`tasks.py@my_task`): Inspect 0.3.273 fails on an absolute task-file path
  without one.
- A scorer that reads an answer from a reply must find it wherever the reply puts it
  ("Picture A is the finished edit." answers "a"). The report shows each raw reply next to the
  parsed answer, so a scorer that misreads is seen at once.
- `inspect_ai.score()` re-scores a saved log without calling the model again.

## `modellab report`

```sh
modellab report logs/<log>.eval --set <gold set> [--scorer NAME] [--sample 24] [--seed 1] [--to folder]
```

Writes a folder:

- `sample/`: a random sample of the evaluated items, drawn by `look` beside the gold set's
  panes and targets, with `eval_target`, `eval_answer`, `eval_score` and `eval_reply` for
  each (a gold set with fields of those names is refused).
- `lowest/`: the items that scored lowest, misses first, the same way.
- `report.md`: where those sheets are, then the numbers:
  - the result with a 95% bootstrap interval (marked rough below 30 items);
  - what always giving each one answer scores on the same items, when the scorer marks an
    answer right exactly when it equals the item's one target (case, spaces and end
    punctuation aside), checked on every run of every item;
  - whether the result beats each best of these: the paired 95% interval of the difference
    must lie above zero, or the result is marked **no better than trivial** and must not
    decide results or choose training data;
  - otherwise, why there is no such baseline: several targets, graded scores, or a scorer
    (`match`, `includes`, a model judge) that can mark an answer right without it being the
    target. Compare with a baseline model's report on the same items instead;
  - how often each target and each answer occurred, and which items failed to run.
- `outputs.jsonl`: the items as `look` outputs, for further sheets.

Several epochs of an item are averaged. A scorer giving several values per item is reported
one value at a time: `--scorer NAME:KEY`. The seed is recorded and makes the sample and
intervals repeatable. Each log item must carry its gold item's id (`Sample(id=...)`); a log
with ids the set lacks, or saved without samples, is refused. The folder appears only once
the report is whole.

## Known to work

- 2026-10-01: a photo archive (40 allowed frames, 20 black and white), qwen3-vl-8b
  through `modellab serve` on an M4 Max. "Will this become black and white?" scored 0.475
  (it answered "no" 39 times); a judge asked which of two pictures is the finished edit
  scored 0.500 after a corrected scorer (it answered "B" 35 times). Both were marked no
  better than trivial; the first report also showed the misreading scorer.
