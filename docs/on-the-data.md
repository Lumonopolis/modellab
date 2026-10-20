# Keeping work on the data

Status, 2026-10-02: steps 1 to 3 of the build order are built (`docs/gold-sets.md`, `docs/evaluation.md`, `docs/training.md`).

A long unattended agent run drifts when the agent's attention moves from the goal and
the data to its own procedure, and to numbers that stand in for them. ModelLab
does not add rules against this. It changes what the agent sees and what is
easiest to do: the data next to the goal's target at every step, and a short
path from a change to its effect on that target.

## 1. Every view shows the item next to the goal's target

`modellab look` renders items from a gold set or a run as sheets an agent can
see whole: the input, the outputs and the target side by side. The target is
what the goal asks for on that item, such as the human label, rating, edit or
answer. It is never a stand-in for it.

- Images and video frames are shown large enough to judge.
- Text is shown in full.
- Structured fields are shown as a table.

Every item carries an id. The default is a random sample with a recorded seed;
`--lowest` and `--highest` sort by any score. The labelling tool uses the same
display configuration, so a person and an agent see the same view.

## 2. Every result arrives with its items

A ModelLab evaluation report (`modellab report`) opens with sheets: a random
sample, and the items that scored lowest. The numbers follow:

- the result, with a 95% interval;
- for a scorer that marks an answer right exactly when it equals the target,
  what always giving one answer scores on the same items; a result that does
  not beat the best such answer is marked no better than trivial. For other
  scorers the report says why it gives no such baseline;
- how often each target and each answer occurred.

A training run ends with the same kind of look: replies on declared items
beside their targets, and the adapter against the untrained model
(`docs/training.md`).

Not built yet: when a scorer (a model or an agent) produces the result, a
measure of how well it agrees with people on the targets, with a sheet of the
items where they disagree, so that a scorer no better than a trivial one cannot
select training data.

## 3. The whole loop runs small first

Evaluation and training recipes run end to end on a small sample by default,
in minutes, and end with the sheets from section 2. The first comparison with
the target comes before any long work, and each change shows its effect on the
target soon after it is made.

Before a long training run, the recipe itself runs one step at the largest
input size and reports the peak, to compare with each host's allowance (RAM
minus `reserve`). It keeps two checkpoints, the latest and the best by the
goal's measure, deletes older ones as it goes, and resumes from the latest, so
`--resumable` costs at most the steps since the last checkpoint. MLX-LM and MLX-VLM keep every checkpoint
they save; the recipe removes the extra ones.

## Build order

Each step starts with the adopt-or-build check used for the machines layer.

1. The gold-set format, the display configuration and `modellab look`, shared
   with the labelling tool.
2. Reports with sheets, and the scorer check, on the adopted evaluation harness.
3. Recipes with the small-sample default, the fit step and checkpoints.
