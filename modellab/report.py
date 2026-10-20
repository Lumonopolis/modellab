"""A report of one evaluation: its items first, then its numbers.

An Inspect AI log is read in the eval environment (modellab/inspect_log.py); its items are drawn
with `look` beside the gold set's panes and targets: a random sample, and the items that scored
lowest, each with the raw reply next to the parsed answer. The numbers follow: the result with
a 95% interval, and how often each target and answer occurred.

When the scorer marks an answer right exactly when it equals the item's one target (ignoring
case, spaces and end punctuation), checked on every scored run, the report also gives what
always giving one answer scores on the same items. A result whose paired interval against each
best constant answer does not lie above zero is marked no better than trivial. For any other
scorer (several targets, partial matches, graded or judged scores) the report says why it gives
no such baseline rather than computing a wrong one.

Standard library only; the sheets come from `look`.
"""

import collections
import datetime
import json
import math
import random

RESAMPLES = 2000
FEW = 30  # below this many items, a bootstrap interval is rough
SHOWN = 60  # characters of an answer or target kept in the counts
FIELDS = ("eval_target", "eval_answer", "eval_score", "eval_reply")  # the outputs' names, never a gold field's


def normal(text):
    """An answer or target as compared: case, surrounding spaces and end punctuation ignored."""
    return str(text).strip().lower().strip(".!?,;:\"'").strip()


def per_item(rows, scorer):
    """(items, failed): one entry per item scored by `scorer` (NAME, or NAME:KEY for a scorer
    giving several values), and the ids of items whose every run failed."""
    name, _, key = scorer.partition(":")
    grouped, failed = collections.defaultdict(list), []
    for row in rows:
        if row["error"]:
            failed.append(row["id"])
            continue
        score = row["scores"].get(name)
        if score is None:
            raise SystemExit(f"item {row['id']} has no score from {name!r}; it has {sorted(row['scores'])}")
        if key:
            if not score["values"] or key not in score["values"]:
                raise SystemExit(f"{name!r} gives item {row['id']} no value {key!r}")
            value = score["values"][key]
        elif score["values"] is not None:
            raise SystemExit(f"{name!r} gives several values per item ({', '.join(score['values'])}); "
                             f"report one, e.g. --scorer {name}:{next(iter(score['values']))}")
        else:
            value = score["value"]
        if value is None:
            raise SystemExit(f"{scorer!r} gives item {row['id']} {score['raw']!r}, which is not a number")
        grouped[row["id"]].append((row, value, score["answer"]))
    items = []
    for item_id, runs in grouped.items():
        first = min(runs, key=lambda run: run[0]["epoch"])[0]
        targets = first["targets"]
        answers = collections.Counter("" if answer is None else str(answer) for _, _, answer in runs)
        items.append({
            "id": item_id, "target": first["target"], "targets": targets,
            "answer": answers.most_common(1)[0][0], "reply": first["reply"], "epochs": len(runs),
            "score": sum(value for _, value, _ in runs) / len(runs),
            "binary": all(value in (0.0, 1.0) for _, value, _ in runs),
            # right exactly when the answer is the one target, on every run of the item
            "exact": len(targets) == 1 and all(
                (value == 1.0) == (answer is not None and normal(answer) == normal(targets[0])) for _, value, answer in runs)})
    return items, sorted(set(failed) - set(grouped))


def numbers(items, seed):
    """The result and its interval; for an exact-match scorer, the constant-answer baselines and
    whether the result beats each best one."""
    n = len(items)
    scores = [item["score"] for item in items]
    out = {"n": n, "result": sum(scores) / n, "seed": seed, "baselines": None, "difference": None,
           "better_than_trivial": None, "no_baseline": None}
    if not all(item["binary"] for item in items):
        out["no_baseline"] = "the scores are graded, not right or wrong"
    elif any(len(item["targets"]) != 1 for item in items):
        out["no_baseline"] = "some items have several targets, so a constant answer would score whatever the scorer gives"
    elif not all(item["exact"] for item in items):
        out["no_baseline"] = ("this scorer does not mark an answer right exactly when it equals the target "
                              "(as match, includes or a model judge may not)")
    classes = [normal(item["targets"][0]) for item in items] if out["no_baseline"] is None else None
    best = []
    if classes is not None:
        out["baselines"] = {value: classes.count(value) / n for value in sorted(set(classes))}
        top = max(out["baselines"].values())
        best = [value for value, share in out["baselines"].items() if share == top]
    rng = random.Random(seed)
    results, gaps = [], {value: [] for value in best}
    for _ in range(RESAMPLES):  # paired: each best constant is scored on the same draw
        draw = [rng.randrange(n) for _ in range(n)]
        results.append(sum(scores[i] for i in draw) / n)
        for value in best:
            gaps[value].append(sum(scores[i] - (classes[i] == value) for i in draw) / n)
    out["interval"] = _interval(results)
    if best:
        intervals = [_interval(gap) for gap in gaps.values()]
        out["difference"] = min(intervals)  # the hardest of the tied best answers
        out["better_than_trivial"] = all(low > 0 for low, _ in intervals)
    if classes is not None:
        out["targets"] = dict(collections.Counter(classes))
        out["answers"] = dict(collections.Counter(normal(item["answer"]) for item in items))
    else:
        out["targets"] = dict(collections.Counter(str(item["target"])[:SHOWN] for item in items))
        out["answers"] = dict(collections.Counter(str(item["answer"])[:SHOWN] for item in items))
    return out


def _interval(values):
    values = sorted(values)
    return [values[math.floor(0.025 * len(values))], values[math.ceil(0.975 * len(values)) - 1]]


def write(folder, log, scorer, figures, failed, sheets):
    """report.md: where the items are, then the numbers."""
    rate = lambda value: f"{value:.3f}"
    lines = [f"# {log['task']}: {log['model']}", "",
             f"- Log: `{log['path']}` (Inspect AI {log['inspect']}, {log['created']})",
             f"- Gold set: `{log['set']}`", f"- Scorer: {scorer}", f"- Items scored: {figures['n']}",
             f"- Made: {datetime.datetime.now().isoformat(timespec='seconds')}", "", "## Look first", ""]
    lines += [f"- `{name}/`: {how}; see `{name}/sheets.md` and its sheets." for name, how in sheets]
    lines += ["", "Read the items before the numbers: a number says how often, the items say what.", "",
              "## Numbers", "",
              f"- Result: {rate(figures['result'])} (95% interval {rate(figures['interval'][0])} to "
              f"{rate(figures['interval'][1])}, {figures['n']} items, bootstrap seed {figures['seed']})"]
    if figures["n"] < FEW:
        lines.append(f"- With fewer than {FEW} items the interval is rough; read the sheets.")
    if figures["baselines"] is None:
        lines.append(f"- No constant-answer baseline: {figures['no_baseline']}. "
                     "Compare with a baseline model's report on the same items instead.")
    else:
        lines.append("- Always giving one answer, scored the same way on the same items: "
                     + ", ".join(f"\"{value}\" {rate(share)}" for value, share in figures["baselines"].items()))
        low, high = figures["difference"]
        verdict = ("better than trivial" if figures["better_than_trivial"]
                   else "**no better than trivial**: do not let it decide results or choose training data")
        lines.append(f"- Against the best of these: {verdict} (paired 95% interval of the difference "
                     f"{low:+.3f} to {high:+.3f})")
    lines += [f"- Targets: {_counts(figures['targets'])}", f"- Answers: {_counts(figures['answers'])}"]
    if failed:
        lines.append(f"- Items that failed to run and are not counted ({len(failed)}): {', '.join(failed[:20])}"
                     + (" ..." if len(failed) > 20 else ""))
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def _counts(counter, most=12):
    ordered = sorted(counter.items(), key=lambda pair: -pair[1])
    text = ", ".join(f"\"{key}\" {count}" for key, count in ordered[:most])
    return text + (f", and {len(ordered) - most} more" if len(ordered) > most else "")


def outputs(path, items):
    """The items as outputs `look` joins to the gold set by id."""
    path.write_text("".join(json.dumps(dict(zip(("id",) + FIELDS, (item["id"], item["target"], item["answer"],
                                                                   item["score"], item["reply"]))),
                                       ensure_ascii=False) + "\n" for item in items))
    return path
