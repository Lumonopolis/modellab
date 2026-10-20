"""Training recipes: MLX-VLM's own LoRA training, with the steps that keep a long run on the data.

Runs inside a ModelLab job, whose PYTHONPATH includes ModelLab. The project's script loads the
model, builds its dataset (MLX-VLM's VisionDataset, or a subclass that adds to it) and calls
`train`. Before anything else, `train` checks the prompts the project's sampler will send
against the training examples as the trainer tokenizes them (`check_prompts`), and stops on any
mismatch. With `fit=True` it then trains one step on the example with the most tokens and
reports time and peak memory, instead of training.

By default `train` runs small: a few examples, one pass, into `<out>/small`, so the whole loop
runs in minutes; `full=True` runs everything. Every run ends with a look (`Look`) at replies on
items the project declares: a health line, the adapter against the untrained model, and sheets.

`check_prompts` is also what a sampler calls before it generates. See docs/training.md.
"""

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SHOWN = 4  # tokens shown before a mismatch
CHECKPOINT = re.compile(r"step-\d{7}")
_WRAPPED = set()  # layer classes MLX-VLM's gradient checkpointing has wrapped in this process
REQUIRED = object()


@dataclass
class Look:
    """What the end of every run shows, on items the project declares (never test items): a
    health line on the replies, the adapter against the untrained model, and sheets.

    items: MLX-VLM items with "id", "images" and "messages"; an assistant message is the target.
    declared: the ids a look may use, as a list or a file with one id per line. Any other item
    is refused before training starts.
    allowed: the ids sheets may show, as a list or a file. Other items count only in the health
    line.
    parse: a reply's text (prefill included) -> its value; None, or an exception, when it does not
    parse.
    prompt: item -> the prompt text; by default the first of the sampler's prompts.
    picture: (item, reply, value) -> an image the project made from the reply (a path), or None.
    max_tokens: by default 1.5 times the longest reply in training.
    compare: how many replies are compared with the untrained model's.
    """
    items: list
    declared: Any
    allowed: Any
    parse: Callable
    prompt: Callable = None
    picture: Callable = None
    max_tokens: int = None
    compare: int = 8


def check_prompts(model, processor, dataset, prompts, max_seq_length=None, **generation):
    """Check, for every training example, each prompt the sampler will send for it.

    A prompt passes when its tokens, made the way MLX-VLM's generate and stream_generate make
    them (the example's images included), are the first tokens of the example as the trainer
    makes them, reach the first token the trainer learns from (where the completion mask
    starts), and leave some of it to write. Otherwise the
    model is asked to continue from tokens it was never trained on: on 2026-10-02 a sampler
    that ended its prompt with '{', where training had the single token '{"', got replies with
    unquoted keys.

    prompts: example -> list of the prompt texts the sampler sends for it.
    max_seq_length: the trainer's; a longer example is cut or skipped by MLX-VLM, so it fails.
    generation: what the sampler passes to generation besides the prompt and images, such as
    resize_shape.

    Returns a summary with each example's token count (and its reply's) and a digest of every array the trainer
    is given (tokens, masks, pixels, anything a dataset subclass adds). Writes it to the run's
    out folder in a ModelLab job. Raises SystemExit with every mismatch, so a training run never
    starts on one.
    """
    import numpy as np
    from mlx_vlm.generate.dispatch import _prepare_generation_inputs

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    stand_in = types.SimpleNamespace(config=model.config)  # all generation reads from the model
    show = lambda ids: " ".join(repr(tokenizer.decode([int(i)])) for i in ids)
    lengths, replies, problems, checked, digest = [], [], [], 0, hashlib.sha256()
    for index, item in enumerate(dataset.dataset):
        name = item.get("id", f"example {index}")
        out = dataset.process(item)
        for key in sorted(out):
            _fold(digest, key, out[key])
        ids = np.array(out["input_ids"]).reshape(-1)
        lengths.append(len(ids))
        if max_seq_length and len(ids) > max_seq_length:
            problems.append(f"{name}: {len(ids)} tokens, more than max_seq_length {max_seq_length}: MLX-VLM "
                            "would cut its reply short, or skip it if its image tokens do not fit")
        images = item.get("images", item.get("image", []))  # as VisionDataset.process reads them
        images = images if isinstance(images, list) else [images] if images else []
        mask = out.get("completion_mask")
        start = int(np.argmax(np.array(mask).reshape(-1) > 0)) if mask is not None else None
        replies.append(len(ids) - start if start is not None else None)
        texts = prompts(item)
        if not texts:
            problems.append(f"{name}: no prompt; declare every prompt the sampler sends for this example")
            continue
        for number, text in enumerate(texts, 1):
            checked += 1
            made = _prepare_generation_inputs(stand_in, processor, text, images or None, None, None,
                                              dict(generation))[0]
            sent = np.array(made).reshape(-1)
            where = f"{name}, prompt {number} of {len(texts)}"
            shared = min(len(sent), len(ids))
            differ = next((i for i in range(shared) if sent[i] != ids[i]), None)
            if differ is not None:
                problems.append(f"{where}: differs at token {differ} of {len(ids)}: training has "
                                f"{show(ids[differ:differ + 1])}, the prompt has {show(sent[differ:differ + 1])}; "
                                f"before it: {show(ids[max(0, differ - SHOWN):differ])}")
            elif len(sent) >= len(ids):
                problems.append(f"{where}: {len(sent)} tokens of the example's {len(ids)}: nothing is left "
                                "for the model to write")
            elif start is not None and len(sent) < start:
                problems.append(f"{where}: ends at token {len(sent)}, before the first trained token {start} "
                                f"({show(ids[len(sent):start][:SHOWN])} ...): the model would write tokens "
                                "training never taught it")
    summary = {"examples": len(lengths), "prompts": checked, "mismatches": len(problems),
               "problems": problems, "tokens": lengths, "reply_tokens": replies, "digest": digest.hexdigest()}
    _record("prompt-check.json", summary)
    if problems:
        listed = "\n".join(f"- {problem}" for problem in problems[:20])
        more = f"\n- and {len(problems) - 20} more (prompt-check.json)" if len(problems) > 20 else ""
        raise SystemExit(f"Prompt check failed: {len(problems)} of {checked} prompts. Each prompt the sampler "
                         f"sends must be the first tokens of its training example as the trainer makes them, "
                         f"and reach the first trained token.\n{listed}{more}")
    print(f"Prompt check passed: {checked} prompts for {len(lengths)} examples.", flush=True)
    return summary


def train(model, processor, dataset, prompts, out, *, look=REQUIRED, full=False, small=8, loss=None,
          generation=None, fit=False, epochs=1, learning_rate=1e-4, rank=16, alpha=32, dropout=0.0, accumulate=1,
          max_seq_length=8192, save_every=None, measure=None, seed=0, report_every=10):
    """Check the sampler's prompts, then train LoRA adapters with MLX-VLM's trainer, or with
    `fit=True` train one step on the longest example into a temporary folder.

    The adapters go on the language model's linear layers, as MLX-VLM's own CLI puts them. The
    learning rate is constant, as there (its TrainingArgs.warmup_steps is not used by its
    trainer). dataset: an MLX-VLM VisionDataset (or subclass). loss: MLX-VLM's loss_fn
    signature; its own loss when None. The steps are rounded up to a whole number of
    `accumulate` steps, so no examples are left out of the last update.

    Every `save_every` steps (a multiple of `accumulate`; default: one pass, rounded up to one)
    the recipe saves a checkpoint to `out/checkpoints`: the adapter and the optimizer's state.
    It keeps two, the latest and the best by `measure(model)` (higher is better; the earlier
    wins a tie; without it, only the latest), and deletes the others as it goes. Run again with
    the same examples, model and settings, it resumes from the latest exactly: same weights,
    optimizer state, order of examples and random numbers, so `modellab run --resumable` loses
    at most `save_every` steps. `out/adapters.safetensors` is the latest adapter.

    Unless `full`, the run is small: `small` examples (chosen by `seed`), one pass, one
    checkpoint, into `out/small`. The prompt check still covers every example. `look`
    (a `Look`, or None to run without one) is required: the run ends with it, in `<run>/look`.
    Returns the result, also written to the run's out folder.
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model

    if look is REQUIRED:
        if not fit:
            raise SystemExit("train needs look=recipe.Look(...): the replies a run ends with. Pass look=None "
                             "to run without one.")
        look = None
    if not full and small < 1:
        raise SystemExit(f"small must be at least 1, not {small}")
    if look:
        prompt_of = look.prompt or (lambda item: prompts(item)[0])
        allowed = _ready(look, prompt_of)  # before anything slow
    checked = check_prompts(model, processor, dataset, prompts, max_seq_length, **(generation or {}))
    base = _model_digest(model)
    mx.random.seed(seed)  # the adapters' starting values
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=rank, alpha=alpha,
                           dropout=dropout, verbose=False)
    optimizer = optim.Adam(learning_rate=learning_rate)
    if fit:
        return _fit(model, optimizer, dataset, checked["tokens"], loss, max_seq_length, learning_rate, measure)

    items, out = dataset.dataset, Path(out)
    if not full:
        chosen = sorted(np.random.default_rng(seed).permutation(len(items))[:small].tolist())
        items, out, epochs, save_every = [items[i] for i in chosen], out / "small", 1, None
        dataset = copy.copy(dataset)
        dataset.dataset = items
        print(f"A small run: {len(items)} examples, one pass, into {out}. full=True runs them all.", flush=True)
    every = save_every or math.ceil(len(items) / accumulate) * accumulate
    if every % accumulate:
        raise SystemExit(f"save_every ({every}) must be a multiple of accumulate ({accumulate}), so each "
                         "checkpoint falls between updates")
    steps = math.ceil(len(items) * epochs / accumulate) * accumulate
    plan = {"examples": checked["digest"], "model": base, "full": full, "small": None if full else small,
            "count": len(items), "epochs": epochs, "steps": steps,
            "seed": seed, "save_every": every, "accumulate": accumulate, "learning_rate": learning_rate,
            "rank": rank, "alpha": alpha, "dropout": dropout, "max_seq_length": max_seq_length}
    state = _read_state(out)
    if state and state["plan"] != plan:
        changed = sorted(key for key in plan if state["plan"].get(key) != plan[key])
        raise SystemExit(f"{out} holds a run with other {', '.join(changed)} (recipe-state.json). Train into "
                         "another folder, or remove recipe-state.json and checkpoints/ to start again.")
    state = state or {"plan": plan, "latest": None, "best": None, "measures": []}
    begun = 0
    if state["latest"]:
        folder = out / "checkpoints" / state["latest"]
        begun = _resume(model, optimizer, folder)
        print(f"Resuming from {state['latest']} (step {begun} of {steps}).", flush=True)
        _settle(model, out, folder, begun, state, measure)  # in case the stop came before it was done
    order = _order(len(items), steps, seed)
    if look:
        if look.max_tokens is None and None in checked["reply_tokens"]:
            raise SystemExit("Look.max_tokens is needed: without a completion mask the recipe cannot tell how long "
                             "the replies in training are")
        max_tokens = look.max_tokens or math.ceil(1.5 * max(checked["reply_tokens"]))
        base_file = out / "look-base.json"
        if not state["latest"]:  # before the first step the adapters change nothing: the untrained model
            compared = [item for item in look.items if str(item["id"]) in allowed][:look.compare]
            untrained = _replies(model, processor, compared, prompt_of, max_tokens, generation)
            out.mkdir(parents=True, exist_ok=True)
            base_file.write_text(json.dumps({"max_tokens": max_tokens,
                                             "replies": {row["id"]: row["ids"] for row in untrained}}) + "\n")
    mx.reset_peak_memory()
    began = time.time()
    for start in range(begun, steps, every):
        end = min(start + every, steps)
        folder = out / "checkpoints" / f"step-{end:07d}"
        _segment(model, optimizer, dataset, [items[i] for i in order[start:end]], seed + start, loss,
                 max_seq_length, learning_rate, accumulate, report_every, folder / "adapters.safetensors")
        _checkpoint(model, optimizer, out, folder, end, state, measure)
    result = {"full": full, "steps": steps, "resumed_from": begun, "seconds": round(time.time() - began, 1),
              "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2), "latest": state["latest"],
              "best": state["best"], "measures": state["measures"]}
    if look:
        result["look"] = _look(model, processor, look, allowed, prompt_of, max_tokens, generation, base_file,
                               out / "look")
    print(json.dumps(result), flush=True)
    _record("train.json", result)
    return result


def _ready(look, prompt_of):
    """Refuse, before training starts, a look that would fail or show what it must not: every
    item declared and named once, the allowed list readable, every prompt made and every picture
    there. Returns the allowed ids."""
    if not look.items:
        raise SystemExit("Look has no items")
    declared, allowed = _ids(look.declared), _ids(look.allowed)
    names = [str(item["id"]) if "id" in item else None for item in look.items]
    outside = [name for name in names if name is not None and name not in declared]
    twice = sorted({name for name in names if name is not None and names.count(name) > 1})
    if None in names or outside or twice:
        raise SystemExit(f"Look items must be declared, each once: {names.count(None)} have no id, {len(outside)} are "
                         f"not in the declared list (such as {', '.join(outside[:5])}), and {len(twice)} ids appear "
                         f"more than once{_named(twice, allowed)}.")
    missing = []
    for name, item in zip(names, look.items):
        prompt_of(item)
        if any("://" not in str(path) and not Path(path).is_file() for path in _images(item)):
            missing.append(name)
    if missing:
        raise SystemExit(f"{len(missing)} look items have a picture that is not there{_named(missing, allowed)}.")
    return allowed


def _named(names, allowed):
    """Which of `names` may be shown, and how many may not: an item that is not allowed is
    never named, even in an error."""
    shown = [name for name in names if name in allowed]
    hidden = len(names) - len(shown)
    parts = ([f"such as {', '.join(shown[:5])}"] if shown else []) + ([f"{hidden} not allowed to be named"]
                                                                      if hidden else [])
    return f" ({'; '.join(parts)})" if parts else ""


def _ids(source):
    """Ids from a list, or from a file with one id per line (blank lines and lines starting with
    '#' are skipped)."""
    if isinstance(source, (str, Path)):
        lines = Path(source).read_text(encoding="utf-8-sig").splitlines()
        return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}
    return {str(value) for value in source}


def _images(item):
    images = item.get("images", item.get("image", []))  # as VisionDataset.process reads them
    return images if isinstance(images, list) else [images] if images else []


def _replies(model, processor, items, prompt_of, max_tokens, generation):
    """Greedy replies, by token ids rather than MLX-VLM's streamed text (which holds back text
    without spaces). The reply is the prompt's prefill (what follows the chat template's
    generation prompt; None when the prompt does not start with it) and what the model wrote."""
    import mlx.core as mx
    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    rows = []
    model.eval()
    try:
        for item in items:
            images = [str(path) if "://" in str(path) else str(Path(path).resolve()) for path in _images(item)]
            prompt = prompt_of(item)
            asked = [turn for turn in item["messages"] if turn["role"] != "assistant"]
            opening = apply_chat_template(processor, model.config, asked, num_images=len(images))
            prefill = prompt[len(opening):] if prompt.startswith(opening) else None
            last = None
            for last in stream_generate(model, processor, prompt, image=images or None, max_tokens=max_tokens,
                                        temperature=0.0, **(generation or {})):
                pass
            ids = list(last.token_ids or [])
            if last.finish_reason == "stop" and ids:
                ids = ids[:-1]  # the end-of-reply token
            rows.append({"id": str(item["id"]), "ids": ids, "prefill": prefill, "finish": last.finish_reason,
                         "reply": (prefill or "") + tokenizer.decode(ids)})
    finally:
        model.train()
        mx.clear_cache()
    return rows


def _differences(base, adapter):
    """For each id in both, in `base`'s order: None when the replies' tokens are the same, else
    the first position where they differ."""
    found = {}
    for key, a in base.items():
        if key in adapter:
            b = adapter[key]
            at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            found[key] = min(len(a), len(b)) if at is None and len(a) != len(b) else at
    return found


def _look(model, processor, look, allowed, prompt_of, max_tokens, generation, base_file, folder):
    """The end of a run: replies on the declared items and their health; the adapter against the
    untrained model and sheets on the allowed ones. Items that are not allowed appear only in
    counts. Writes `folder`; returns the summary."""
    import statistics

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    show = lambda ids: " ".join(repr(tokenizer.decode([int(i)])) for i in ids)
    rows = _replies(model, processor, look.items, prompt_of, max_tokens, generation)
    for row in rows:
        try:
            row["value"] = look.parse(row["reply"])
        except Exception as error:  # a reply that does not parse is what the health line counts
            row["value"], row["error"] = None, f"{type(error).__name__}: {error}"
    shown = [(item, row) for item, row in zip(look.items, rows) if row["id"] in allowed]
    lengths = [len(row["ids"]) for row in rows]
    unknown = sum(row["prefill"] is None for row in rows)
    health = {"replies": len(rows), "parsed": sum(row["value"] is not None for row in rows),
              "at_max_tokens": sum(row["finish"] == "length" for row in rows), "max_tokens": max_tokens,
              "tokens": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
              "prefill_unknown": unknown, "first_tokens": {row["id"]: show(row["ids"][:3]) for _, row in shown[:3]}}
    line = (f"Replies: {health['parsed']} of {len(rows)} parse; {health['at_max_tokens']} stopped at max_tokens "
            f"({max_tokens}); {min(lengths)} to {max(lengths)} tokens (median {health['tokens']['median']}).")
    if health["first_tokens"]:
        line += " First tokens: " + "; ".join(f"{key}: {value}" for key, value in health["first_tokens"].items()) + "."
    if unknown:
        line += (f" {unknown} prompts do not start with the chat template's own generation prompt, so those "
                 "replies were parsed without the prompt's prefill.")
    lines = [line]
    summary = {"health": health, "against_untrained": None, "not_shown": len(rows) - len(shown)}
    kept = json.loads(base_file.read_text()) if base_file.exists() else None
    if kept is None:
        lines.append("Adapter against the untrained model: no untrained replies were kept for this run.")
    elif kept["max_tokens"] != max_tokens:
        lines.append(f"Adapter against the untrained model: not compared, since max_tokens changed from "
                     f"{kept['max_tokens']} to {max_tokens}.")
    else:
        found = _differences(kept["replies"], {row["id"]: row["ids"] for _, row in shown})
        differ = {key: at for key, at in found.items() if at is not None}
        summary["against_untrained"] = {"compared": len(found), "differ": len(differ), "first": differ}
        line = f"Adapter against the untrained model: {len(differ)} of {len(found)} replies differ"
        if not found:
            line = "Adapter against the untrained model: no allowed items to compare"
        elif differ:
            key, at = min(differ.items(), key=lambda pair: pair[1])
            adapted = next(row["ids"] for _, row in shown if row["id"] == key)
            line += (f"; earliest at token {at} ({key}): untrained {show(kept['replies'][key][at:at + 1]) or 'ended'}, "
                     f"adapter {show(adapted[at:at + 1]) or 'ended'}.")
        else:
            line += ": the adapter changed nothing. Check that it trained and loaded."
        lines.append(line)
    for line in lines:
        print(line, flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    summary["sheets"] = _sheets(folder, look, shown, lines, len(rows) - len(shown))
    (folder / "replies.jsonl").write_text("".join(json.dumps({k: v for k, v in row.items() if k != "value"},
                                                             ensure_ascii=False) + "\n" for _, row in shown))
    (folder / "look.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str) + "\n")
    return summary


def _sheets(folder, look, shown, lines, hidden):
    """The allowed items' pictures (and any picture the project made from a reply), drawn with
    `modellab look`'s draw_sheets, and sheets.md with the target beside the reply, part by part."""
    from . import look as sheets

    made, notes = {}, {}
    for item, row in shown:
        try:
            path = look.picture(item, row["reply"], row["value"]) if look.picture else None
        except Exception as error:  # the project's picture fails: say so beside the reply
            path, notes[row["id"]] = None, f"Look.picture failed: {type(error).__name__}: {error}"
        if path is not None and not Path(path).is_file():
            path, notes[row["id"]] = None, f"Look.picture gave {path}, which is not a file"
        made[row["id"]] = path and {"file": str(Path(path).resolve())}
    count = max((len(_images(item)) for item, _ in shown), default=0)
    panes = [{"field": f"picture_{n}", "label": f"Picture {n}", "show": "image", "target": False, "hidden": False}
             for n in range(1, count + 1)]
    if look.picture:
        panes.append({"field": "reply_picture", "label": "Made from the reply", "show": "image", "target": False,
                      "hidden": False})
    entries = [{"id": row["id"], "reply_picture": made[row["id"]],
                **{f"picture_{n}": {"file": str(path) if "://" in str(path) else str(Path(path).resolve())}
                   for n, path in enumerate(_images(item), 1)}} for item, row in shown]
    width = max(1600, sheets.MARGIN + len(panes) * (sheets.MIN_CELL + sheets.MARGIN))
    placed = sheets.draw_sheets("/", panes, entries, width, 1600, folder, {"panes": panes, "questions": []}, {}) \
        if panes and entries else {}
    text = ["# The end of the run", "", *[f"- {line}" for line in lines], "",
            f"Sheets show the {len(shown)} allowed items; {hidden} other declared items count only in the lines "
            "above.", ""]
    for item, row in shown:
        target = next((turn["content"] for turn in item["messages"] if turn["role"] == "assistant"), None)
        try:
            wanted = look.parse(target) if target is not None else None
        except Exception:
            wanted = None
        text += [f"## {sheets.cell(row['id'])}", ""]
        if row["id"] in placed:
            text += [f"Sheet `{placed[row['id']][0]}`, row {placed[row['id']][1]}.", ""]
        text += [f"Stopped: {row['finish']}, {len(row['ids'])} tokens." + (f" Does not parse: {row['error']}"
                                                                            if "error" in row else ""), ""]
        if row["id"] in notes:
            text += [notes[row["id"]], ""]
        if isinstance(wanted, dict) and isinstance(row["value"], dict):
            text += ["| Part | Target | Reply |", "|---|---|---|"]
            for part in list(wanted) + [key for key in row["value"] if key not in wanted]:
                a, b = wanted.get(part, "—"), row["value"].get(part, "—")
                text.append(f"| {sheets.cell(part)}{'' if a == b else ' **≠**'} | {sheets.cell(a)} | {sheets.cell(b)} |")
            text.append("")
        else:
            text += ["**Target**", "", *_fenced("" if target is None else str(target)), "",
                     "**Reply**", "", *_fenced(row["reply"]), ""]
    (folder / "sheets.md").write_text("\n".join(text))
    return {"items": len(shown), "files": sorted({sheet for sheet, _ in placed.values()})}


def _fenced(text):
    """Text in a code fence longer than any run of backticks inside it."""
    fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    return [fence, text, fence]


def _fit(model, optimizer, dataset, tokens, loss, max_seq_length, learning_rate, measure):
    """One step on the example with the most tokens, into a temporary folder (a fit adapter
    must never pass for a trained one), then the measure once, so the peak includes both."""
    import mlx.core as mx

    longest = max(range(len(tokens)), key=tokens.__getitem__)
    result = {"fit_example": dataset.dataset[longest].get("id", longest), "tokens": tokens[longest]}
    print(json.dumps(result), flush=True)
    with tempfile.TemporaryDirectory() as scratch:
        mx.reset_peak_memory()
        began = time.time()
        _segment(model, optimizer, dataset, [dataset.dataset[longest]], 0, loss, max_seq_length, learning_rate,
                 1, 1, Path(scratch) / "adapters.safetensors")
        # Timed here: MLX-VLM's It/sec divides steps_per_report, not the steps run, by the time.
        result.update(steps=1, seconds=round(time.time() - began, 1))
        if measure:
            result["measure"] = _measured(model, measure)
            result["measure_seconds"] = round(time.time() - began - result["seconds"], 1)
        result["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    print(json.dumps(result), flush=True)
    _record("fit.json", result)
    return result


def _segment(model, optimizer, dataset, chosen, seed, loss, max_seq_length, learning_rate, accumulate,
             report_every, adapter_file):
    """MLX-VLM's trainer over `chosen`, with its shuffle and random numbers fixed by `seed`. It
    saves the adapter to `adapter_file` at the end. Its gradient checkpointing wraps the layer
    class each time it is asked, so it is asked once per class in a process."""
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.trainer import TrainingArgs, train as mlx_train

    view = copy.copy(dataset)
    view.dataset = chosen
    np.random.seed(seed % 2**32)  # MLX-VLM shuffles the batches with numpy
    mx.random.seed(seed % 2**32)  # dropout
    layers = {type(child.layers[0]) for child in model.children().values() if hasattr(child, "layers")}
    args = TrainingArgs(batch_size=1, iters=len(chosen), steps_per_report=report_every,
                        steps_per_eval=len(chosen) + 1, steps_per_save=len(chosen) + 1, max_seq_length=max_seq_length,
                        adapter_file=str(adapter_file), grad_checkpoint=not layers <= _WRAPPED,
                        learning_rate=learning_rate, gradient_accumulation_steps=accumulate)
    _WRAPPED.update(layers)
    mlx_train(model=model, optimizer=optimizer, train_dataset=view, args=args,
              train_on_completions=dataset.train_on_completions, **({"loss_fn": loss} if loss else {}))


def _order(count, steps, seed):
    """Which examples each step trains on: a fresh shuffle for each pass, fixed by `seed`. MLX-VLM
    shuffles each segment's examples again, also fixed by the seed."""
    import numpy as np

    rng, order = np.random.default_rng(seed), []
    while len(order) < steps:
        order += rng.permutation(count).tolist()
    return order[:steps]


def _fold(digest, key, value):
    """Add one array the trainer is given to `digest`."""
    import mlx.core as mx
    import numpy as np

    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for index, part in enumerate(value):
            _fold(digest, f"{key}.{index}", part)
        return
    if isinstance(value, mx.array):
        value = np.array(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value)
    value = np.ascontiguousarray(value)
    if value.dtype == object:
        raise SystemExit(f"cannot fingerprint the dataset's {key!r} ({type(value).__name__}) to resume a run safely")
    digest.update(f"{key}:{value.dtype}:{value.shape};".encode())
    digest.update(value.tobytes())


def _model_digest(model):
    """The base model's weights by name, shape and type: a 4-bit and an 8-bit build differ."""
    from mlx.utils import tree_flatten

    names = [(name, list(value.shape), str(value.dtype)) for name, value in tree_flatten(model.parameters())]
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()


def _measured(model, measure):
    """The measure in evaluation mode (no dropout); NaN counts as no measure."""
    import mlx.core as mx

    model.eval()
    try:
        value = float(measure(model))
    finally:
        model.train()
        mx.clear_cache()
    return None if math.isnan(value) else value


def _checkpoint(model, optimizer, out, folder, step, state, measure):
    """After `step`: the optimizer state beside the adapter MLX-VLM saved, then the state file
    (atomically) naming it the latest, then the rest (_settle). A stop at any point leaves a
    checkpoint to resume from."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.save_safetensors(str(folder / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    state["latest"] = folder.name
    _write_state(out, state)
    _settle(model, out, folder, step, state, measure)


def _settle(model, out, folder, step, state, measure):
    """Finish the latest checkpoint: its adapter into `out`, its measure and the best, and the
    others deleted. A resumed run calls it first, so a stop or a failing measure here loses
    nothing; it measures a checkpoint only once."""
    for name in ("adapters.safetensors", "adapter_config.json"):  # the latest adapter, never half-written
        shutil.copyfile(folder / name, out / f".{name}.tmp")
        os.replace(out / f".{name}.tmp", out / name)
    if measure and all(done != step for done, _ in state["measures"]):
        value = _measured(model, measure)
        state["measures"].append([step, value])
        print(f"Step {step}: measure {value}", flush=True)
        if value is not None and (state["best"] is None or value > state["best"]["measure"]):
            state["best"] = {"checkpoint": folder.name, "measure": value}
        _write_state(out, state)
    keep = {state["latest"], state["best"] and state["best"]["checkpoint"]}
    for old in (out / "checkpoints").iterdir():
        if old.is_dir() and CHECKPOINT.fullmatch(old.name) and old.name not in keep:
            shutil.rmtree(old)


def _resume(model, optimizer, folder):
    """Load a checkpoint's adapter and optimizer state; the step it was saved after."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    if not (folder / "optimizer.safetensors").exists():
        raise SystemExit(f"{folder.parent.parent}/recipe-state.json names {folder.name}, which is missing or "
                         "incomplete; remove recipe-state.json and checkpoints/ to start again")
    saved = mx.load(str(folder / "adapters.safetensors"))
    expected = dict(tree_flatten(model.trainable_parameters()))
    wrong = sorted(set(saved) ^ set(expected)) + sorted(k for k in saved.keys() & expected.keys()
                                                         if saved[k].shape != expected[k].shape)
    if wrong:
        raise SystemExit(f"{folder}: its adapter does not match this model's ({len(wrong)} weights differ, "
                         f"such as {wrong[0]})")
    model.load_weights(list(saved.items()), strict=False)
    optimizer.state = tree_unflatten(list(mx.load(str(folder / "optimizer.safetensors")).items()))
    return int(folder.name[5:])


def _read_state(out):
    path = out / "recipe-state.json"
    return json.loads(path.read_text()) if path.exists() else None


def _write_state(out, state):
    path = out / "recipe-state.json"
    path.with_suffix(".tmp").write_text(json.dumps(state, indent=1) + "\n")
    os.replace(path.with_suffix(".tmp"), path)


def _record(name, data):
    """Keep a result in the run's out folder when running as a ModelLab job."""
    if run := os.environ.get("MODELLAB_RUN_DIR"):
        (Path(run) / "out" / name).write_text(json.dumps(data, indent=1) + "\n")
