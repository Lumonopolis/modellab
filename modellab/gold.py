"""Gold sets: a folder of items, their media, a task and the answers people gave.

<set>/task.toml        what an item shows (panes) and what is asked (questions); docs/gold-sets.md
<set>/items.jsonl      one item per line: an "id" and named fields
<set>/media/           files named by content: <sha256>.<ext>
<set>/labels.jsonl     answers, appended; the latest revision per item and labeller counts
<set>/versions/<id>/   what `freeze` kept: task.toml, items.jsonl, labels.jsonl, manifest.json

A media field is {"file": "<path>"} while a set is being built (relative to the set) and
{"media": "<sha256>.<ext>"} once frozen. Freezing clones files into media/, so a frozen
version never depends on files outside the set. Standard library only: this runs anywhere.
"""

import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import time
import tomllib
from pathlib import Path

DISPLAYS = {"image", "video", "audio", "text", "markdown", "chat", "fields", "diff"}
VISUAL = {"image", "video"}
KINDS = {"choice", "multi", "scale", "number", "text", "rank", "span", "region"}
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
MEDIA = re.compile(r"[0-9a-f]{64}\.[a-z0-9]{1,8}")
_TOP = {"title", "instructions", "layout", "pane", "question", "queue", "plugins"}
_PANE = {"field", "show", "label", "hidden", "key", "target", "against"}
_QUESTION = {"id", "kind", "prompt", "help", "required", "when", "unless", "default", "pane", "panes",
             "options", "range", "labels", "unit", "ties", "tags", "details"}
_QUEUE = {"order", "seed", "blind", "shuffle", "labellers", "auto_save"}
RESERVED = {"Enter", "Backspace", "Escape", "d", "s", "f", "?"}  # the page's own keys
VIDEO_KEYS = {",", ".", " "}  # the page's too, when boxes are drawn on a video
MARKS = {"span", "region"}


class Invalid(ValueError):
    """A gold set does not follow the format; the message says where."""


# Task

def load_task(folder):
    folder = Path(folder)
    try:
        raw = tomllib.loads((folder / "task.toml").read_text())
    except FileNotFoundError:
        raise Invalid(f"{folder}: no task.toml")
    except tomllib.TOMLDecodeError as error:
        raise Invalid(f"{folder / 'task.toml'}: {error}")
    return normalize_task(raw, folder)


def normalize_task(raw, folder):
    """The task with defaults filled in, or Invalid naming the first problem."""
    _keys(raw, _TOP, "task.toml")
    plugins = raw.get("plugins", {})
    displays = DISPLAYS | set(plugins.get("displays", {}))
    kinds = KINDS | set(plugins.get("kinds", {}))
    for kind, files in plugins.get("kinds", {}).items():
        if not isinstance(files, dict) or "page" not in files or set(files) - {"page", "check"}:
            raise Invalid(f"plugin kind {kind!r}: give {{page = \"<file>.js\", check = \"<file>.py\"}}")
        for file in files.values():
            _inside(folder, file, f"plugin {kind}")
    for display, file in plugins.get("displays", {}).items():
        _inside(folder, file, f"plugin {display}")

    panes, fields = [], set()
    for pane in raw.get("pane", []):
        _keys(pane, _PANE, "a [[pane]]")
        field = _need(pane, "field", str, "a [[pane]]")
        where = f"pane {field!r}"
        if field in fields:
            raise Invalid(f"{where} appears twice")
        if pane.get("show") not in displays:
            raise Invalid(f"{where}: show must be one of {sorted(displays)}")
        if pane.get("hidden") and not pane.get("key"):
            raise Invalid(f"{where} is hidden, so it needs a key to show it")
        if pane["show"] == "diff" and not isinstance(pane.get("against"), str):
            raise Invalid(f"{where}: a diff needs `against`, the field it is compared with")
        fields.add(field)
        panes.append({"label": field, "hidden": False, "key": None, "target": False, **pane})
    if not panes:
        raise Invalid("task.toml needs at least one [[pane]]")

    layout = {"arrange": "row", "compare": [], **raw.get("layout", {})}
    _keys(layout, {"arrange", "compare"}, "[layout]")
    if layout["arrange"] not in ("row", "column", "grid"):
        raise Invalid("[layout] arrange must be row, column or grid")
    _fields_of(layout["compare"], fields, "[layout] compare")

    questions, ids = [], set()
    for question in raw.get("question", []):
        questions.append(_question(question, fields, panes, ids, kinds))
        ids.add(questions[-1]["id"])
        if check := plugins.get("kinds", {}).get(questions[-1]["kind"], {}).get("check"):
            questions[-1]["plugin_check"] = str(_inside(folder, check, "plugin check"))
    for number, question in enumerate(questions):
        above = {earlier["id"] for earlier in questions[:number]}
        for rule in ("when", "unless"):
            # Only questions above: answers are checked in order, on the page and here alike.
            _condition(question[rule], above, f"question {question['id']!r} {rule} (only questions above it)")
        if len(question["panes"]) > 1 and question["kind"] != "rank" and any(o["key"] for o in question.get("options", [])):
            raise Invalid(f"question {question['id']!r}: option keys need a question about one pane")
    marked = [question["panes"][0] for question in questions if question["kind"] in MARKS]
    if twice := sorted({pane for pane in marked if marked.count(pane) > 1}):
        raise Invalid(f"panes {twice} each take one span or region question; tags mark different things")

    queue = {"order": "file", "seed": 0, "blind": False, "shuffle": [], "labellers": 1, "auto_save": {},
             **raw.get("queue", {})}
    _keys(queue, _QUEUE, "[queue]")
    if queue["order"] not in ("file", "random") and not str(queue["order"]).startswith("field:"):
        raise Invalid("[queue] order must be file, random or field:<name>")
    _fields_of(queue["shuffle"], fields, "[queue] shuffle")
    _shuffle(queue["shuffle"], panes, questions)
    if not isinstance(queue["labellers"], int) or queue["labellers"] < 1:
        raise Invalid("[queue] labellers must be a whole number of at least 1")
    _condition(queue["auto_save"], ids, "[queue] auto_save")
    if instructions := raw.get("instructions"):
        _inside(folder, instructions, "instructions")
    if queue["blind"]:
        _blind(panes, questions, queue)

    keys = [pane["key"] for pane in panes if pane["key"]]
    keys += [option["key"] for question in questions for option in question.get("options", [])
             + (question["tags"] if question["kind"] in MARKS else []) if option["key"]]
    shows = {pane["field"]: pane["show"] for pane in panes}
    reserved = RESERVED | (VIDEO_KEYS if any(question["kind"] == "region" and shows[question["panes"][0]] == "video"
                                             for question in questions) else set())
    if bad := sorted({str(key) for key in keys if not isinstance(key, str) or len(key) != 1 or key.lower() in reserved}):
        raise Invalid(f"keys {bad}: a key is one character, not one of the page's own {sorted(reserved)}")
    keys = [key.lower() for key in keys]  # the page reads keys without case, so Caps Lock changes nothing
    if duplicates := sorted({key for key in keys if keys.count(key) > 1}):
        raise Invalid(f"keys used twice: {duplicates}")
    return {"title": raw.get("title", folder.name), "instructions": raw.get("instructions"), "layout": layout,
            "panes": panes, "questions": questions, "queue": queue, "plugins": plugins}


def _shuffle(shuffled, panes, questions):
    """Shuffled panes are shown as A, B, ... in a random order; nothing may tell them apart."""
    if not shuffled:
        return
    group = [pane for pane in panes if pane["field"] in shuffled]
    if len({pane["show"] for pane in group}) > 1 or any(pane["hidden"] for pane in group):
        raise Invalid("[queue] shuffle: shuffled panes must show the same way and none may be hidden")
    for question in questions:
        covered = set(question["panes"]) & set(shuffled)
        if covered and covered != set(shuffled):
            raise Invalid(f"question {question['id']!r} is about some shuffled panes but not all, "
                          "which would tell them apart")


def _blind(panes, questions, queue):
    """Blind mode withholds target panes while labelling; nothing else may carry them."""
    targets = {pane["field"] for pane in panes if pane["target"]}
    for pane in panes:
        if pane["show"] == "diff" and pane.get("against") in targets:
            raise Invalid(f"pane {pane['field']!r} is a diff against the target {pane['against']!r}, which blind mode hides")
    if str(queue["order"]).removeprefix("field:") in targets:
        raise Invalid("[queue] order by a target field would reveal it; blind mode hides targets")
    for question in questions:
        if set(question["panes"]) & targets:
            raise Invalid(f"question {question['id']!r} is about a target pane, which blind mode hides")


def _question(raw, fields, panes, ids, kinds):
    _keys(raw, _QUESTION, "a [[question]]")
    qid = _need(raw, "id", str, "a [[question]]")
    where = f"question {qid!r}"
    if not ID.fullmatch(qid) or qid in ids:
        raise Invalid(f"{where}: ids are letters, digits, '_', '.' or '-', and unique")
    kind = raw.get("kind")
    if kind not in kinds:
        raise Invalid(f"{where}: kind must be one of {sorted(kinds)}")
    # No marks is an answer too, so a span or region is required only when it must have one.
    question = {"prompt": qid, "help": "", "required": kind not in {"text"} | MARKS, "when": {}, "unless": {}, "default": None,
                "details": kind == "text", **raw}
    if "pane" in raw and "panes" in raw:
        raise Invalid(f"{where}: give `pane` or `panes`, not both")
    question["panes"] = [raw["pane"]] if "pane" in raw else raw.get("panes", [])
    question.pop("pane", None)
    _fields_of(question["panes"], fields, f"{where} panes")
    shows = {pane["field"]: pane["show"] for pane in panes}
    if "tags" in raw and kind in KINDS - MARKS:
        raise Invalid(f"{where}: tags belong to span and region questions")
    if kind in ("choice", "multi"):
        options = raw.get("options")
        if not isinstance(options, list) or not options:
            raise Invalid(f"{where}: needs options")
        question["options"] = [_option(option, where) for option in options]
        values = [option["value"] for option in question["options"]]
        if len(set(values)) != len(values):
            raise Invalid(f"{where}: an option appears twice")
    elif kind == "scale":
        low, high = _range(raw, where, int, required=True)
        question["labels"] = {str(key): str(value) for key, value in raw.get("labels", {}).items()}
        if bad := [key for key in question["labels"] if not key.lstrip("-").isdigit() or not low <= int(key) <= high]:
            raise Invalid(f"{where}: labels {bad} are outside the range")
    elif kind == "number":
        _range(raw, where, (int, float), required=False)
    elif kind == "rank":
        if len(question["panes"]) < 2:
            raise Invalid(f"{where}: a rank needs at least two panes")
        question["ties"] = bool(raw.get("ties", False))
    elif kind in MARKS:
        wanted = {"text", "markdown"} if kind == "span" else VISUAL
        if len(question["panes"]) != 1 or shows[question["panes"][0]] not in wanted:
            raise Invalid(f"{where}: a {kind} marks one pane showing {' or '.join(sorted(wanted))}")
        tags = raw.get("tags", [])
        if not isinstance(tags, list):
            raise Invalid(f"{where}: tags are a list, like options")
        question["tags"] = [_option(tag, f"{where} tag") for tag in tags]
        values = [tag["value"] for tag in question["tags"]]
        if len(set(values)) != len(values):
            raise Invalid(f"{where}: a tag appears twice")
    return question


def _option(option, where):
    if isinstance(option, str):
        return {"value": option, "label": option, "key": None}
    if not isinstance(option, dict) or not isinstance(option.get("value"), str):
        raise Invalid(f"{where}: a name or a table with `value` (and `label`, `key`)")
    _keys(option, {"value", "label", "key"}, f"{where} option")
    return {"label": option["value"], "key": None, **option}


def _range(raw, where, types, required):
    bounds = raw.get("range")
    if bounds is None and not required:
        return None, None
    if (not isinstance(bounds, list) or len(bounds) != 2 or not all(isinstance(b, types) for b in bounds)
            or bounds[0] >= bounds[1]):
        raise Invalid(f"{where}: range is [lowest, highest]")
    return bounds


def _condition(rule, ids, where):
    """A condition is a table {question: [values]}: every question must have one of its values."""
    if not isinstance(rule, dict):
        raise Invalid(f"{where}: a condition is a table of question = [values]")
    for question, values in rule.items():
        if question not in ids or not isinstance(values, list):
            raise Invalid(f"{where}: {question!r} must name a question and list values")
        if any(isinstance(value, bool) for value in values):
            raise Invalid(f"{where}: {question!r} lists true or false; answers are options, numbers or text")


def _fields_of(names, fields, where):
    if not isinstance(names, list) or any(name not in fields for name in names):
        raise Invalid(f"{where}: every name must be a pane's field")


def _keys(table, allowed, where):
    if not isinstance(table, dict):
        raise Invalid(f"{where} must be a table")
    if unknown := sorted(set(table) - allowed):
        raise Invalid(f"{where}: unknown keys {unknown}")


def _need(table, key, kind, where):
    if not isinstance(table.get(key), kind):
        raise Invalid(f"{where} needs `{key}`")
    return table[key]


def _inside(folder, relative, where):
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise Invalid(f"{where}: {relative!r} must be a file inside the set")
    return path


# Answers

def holds(rule, answers, pane=None):
    """Whether a condition holds. For a question asked of several panes, `pane` picks its answer."""
    for question, values in rule.items():
        answer = answers.get(question)
        if isinstance(answer, dict):
            answer = answer.get(pane) if pane else None
        if answer not in values:
            return False
    return True


def asked(question, answers, pane=None):
    return holds(question["when"], answers, pane) and (
        not question["unless"] or not holds(question["unless"], answers, pane))


def per_pane(question):
    """A question about panes is answered for each pane, except a rank, which orders them."""
    return bool(question["panes"]) and question["kind"] != "rank"


def check_answers(task, answers, item=None):
    """Problems with one item's answers, as sentences; empty when they are complete and valid.

    Questions are taken in order. A question is asked when its conditions hold on the answers
    kept so far, so an answer to a question no longer asked is refused, not stored. With the
    item, a span's text is checked against the text it marks.
    """
    problems = []
    shows = {pane["field"]: pane["show"] for pane in task["panes"]}
    known = {question["id"] for question in task["questions"]}
    if unknown := sorted(set(answers) - known):
        problems.append(f"unknown questions {unknown}")
    kept = {}
    for question in task["questions"]:
        qid, several = question["id"], per_pane(question)
        targets = question["panes"] if several else [None]
        value = answers.get(qid)
        if several and value is not None and (not isinstance(value, dict) or set(value) - set(targets)):
            problems.append(f"{qid}: answer per pane, as {{pane: answer}}")
            continue
        for pane in targets:
            one = value.get(pane) if several and value else value
            where = qid + (f" for {pane}" if pane else "")
            if not asked(question, kept, pane):
                if one is not None:
                    problems.append(f"{where}: not asked, given the answers above it")
            elif one is None:
                if question["required"]:
                    problems.append(f"{where}: required")
            elif problem := _check_value(question, one, shows.get(pane), _UNKNOWN if item is None else item.get(pane)):
                problems.append(f"{where}: {problem}")
            elif several:
                kept.setdefault(qid, {})[pane] = one
            else:
                kept[qid] = one
    return problems


_UNKNOWN = object()  # the item was not given, so a span's text cannot be checked


def _check_value(question, value, display=None, field=_UNKNOWN):
    kind = question["kind"]
    values = [option["value"] for option in question.get("options", [])]
    if kind == "choice":
        return None if value in values else f"{value!r} is not an option"
    if kind == "multi":
        ok = isinstance(value, list) and all(v in values for v in value) and len(set(value)) == len(value)
        return None if ok else "a list of distinct options"
    if kind == "scale":
        low, high = question["range"]
        return None if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high \
            else f"a whole number from {low} to {high}"
    if kind == "number":
        low, high = question.get("range") or (None, None)
        if not _finite(value):
            return "a number"
        return None if low is None or low <= value <= high else f"a number from {low} to {high}"
    if kind == "text":
        return None if isinstance(value, str) else "text"
    if kind == "rank":
        groups = value if isinstance(value, list) else None
        flat = [pane for group in groups or [] if isinstance(group, list) for pane in group]
        if groups is None or sorted(flat) != sorted(question["panes"]) or len(groups) != sum(map(bool, groups)):
            return "every pane once, as a list of groups from best to worst"
        return None if question["ties"] or all(len(group) == 1 for group in groups) else "ties are not allowed"
    if kind in MARKS:
        return _marks(question, value, display, field)
    if path := question.get("plugin_check"):
        return _plugin(path).check(question, value)
    return None  # a plugin kind without a check part accepts what its page sends


_PLUGINS = {}


def _plugin(path):
    """A plugin kind's check part: a module with check(question, value) -> problem or None."""
    if path not in _PLUGINS:
        spec = importlib.util.spec_from_file_location(f"modellab_plugin_{len(_PLUGINS)}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PLUGINS[path] = module
    return _PLUGINS[path]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _marks(question, marks, display, text):
    """A span marks text by character offsets (code points), from start up to end, in order and
    without overlaps, and repeats the text it marks. A region is a box in fractions of the
    picture as shown; on a video it carries its time in seconds."""
    if not isinstance(marks, list):
        return "a list of marks"
    if question["kind"] == "span" and text is not _UNKNOWN and not isinstance(text, str):
        return "the item has no text here to mark"
    if question["required"] and not marks:
        return "mark at least one"
    tags = [tag["value"] for tag in question["tags"]]
    span = question["kind"] == "span"
    allowed = {"start", "end", "text"} if span else {"x", "y", "w", "h"} | ({"time"} if display == "video" else set())
    before = 0
    for number, mark in enumerate(marks, 1):
        where = f"mark {number}"
        if not isinstance(mark, dict) or set(mark) - allowed - {"tag"}:
            return f"{where}: a table of {', '.join(sorted(allowed))}" + (" and tag" if tags else "")
        if tags and mark.get("tag") not in tags:
            return f"{where}: tag must be one of {tags}"
        if not tags and "tag" in mark:
            return f"{where}: this question has no tags"
        if span:
            start, end = mark.get("start"), mark.get("end")
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end)) or not 0 <= start < end:
                return f"{where}: start and end are character offsets, 0 <= start < end"
            if start < before:
                return f"{where}: marks are in order and do not overlap"
            before = end
            if not isinstance(mark.get("text"), str):
                return f"{where}: text is the text it marks"
            if text is not _UNKNOWN and (end > len(text) or text[start:end] != mark["text"]):
                return f"{where}: {mark['text']!r} is not the text at {start}..{end}"
        else:
            x, y, w, h = (mark.get(key) for key in ("x", "y", "w", "h"))
            if not all(map(_finite, (x, y, w, h))) or x < 0 or y < 0 or w <= 0 or h <= 0 \
                    or x + w > 1 + 1e-6 or y + h > 1 + 1e-6:
                return f"{where}: x, y, w and h are fractions of the picture, and the box is inside it"
            if display == "video" and not (_finite(mark.get("time")) and mark["time"] >= 0):
                return f"{where}: a box on a video needs its time in seconds"
    return None


# Items and labels

def read_items(path):
    items, seen = [], set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise Invalid(f"{path}:{number}: {error}")
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not ID.fullmatch(item["id"]):
            raise Invalid(f"{path}:{number}: an item needs an id of letters, digits, '_', '.' or '-'")
        if item["id"] in seen:
            raise Invalid(f"{path}:{number}: id {item['id']!r} appears twice")
        seen.add(item["id"])
        items.append(item)
    return items


def read_labels(path):
    """Every label record, in the order written."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise Invalid(f"{path}:{number}: {error}")
    return records


def latest(records):
    """{(item, labeller): record} for the newest revision of each."""
    newest = {}
    for record in records:
        key = (record["item"], record["labeller"])
        if key not in newest or record["revision"] > newest[key]["revision"]:
            newest[key] = record
    return newest


def media_path(folder, value):
    """The file a media field names, or None when the field is not media."""
    if not isinstance(value, dict):
        return None
    if "media" in value:
        if not MEDIA.fullmatch(str(value["media"])):
            raise Invalid(f"media name {value['media']!r} is not <sha256>.<ext>")
        return Path(folder) / "media" / value["media"]
    if "file" in value:
        path = Path(value["file"])
        return path if path.is_absolute() else Path(folder) / path
    return None


# The whole set

def open_set(spec):
    """(set folder, folder holding this version's files) for "<folder>" or "<folder>@<id>"."""
    folder, _, version = str(spec).partition("@")
    folder = Path(folder)
    if not version:
        return folder, folder
    matches = sorted((folder / "versions").glob(f"{version}*")) if len(version) >= 6 else []
    if len(matches) != 1:
        raise Invalid(f"{spec}: no single frozen version starts with {version!r}")
    return folder, matches[0]


def check(spec, outputs=()):
    """Problems in a gold set (and outputs joined to it), as sentences. Empty means usable."""
    folder, files = open_set(spec)
    try:
        task = load_task(files)
        items = read_items(files / "items.jsonl")
        records = read_labels(files / "labels.jsonl")
    except (Invalid, FileNotFoundError) as error:
        return [str(error)]
    joined = join(items, [read_items(path) for path in outputs], [Path(path).parent for path in outputs])
    problems = []
    for item in joined:
        for pane in task["panes"]:
            for field in [pane["field"]] + ([pane["against"]] if pane["show"] == "diff" else []):
                if field not in item:
                    problems.append(f"item {item['id']}: no field {field!r} for its pane")
                elif (path := media_path(folder, item[field])) is not None and not path.is_file():
                    problems.append(f"item {item['id']}: {field} names a missing file {path}")
    by_id = {item["id"]: item for item in items}
    for number, record in enumerate(records, 1):
        if not _record_shape(record):
            problems.append(f"labels line {number}: a record has item, labeller, revision (from 1), "
                            "and answers or skip")
        elif record.get("item") not in by_id:
            problems.append(f"labels: answer for unknown item {record.get('item')!r}")
        elif not record.get("skip"):
            problems += [f"labels: {record['item']} by {record['labeller']} r{record['revision']}: {problem}"
                         for problem in check_answers(task, record.get("answers", {}), by_id[record["item"]])]
    return problems


def _record_shape(record):
    return (isinstance(record, dict) and isinstance(record.get("item"), str)
            and isinstance(record.get("labeller"), str) and isinstance(record.get("revision"), int)
            and record["revision"] >= 1 and (record.get("skip") is True or isinstance(record.get("answers"), dict)))


def join(items, outputs, bases):
    """Items with each output's fields added by id. An output file names its media relative to
    itself; those paths become absolute here."""
    by_id = {item["id"]: dict(item) for item in items}
    for rows, base in zip(outputs, bases):
        for row in rows:
            if row["id"] not in by_id:
                raise Invalid(f"output for unknown item {row['id']!r}")
            for field, value in row.items():
                if field != "id":
                    if isinstance(value, dict) and "file" in value and not Path(value["file"]).is_absolute():
                        value = {"file": str((base / value["file"]).resolve())}
                    by_id[row["id"]][field] = value
    return list(by_id.values())


def freeze(folder):
    """Keep the set as it is now under versions/<id>/ and return the id. Files named by path are
    cloned into media/ first, so the version stands on its own; identical content keeps its id."""
    folder = Path(folder)
    if problems := check(folder):
        raise Invalid("; ".join(problems[:5]) + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""))
    items = read_items(folder / "items.jsonl")
    for item in items:
        for field, value in list(item.items()):
            if isinstance(value, dict) and "file" in value:
                item[field] = {"media": _store(folder, media_path(folder, value))}
    _write(folder / "items.jsonl", "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
    task = load_task(folder)
    names = ["task.toml", "items.jsonl"] + [name for name in ("labels.jsonl",) if (folder / name).exists()]
    extra = [task["instructions"]] if task["instructions"] else []
    for files in task["plugins"].get("kinds", {}).values():
        extra += list(files.values())
    extra += list(task["plugins"].get("displays", {}).values())
    media = sorted({value["media"] for item in items for value in item.values()
                    if isinstance(value, dict) and "media" in value})
    content = {"format": 1, "files": {name: _sha(folder / name) for name in names + extra}, "media": media}
    version = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    target = folder / "versions" / version
    if not target.exists():
        staging = folder / "versions" / f".{version}.{os.getpid()}"
        for name in names + extra:
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            (staging / name).write_bytes((folder / name).read_bytes())
        _write(staging / "manifest.json", json.dumps({**content, "id": version, "frozen": time.time()}, indent=1))
        os.replace(staging, target)
    return version


def _store(folder, source):
    digest = _sha(source)
    name = f"{digest}{source.suffix.lower()}" if source.suffix else f"{digest}.bin"
    target = folder / "media" / name
    if not target.exists():
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_name(f".{name}.{os.getpid()}")
        # An APFS clone takes no space and does not change when the original does.
        if subprocess.run(["cp", "-c", str(source), str(temporary)], capture_output=True).returncode != 0:
            subprocess.run(["cp", str(source), str(temporary)], check=True)
        os.replace(temporary, target)
    return name


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)
