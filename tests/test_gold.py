"""Gold sets: the task file, answers, checks and frozen versions."""

import json
import subprocess
import sys

import pytest

from conftest import CLI
from modellab import gold

TASK = '''
title = "Edit feedback"
instructions = "guide.md"

[layout]
compare = ["model_edit", "user_edit"]

[[pane]]
field = "default_render"
show = "image"
hidden = true
key = "o"

[[pane]]
field = "model_edit"
show = "image"

[[pane]]
field = "user_edit"
show = "image"
target = true

[[pane]]
field = "caption"
show = "text"

[[question]]
id = "verdict"
kind = "choice"
options = ["keep", "close", "rework", {value = "not mine", key = "n"}]
panes = ["model_edit"]

[[question]]
id = "issues"
kind = "multi"
options = ["crop", "tone"]
panes = ["model_edit"]
unless = {verdict = ["keep"]}

[[question]]
id = "quality"
kind = "scale"
range = [1, 5]
labels = {1 = "unusable", 5 = "excellent"}

[[question]]
id = "better"
kind = "rank"
panes = ["model_edit", "user_edit"]
ties = true

[[question]]
id = "note"
kind = "text"

[queue]
order = "random"
auto_save = {quality = [5]}
'''


def make_set(tmp_path, task=TASK, items=None):
    folder = tmp_path / "set"
    (folder / "sources").mkdir(parents=True)
    (folder / "task.toml").write_text(task)
    (folder / "guide.md").write_text("Judge the edit against the photographer's own.\n")
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (folder / "sources" / name).write_bytes(name.encode() * 100)
    items = items or [{"id": "p1", "default_render": {"file": "sources/a.jpg"},
                       "model_edit": {"file": "sources/b.jpg"}, "user_edit": {"file": "sources/c.jpg"},
                       "caption": "Enoshima beach"}]
    (folder / "items.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items))
    return folder


def label(folder, answers, item="p1", labeller="rater1", revision=1):
    with open(folder / "labels.jsonl", "a") as handle:
        handle.write(json.dumps({"item": item, "labeller": labeller, "revision": revision,
                                 "at": "2026-09-30T00:00:00", "answers": answers}) + "\n")


def test_a_task_gets_its_defaults(tmp_path):
    task = gold.load_task(make_set(tmp_path))
    verdict, _, quality, better, note = task["questions"]
    assert [pane["field"] for pane in task["panes"]] == ["default_render", "model_edit", "user_edit", "caption"]
    assert task["panes"][0]["label"] == "default_render" and task["panes"][2]["target"]
    assert verdict["options"][3] == {"value": "not mine", "label": "not mine", "key": "n"}
    assert (verdict["required"], note["required"]) == (True, False)
    assert better["ties"] and quality["labels"] == {"1": "unusable", "5": "excellent"}
    assert task["queue"]["labellers"] == 1 and task["layout"]["arrange"] == "row"


@pytest.mark.parametrize("change, message", [
    (('show = "image"\nhidden = true\nkey = "o"', 'show = "image"\nhidden = true'), "needs a key"),
    (('key = "n"', 'key = "o"'), "keys used twice"),
    (('key = "n"', 'key = "s"'), "page's own"),
    (('kind = "rank"', 'kind = "sorting"'), "kind must be one of"),
    (('panes = ["model_edit", "user_edit"]', 'panes = ["model_edit"]'), "at least two panes"),
    (('unless = {verdict = ["keep"]}', 'unless = {verdikt = ["keep"]}'), "must name a question"),
    (('show = "text"', 'show = "picture"'), "show must be one of"),
    (('instructions = "guide.md"', 'instructions = "../guide.md"'), "inside the set"),
    (('title = "Edit feedback"', 'title = "Edit feedback"\ncolour = "blue"'), "unknown keys"),
])
def test_task_mistakes_are_named(tmp_path, change, message):
    with pytest.raises(gold.Invalid, match=message):
        gold.load_task(make_set(tmp_path, TASK.replace(*change)))


@pytest.mark.parametrize("extra, message", [
    ('[[pane]]\nfield = "changes"\nshow = "diff"\nagainst = "user_edit"\n', "diff against the target"),
    ('', "about a target pane"),
])
def test_blind_mode_refuses_what_would_show_the_target(tmp_path, extra, message):
    task = TASK.replace('[queue]\norder = "random"', '[queue]\norder = "random"\nblind = true')
    if extra:
        task = task.replace('panes = ["model_edit", "user_edit"]\nties = true', 'panes = ["model_edit", "caption"]\nties = true')
        task = task.replace("[[question]]", extra + "\n[[question]]", 1)
    with pytest.raises(gold.Invalid, match=message):
        gold.load_task(make_set(tmp_path, task))
    by_target = TASK.replace('panes = ["model_edit", "user_edit"]\nties = true', 'panes = ["model_edit", "caption"]\nties = true')
    by_target = by_target.replace('[queue]\norder = "random"', '[queue]\norder = "field:user_edit"\nblind = true')
    with pytest.raises(gold.Invalid, match="order by a target field"):
        gold.load_task(make_set(tmp_path / "order", by_target))


def test_conditions_name_only_questions_above_and_keys_ignore_case(tmp_path):
    later = TASK.replace('panes = ["model_edit"]\nunless = {verdict = ["keep"]}', 'panes = ["model_edit"]\nunless = {quality = [5]}')
    with pytest.raises(gold.Invalid, match="only questions above it"):
        gold.load_task(make_set(tmp_path, later))
    shouting = TASK.replace('key = "n"', 'key = "O"')
    with pytest.raises(gold.Invalid, match="keys used twice"):
        gold.load_task(make_set(tmp_path / "case", shouting))
    two_panes = TASK.replace('options = ["keep", "close", "rework", {value = "not mine", key = "n"}]\npanes = ["model_edit"]',
                             'options = ["keep", "close", "rework", {value = "not mine", key = "n"}]\npanes = ["model_edit", "user_edit"]')
    with pytest.raises(gold.Invalid, match="option keys need a question about one pane"):
        gold.load_task(make_set(tmp_path / "keys", two_panes))


def test_answers_are_checked_in_order_so_a_chain_of_conditions_holds(tmp_path):
    chain = TASK + '''
[[question]]
id = "why"
kind = "choice"
options = ["colour", "light"]
panes = ["model_edit"]
when = {issues = [["tone"]]}
'''
    task = gold.load_task(make_set(tmp_path, chain))
    base = {"quality": 3, "better": [["model_edit"], ["user_edit"]]}
    rework = {**base, "verdict": {"model_edit": "rework"}, "issues": {"model_edit": ["tone"]}, "why": {"model_edit": "light"}}
    assert gold.check_answers(task, rework) == []
    kept = {**base, "verdict": {"model_edit": "keep"}, "issues": {"model_edit": ["tone"]}, "why": {"model_edit": "light"}}
    assert gold.check_answers(task, kept) == ["issues for model_edit: not asked, given the answers above it",
                                              "why for model_edit: not asked, given the answers above it"]


def test_marking_kinds_need_the_right_pane_and_one_per_pane(tmp_path):
    span_on_image = TASK + '\n[[question]]\nid = "odd"\nkind = "span"\npane = "model_edit"\n'
    with pytest.raises(gold.Invalid, match="marks one pane showing markdown or text"):
        gold.load_task(make_set(tmp_path, span_on_image))
    span_on_text = TASK + '\n[[question]]\nid = "names"\nkind = "span"\npane = "caption"\ntags = ["place", {value = "person", key = "p"}]\n'
    names = gold.load_task(make_set(tmp_path / "ok", span_on_text))["questions"][-1]
    assert names["tags"] == [{"value": "place", "label": "place", "key": None}, {"value": "person", "label": "person", "key": "p"}]
    assert names["required"] is False  # no marks is an answer too
    with pytest.raises(gold.Invalid, match="keys used twice"):
        gold.load_task(make_set(tmp_path / "keys", span_on_text.replace('key = "p"', 'key = "N"')))
    again = span_on_text + '\n[[question]]\nid = "more"\nkind = "span"\npane = "caption"\n'
    with pytest.raises(gold.Invalid, match="each take one span or region question"):
        gold.load_task(make_set(tmp_path / "twice", again))


MARKED = TASK + '''
[[pane]]
field = "clip"
show = "video"

[[question]]
id = "names"
kind = "span"
pane = "caption"
tags = ["place", "person"]

[[question]]
id = "faces"
kind = "region"
pane = "model_edit"
required = true

[[question]]
id = "moments"
kind = "region"
pane = "clip"
'''


def test_marks_are_checked_against_the_item(tmp_path):
    folder = make_set(tmp_path, MARKED)
    (folder / "sources" / "clip.mp4").write_bytes(b"video")
    task = gold.load_task(folder)
    item = {"id": "p1", "caption": "Enoshima 🌊 beach", "clip": {"file": "sources/clip.mp4"}}
    base = {"verdict": {"model_edit": "keep"}, "quality": 4, "better": [["model_edit"], ["user_edit"]]}
    face = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    # Offsets count code points, as Python does: the wave is one character.
    names = [{"start": 0, "end": 8, "text": "Enoshima", "tag": "place"}, {"start": 11, "end": 16, "text": "beach", "tag": "place"}]
    good = {**base, "names": {"caption": names}, "faces": {"model_edit": [face]},
            "moments": {"clip": [{**face, "time": 1.25}]}}
    assert gold.check_answers(task, good, item) == []
    assert gold.check_answers(task, {**good, "names": {"caption": []}, "moments": {"clip": []}}, item) == []

    def problem(question, pane, marks):
        return gold.check_answers(task, {**good, question: {pane: marks}}, item)
    assert problem("names", "caption", [{**names[0], "text": "Enoshim"}]) == \
        ["names for caption: mark 1: 'Enoshim' is not the text at 0..8"]
    assert problem("names", "caption", [names[1], names[0]]) == \
        ["names for caption: mark 2: marks are in order and do not overlap"]
    assert problem("names", "caption", [{**names[0], "tag": "boat"}]) == \
        ["names for caption: mark 1: tag must be one of ['place', 'person']"]
    assert problem("names", "caption", [{"start": 5, "end": 2, "text": "", "tag": "place"}]) == \
        ["names for caption: mark 1: start and end are character offsets, 0 <= start < end"]
    assert problem("faces", "model_edit", []) == ["faces for model_edit: mark at least one"]
    assert problem("faces", "model_edit", [{**face, "x": 0.9}]) == \
        ["faces for model_edit: mark 1: x, y, w and h are fractions of the picture, and the box is inside it"]
    assert problem("faces", "model_edit", [{**face, "w": float("nan")}]) == \
        ["faces for model_edit: mark 1: x, y, w and h are fractions of the picture, and the box is inside it"]
    assert problem("faces", "model_edit", [{**face, "time": 2}]) == ["faces for model_edit: mark 1: a table of h, w, x, y"]
    assert problem("faces", "model_edit", [{**face, "tag": "eye"}]) == ["faces for model_edit: mark 1: this question has no tags"]
    assert problem("moments", "clip", [face]) == ["moments for clip: mark 1: a box on a video needs its time in seconds"]
    assert problem("names", "caption", [{**names[1], "end": 500}]) == \
        ["names for caption: mark 1: 'beach' is not the text at 11..500"]
    assert problem("names", "caption", [{"start": 100, "end": 101, "text": "", "tag": "place"}]) == \
        ["names for caption: mark 1: '' is not the text at 100..101"]
    assert gold.check_answers(task, good, {**item, "caption": None}) == \
        ["names for caption: the item has no text here to mark"]
    label(folder, {**good, "names": {"caption": [{**names[0], "text": "Kamakura"}]}})
    assert any("'Kamakura' is not the text at 0..8" in line for line in gold.check(folder))


def test_tags_belong_to_marks_and_video_keys_are_reserved_only_with_a_region_video(tmp_path):
    with pytest.raises(gold.Invalid, match="tags belong to span and region questions"):
        gold.load_task(make_set(tmp_path, TASK.replace('options = ["crop", "tone"]', 'options = ["crop", "tone"]\ntags = ["x"]')))
    dotted = TASK.replace('key = "n"', 'key = "."')
    assert gold.load_task(make_set(tmp_path / "dot", dotted))["questions"][0]["options"][3]["key"] == "."
    with pytest.raises(gold.Invalid, match="page's own"):
        gold.load_task(make_set(tmp_path / "video", dotted + MARKED.removeprefix(TASK)))


def test_plugin_answers_are_checked_by_the_plugin(tmp_path):
    task_text = TASK + '''
[[question]]
id = "mood"
kind = "mood"
required = false

[plugins]
kinds = {mood = {page = "mood.js", check = "mood.py"}}
'''
    folder = make_set(tmp_path, task_text)
    (folder / "mood.js").write_text("// the page part\n")
    (folder / "mood.py").write_text("def check(question, value):\n    return None if value in ('calm', 'tense') else 'calm or tense'\n")
    task = gold.load_task(folder)
    base = {"verdict": {"model_edit": "keep"}, "quality": 4, "better": [["model_edit"], ["user_edit"]]}
    assert gold.check_answers(task, {**base, "mood": "calm"}) == []
    assert gold.check_answers(task, {**base, "mood": "angry"}) == ["mood: calm or tense"]


def test_a_number_is_finite(tmp_path):
    task = gold.load_task(make_set(tmp_path, TASK + '\n[[question]]\nid = "minutes"\nkind = "number"\nrequired = false\n'))
    base = {"verdict": {"model_edit": "keep"}, "quality": 4, "better": [["model_edit"], ["user_edit"]]}
    assert gold.check_answers(task, {**base, "minutes": 2.5}) == []
    assert gold.check_answers(task, {**base, "minutes": float("inf")}) == ["minutes: a number"]


def test_answers_are_checked_by_kind_and_condition(tmp_path):
    task = gold.load_task(make_set(tmp_path))
    good = {"verdict": {"model_edit": "rework"}, "issues": {"model_edit": ["tone"]}, "quality": 3,
            "better": [["user_edit"], ["model_edit"]], "note": "too cool"}
    assert gold.check_answers(task, good) == []
    keep = {"verdict": {"model_edit": "keep"}, "quality": 5, "better": [["model_edit", "user_edit"]]}
    assert gold.check_answers(task, keep) == []  # issues are not asked after a keep; ties are allowed
    assert gold.check_answers(task, {**keep, "verdict": {"model_edit": "rework"}}) == \
        ["issues for model_edit: required"]
    problems = gold.check_answers(task, {"verdict": {"model_edit": "maybe"}, "quality": 7,
                                         "better": [["model_edit"]], "mystery": 1})
    assert problems == ["unknown questions ['mystery']", "verdict for model_edit: 'maybe' is not an option",
                        "issues for model_edit: required", "quality: a whole number from 1 to 5",
                        "better: every pane once, as a list of groups from best to worst"]
    assert gold.check_answers(task, {**good, "verdict": "keep"}) == \
        ["verdict: answer per pane, as {pane: answer}"]


def test_check_reports_missing_fields_files_and_bad_labels(tmp_path):
    folder = make_set(tmp_path, items=[{"id": "p1", "default_render": {"file": "sources/a.jpg"},
                                        "model_edit": {"file": "sources/gone.jpg"}, "caption": "x"}])
    label(folder, {"quality": 9})
    label(folder, {"quality": 3}, item="p9")
    problems = gold.check(folder)
    assert "item p1: no field 'user_edit' for its pane" in problems
    assert any("model_edit names a missing file" in problem for problem in problems)
    assert any(problem.startswith("labels: p1 by rater1 r1: quality: a whole number") for problem in problems)
    assert "labels: answer for unknown item 'p9'" in problems
    with open(folder / "labels.jsonl", "a") as handle:
        handle.write(json.dumps({"item": "p1", "answers": {}}) + "\n")
    assert any(problem.startswith("labels line 3: a record has") for problem in gold.check(folder))


def test_outputs_join_by_id_with_media_relative_to_themselves(tmp_path):
    folder = make_set(tmp_path, items=[{"id": "p1", "default_render": {"file": "sources/a.jpg"},
                                        "user_edit": {"file": "sources/c.jpg"}, "caption": "x"}])
    run = tmp_path / "run" / "out"
    (run / "renders").mkdir(parents=True)
    (run / "renders" / "p1.jpg").write_bytes(b"model")
    (run / "outputs.jsonl").write_text(json.dumps({"id": "p1", "model_edit": {"file": "renders/p1.jpg"}}) + "\n")
    assert gold.check(folder) == ["item p1: no field 'model_edit' for its pane"]
    assert gold.check(folder, [run / "outputs.jsonl"]) == []


def test_freeze_keeps_a_version_that_stands_on_its_own(tmp_path):
    folder = make_set(tmp_path)
    label(folder, {"verdict": {"model_edit": "keep"}, "quality": 5, "better": [["model_edit", "user_edit"]]})
    first = gold.freeze(folder)
    assert gold.freeze(folder) == first  # the same content keeps its id
    item = gold.read_items(folder / "items.jsonl")[0]
    assert gold.MEDIA.fullmatch(item["model_edit"]["media"])
    manifest = json.loads((folder / "versions" / first / "manifest.json").read_text())
    assert manifest["id"] == first and "guide.md" in manifest["files"] and len(manifest["media"]) == 3
    for source in (folder / "sources").iterdir():
        source.unlink()  # the frozen version needs none of the original files
    assert gold.check(f"{folder}@{first[:8]}") == []
    label(folder, {"verdict": {"model_edit": "close"}, "issues": {"model_edit": ["tone"]}, "quality": 3,
                   "better": [["user_edit"], ["model_edit"]]}, revision=2)
    second = gold.freeze(folder)
    assert second != first and sorted(p.name for p in (folder / "versions").iterdir()) == sorted([first, second])
    newest = gold.latest(gold.read_labels(folder / "labels.jsonl"))
    assert newest[("p1", "rater1")]["revision"] == 2


def test_a_set_with_problems_is_not_frozen(tmp_path):
    folder = make_set(tmp_path)
    label(folder, {"quality": 9})
    with pytest.raises(gold.Invalid, match="quality"):
        gold.freeze(folder)
    assert not (folder / "versions").exists()


def test_the_gold_commands(tmp_path):
    folder = make_set(tmp_path)
    ok = subprocess.run([sys.executable, CLI, "gold", "check", str(folder)], capture_output=True, text=True)
    assert ok.returncode == 0 and "no problems" in ok.stdout
    frozen = subprocess.run([sys.executable, CLI, "gold", "freeze", str(folder)], capture_output=True, text=True)
    assert frozen.returncode == 0 and frozen.stdout.strip().startswith(f"{folder}@")
    label(folder, {"quality": 0})
    bad = subprocess.run([sys.executable, CLI, "gold", "check", str(folder)], capture_output=True, text=True)
    assert bad.returncode == 1 and "quality" in bad.stdout


@pytest.mark.parametrize("queue, message", [
    ('shuffle = ["default_render", "model_edit"]', "none may be hidden"),
    ('shuffle = ["model_edit", "caption"]', "show the same way"),
    ('shuffle = ["model_edit", "user_edit"]', "some shuffled panes but not all"),
])
def test_shuffled_panes_cannot_be_told_apart(tmp_path, queue, message):
    task = TASK.replace('[queue]\norder = "random"', f'[queue]\norder = "random"\n{queue}')
    with pytest.raises(gold.Invalid, match=message):
        gold.load_task(make_set(tmp_path, task))


def test_conditions_do_not_take_true_or_false(tmp_path):
    task = TASK.replace('auto_save = {quality = [5]}', 'auto_save = {quality = [true]}')
    with pytest.raises(gold.Invalid, match="true or false"):
        gold.load_task(make_set(tmp_path, task))
