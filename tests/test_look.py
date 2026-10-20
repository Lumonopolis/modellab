"""`modellab look` end to end: sheets and sheets.md from a gold set with outputs. Needs an
installed environment (Pillow and OpenCV), like tests/test_mlx.py:

    MODELLAB_TEST_MLX_ENV=<root>/envs/mlx python3 -m pytest tests/test_look.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import CLI

MLX_ENV = os.environ.get("MODELLAB_TEST_MLX_ENV")
pytestmark = pytest.mark.skipif(not MLX_ENV, reason="set MODELLAB_TEST_MLX_ENV to an installed mlx environment")

TASK = '''
title = "Edit check"

[[pane]]
field = "default"
show = "image"
hidden = true
key = "o"

[[pane]]
field = "model"
show = "image"

[[pane]]
field = "target"
show = "image"
target = true

[[pane]]
field = "clip"
show = "video"

[[pane]]
field = "caption"
show = "text"

[[pane]]
field = "reply"
show = "diff"
against = "caption"

[[question]]
id = "verdict"
kind = "choice"
options = ["keep", "rework"]
panes = ["model"]

[[question]]
id = "flaws"
kind = "region"
pane = "default"
tags = ["dust", "scratch"]
'''

VIDEO = ("import cv2, numpy, sys\n"
         "out = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))\n"
         "for f in range(20): out.write(numpy.full((120, 160, 3), (f * 10, 80, 90), numpy.uint8))\n"
         "out.release()\n")


@pytest.fixture
def lab_with_set(lab, tmp_path):
    from PIL import Image
    (lab.root / "envs").mkdir(parents=True)
    (lab.root / "envs" / "mlx").symlink_to(Path(MLX_ENV))
    folder = tmp_path / "set"
    (folder / "sources").mkdir(parents=True)
    (folder / "task.toml").write_text(TASK)
    subprocess.run([str(Path(MLX_ENV) / "bin" / "python"), "-c", VIDEO, str(folder / "sources" / "clip.mp4")],
                   check=True, capture_output=True)
    run = tmp_path / "run" / "out"
    (run / "renders").mkdir(parents=True)
    items, outputs = [], []
    for index in range(9):
        for name, where in (("default", folder / "sources"), ("target", folder / "sources"), ("model", run / "renders")):
            Image.new("RGB", (300, 200), (20 * index, 100, 150)).save(where / f"{name}-{index}.png")
        item = {"id": f"p{index}", "default": {"file": f"sources/default-{index}.png"},
                "target": {"file": f"sources/target-{index}.png"}, "clip": {"file": "sources/clip.mp4"},
                "caption": f"scene {index}"}
        if index == 8:
            del item["target"]  # drawn as missing, not a crash
        items.append(item)
        outputs.append({"id": f"p{index}", "model": {"file": f"renders/model-{index}.png"},
                        "reply": f"scene {index}, warmer", "distance": index / 10})
    (folder / "items.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items))
    (run / "outputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in outputs))
    with open(folder / "labels.jsonl", "w") as handle:
        handle.write(json.dumps({"item": "p8", "labeller": "rater1", "revision": 1, "at": "t",
                                 "answers": {"verdict": {"model": "rework"}}}) + "\n")
        handle.write(json.dumps({"item": "p7", "labeller": "rater1", "revision": 1, "at": "t", "answers": {
            "verdict": {"model": "keep"}, "flaws": {"default": [{"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "tag": "scratch"}]}}}) + "\n")
    return folder, run / "outputs.jsonl", tmp_path


def look(*args, cwd=None):
    return subprocess.run([sys.executable, CLI, "look", *map(str, args)], capture_output=True, text=True, cwd=cwd)


def test_sheets_show_items_as_rows_and_the_rest_as_text(lab_with_set):
    folder, outputs, tmp = lab_with_set
    result = look(folder, "--outputs", outputs, "--highest", "distance", "--sample", "9", "--to", tmp / "look")
    assert result.returncode == 0, result.stderr
    from PIL import Image
    sheets = sorted((tmp / "look").glob("sheet-*.png"))
    assert len(sheets) == 2 and Image.open(sheets[0]).size == (1600, 1600)
    index = (tmp / "look" / "sheets.md").read_text()
    assert "the 9 items with the highest distance" in index
    assert index.index("## p8") < index.index("## p7") < index.index("## p0")
    assert "Sheet `sheet-01.png`, row 1." in index.split("## p8")[1].split("## p7")[0]
    assert "+scene 7, warmer" in index and "| distance | 0.7 |" in index
    assert 'Label by rater1 (revision 1): {"verdict": {"model": "rework"}}' in index
    # The box drawn on p7's default render is outlined in its tag's colour, as on the page (the second tag's blue).
    blue = sum(1 for pixel in Image.open(sheets[0]).convert("RGB").get_flattened_data() if pixel == (0x19, 0x71, 0xc2))
    assert blue > 500, blue


def test_a_random_sample_is_repeatable_by_its_seed(lab_with_set):
    folder, outputs, tmp = lab_with_set
    for name in ("one", "two"):
        assert look(folder, "--outputs", outputs, "--sample", "4", "--seed", "7", "--to", tmp / name).returncode == 0
    first, second = ((tmp / name / "sheets.md").read_text().split("- Made")[0] for name in ("one", "two"))
    assert first == second and "a random sample of 4 of 9 items, seed 7" in first


def test_look_refuses_what_it_cannot_draw(lab_with_set):
    folder, outputs, tmp = lab_with_set
    narrow = look(folder, "--outputs", outputs, "--size", "700x1600", "--to", tmp / "narrow")
    assert narrow.returncode != 0 and "cells need at least 200 px" in narrow.stderr
    (tmp / "taken").mkdir()
    taken = look(folder, "--outputs", outputs, "--to", tmp / "taken")
    assert taken.returncode != 0 and "exists" in taken.stderr
    broken = folder.parent / "broken"
    broken.mkdir()
    (broken / "task.toml").write_text("title = 'x'\n")
    missing = look(broken, "--to", tmp / "broken-look")
    assert missing.returncode != 0 and "needs at least one [[pane]]" in missing.stderr


def test_the_default_folder_is_new_and_in_the_current_directory(lab_with_set):
    folder, outputs, tmp = lab_with_set
    result = look(folder, "--outputs", outputs, "--sample", "2", cwd=tmp)
    assert result.returncode == 0, result.stderr
    [made] = tmp.glob("look-set-*")
    assert (made / "sheets.md").exists()


def test_rows_fill_the_sheet_and_the_sheet_names_its_text_panes(lab_with_set):
    """Two 3:2 panes leave room for 2.8 rows at full size: the pictures shrink a little to fit three."""
    from PIL import Image
    folder, _, tmp = lab_with_set
    wide = folder.parent / "wide"
    (wide / "sources").mkdir(parents=True)
    (wide / "task.toml").write_text('[[pane]]\nfield = "before"\nshow = "image"\n\n[[pane]]\nfield = "after"\nshow = "image"\n\n'
                                    '[[pane]]\nfield = "settings"\nshow = "fields"\nlabel = "Settings"\n')
    for name in ("before", "after"):
        Image.new("RGB", (1500, 1000), (200, 90, 40)).save(wide / "sources" / f"{name}.png")
    rows = [{"id": f"w{n}", "before": {"file": "sources/before.png"}, "after": {"file": "sources/after.png"},
             "settings": {"exposure": 0.3}} for n in range(3)]
    (wide / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = look(wide, "--sample", "3", "--to", tmp / "wide-look")
    assert result.returncode == 0, result.stderr
    assert [path.name for path in sorted((tmp / "wide-look").glob("sheet-*.png"))] == ["sheet-01.png"]
    sheet = Image.open(tmp / "wide-look" / "sheet-01.png").convert("RGB")
    orange = [y for y in range(0, 1600, 4) if sheet.getpixel((400, y)) == (200, 90, 40)]
    assert orange[-1] > 1400, orange[-1]  # the third row reaches the bottom of the sheet
    assert (tmp / "wide-look" / "sheets.md").read_text().count("**Settings**") == 3


BLIND = '''
title = "Which edit"

[[pane]]
field = "a"
show = "image"

[[pane]]
field = "b"
show = "image"

[[pane]]
field = "ref"
show = "image"

[[pane]]
field = "truth"
show = "text"
target = true

[[pane]]
field = "note"
show = "text"

[[question]]
id = "flaws"
kind = "region"
pane = "ref"
required = false

[queue]
shuffle = ["a", "b"]
'''


def test_blind_sheets_withhold_the_target_and_letter_the_shuffled_panes(lab_with_set):
    """An agent judge sees what a person labelling blind sees; the key stays outside its folder."""
    from PIL import Image
    folder, _, tmp = lab_with_set
    blind = folder.parent / "blind"
    (blind / "sources").mkdir(parents=True)
    (blind / "task.toml").write_text(BLIND)
    for name, colour in (("a", (220, 30, 30)), ("b", (30, 30, 220)), ("ref", (128, 128, 128))):
        Image.new("RGB", (300, 200), colour).save(blind / "sources" / f"{name}.png")
    rows = [{"id": f"x{n}", "a": {"file": "sources/a.png"}, "b": {"file": "sources/b.png"}, "ref": {"file": "sources/ref.png"},
             "truth": "SECRET-a-wins",
             "note": f"note {n}", "arm": "adapter"} for n in range(8)]
    (blind / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (blind / "labels.jsonl").write_text(json.dumps({"item": "x0", "labeller": "rater1", "revision": 1, "at": "t", "answers": {
        "flaws": {"ref": [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}]}}}) + "\n")
    result = look(blind, "--blind", "--sample", "8", "--seed", "3", "--to", tmp / "judge")
    assert result.returncode == 0, result.stderr
    index = (tmp / "judge" / "sheets.md").read_text()
    assert "SECRET" not in index and "Label by" not in index and "adapter" not in index  # no target, labels or other fields
    assert "note 3" in index and "**A**" not in index
    key = json.loads((tmp / "judge-key.json").read_text())  # beside the folder, never inside it
    assert not list((tmp / "judge").glob("*key*"))
    orders = {tuple(key["items"][row["id"]][letter] for letter in "AB") for row in rows}
    assert orders == {("a", "b"), ("b", "a")}  # both orders occur across eight items
    sheet = Image.open(tmp / "judge" / "sheet-01.png").convert("RGB")
    first = sorted(row["id"] for row in rows)[0]  # the sample keeps file order; x0 is in row 1
    centre_of_a = (8 + 522 // 2, 34 + 22 + 348 // 2)  # column A of row 1: margin, header with its text note, half a cell
    expected = (220, 30, 30) if key["items"][first]["A"] == "a" else (30, 30, 220)
    assert sheet.getpixel(centre_of_a) == expected  # the sheet draws the shuffled order the key records
    # x0's labelled box on ref: the 300 x 200 picture sits centred in its 522 x 348 cell (never enlarged).
    box_edge = (8 + 2 * 530 + (522 - 300) // 2 + 30 + 1, 34 + 22 + (348 - 200) // 2 + 50)
    assert sheet.getpixel(box_edge) == (128, 128, 128)  # labels, boxes included, are not drawn blind
    assert look(blind, "--sample", "8", "--seed", "3", "--to", tmp / "open").returncode == 0
    assert Image.open(tmp / "open" / "sheet-01.png").convert("RGB").getpixel(box_edge) != (128, 128, 128)  # but are drawn open


def test_blind_sheets_refuse_a_view_that_would_show_the_target(lab_with_set):
    folder, _, tmp = lab_with_set
    leaky = folder.parent / "leaky"
    (leaky / "sources").mkdir(parents=True)
    (leaky / "task.toml").write_text(BLIND + '\n[[pane]]\nfield = "changes"\nshow = "diff"\nagainst = "truth"\n')
    (leaky / "items.jsonl").write_text(json.dumps({"id": "x0", "a": "1", "b": "2", "ref": "3", "truth": "t", "note": "n",
                                                   "changes": "c"}) + "\n")
    result = look(leaky, "--blind", "--to", tmp / "leaky-look")
    assert result.returncode != 0 and "diff against the target" in result.stderr
    (leaky / "task.toml").write_text(BLIND)
    sorted_by_target = look(leaky, "--blind", "--lowest", "truth", "--to", tmp / "sorted-look")
    assert sorted_by_target.returncode != 0 and "sorting by the target" in sorted_by_target.stderr
