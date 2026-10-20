"""Sheets for agents: a gold set's items, with outputs joined, rendered as PNG images an
agent sees whole, and a sheets.md with everything that is better read as text.

Runs inside the ModelLab environment (Pillow, OpenCV): `modellab look` starts it.
"""

import datetime
import difflib
import json
import random
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import gold

MARGIN, HEADER, CAPTION, NOTE = 8, 34, 22, 22
MIN_CELL = 200  # below this, vision models read images unreliably
FRAMES = 4      # frames shown for a video
COLOURS = ("#d9480f", "#1971c2", "#2f9e44", "#9c36b5", "#c2255c", "#9a6700")  # a box's tag, as the page shows it


def main(request):
    folder, files = gold.open_set(request["set"])
    task = gold.load_task(files)
    items = gold.read_items(files / "items.jsonl")
    outputs = [Path(path) for path in request["outputs"]]
    items = gold.join(items, [gold.read_items(path) for path in outputs], [path.parent for path in outputs])
    if "ids" in request:  # a report draws only from the items it evaluated
        wanted = set(request["ids"])
        items = [item for item in items if item["id"] in wanted]
    labels = gold.latest(gold.read_labels(files / "labels.jsonl"))
    sort = request.get("lowest") or request.get("highest")
    if request.get("blind") and sort in {pane["field"] for pane in task["panes"] if pane["target"]}:
        raise SystemExit(f"--blind: sorting by the target {sort!r} would tell the judge about it")
    chosen, how = select(items, request)
    target = Path(request["to"])
    key = None
    if request.get("blind"):
        task, chosen, key = blinded(task, chosen, request["seed"])
        labels = {}  # people's answers are not for a judge's eyes
    visual = [pane for pane in task["panes"] if pane["show"] in gold.VISUAL]
    width, height = request["size"]
    target.mkdir(parents=True, exist_ok=False)
    placed = draw_sheets(folder, visual, chosen, width, height, target, task, labels) if visual else {}
    write_index(folder, request, task, chosen, how, labels, placed, target)
    result = {"to": str(target), "sheets": sorted({sheet for sheet, _ in placed.values()}), "items": len(chosen)}
    if key is not None:
        # Beside the folder, never inside it: a judge given the folder cannot read which pane is which.
        written = target.parent / f"{target.name}-key.json"
        written.write_text(json.dumps({"seed": request["seed"], "items": key}, indent=1))
        result["key"] = str(written)
    return result


def blinded(task, items, seed):
    """The view a person labelling blind gets: target panes withheld, the task's shuffled panes
    in a random order per item, named A, B, ... by place, and no other field of the item.
    (task, items, key) with key = {item id: {letter: field}}."""
    targets = {pane["field"] for pane in task["panes"] if pane["target"]}
    if leaks := [pane["field"] for pane in task["panes"] if pane["show"] == "diff" and pane.get("against") in targets]:
        raise SystemExit(f"--blind: panes {leaks} are a diff against the target, which would show it")
    shuffled = [field for field in task["queue"]["shuffle"] if field not in targets]
    letters = [chr(65 + place) for place in range(len(shuffled))]
    panes, slot = [], iter(letters)
    for pane in task["panes"]:
        if pane["target"]:
            continue
        if pane["field"] in shuffled:
            letter = next(slot)
            panes.append({**pane, "field": letter, "label": letter})
        else:
            panes.append(pane)
    key, views = {}, []
    for item in items:
        order = list(shuffled)
        random.Random(f"{seed}:{item['id']}").shuffle(order)
        key[item["id"]] = dict(zip(letters, order))
        view = {"id": item["id"]}
        for pane in panes:
            source = key[item["id"]].get(pane["field"], pane["field"])
            if source in item:
                view[pane["field"]] = item[source]
            if pane.get("against") in item:
                view[pane["against"]] = item[pane["against"]]
        views.append(view)
    return {**task, "panes": panes}, views, key


def select(items, request):
    if field := request.get("lowest") or request.get("highest"):
        scored = [item for item in items if isinstance(item.get(field), (int, float))]
        if not scored:
            raise SystemExit(f"No item has a number in {field!r}")
        scored.sort(key=lambda item: item[field], reverse=bool(request.get("highest")))
        order = "highest" if request.get("highest") else "lowest"
        return scored[:request["sample"]], f"the {request['sample']} items with the {order} {field}"
    seed = request["seed"]
    picked = sorted(random.Random(seed).sample(range(len(items)), min(request["sample"], len(items))))
    return [items[index] for index in picked], f"a random sample of {len(picked)} of {len(items)} items, seed {seed}"


def draw_sheets(folder, panes, items, width, height, target, task, labels):
    """Items as rows and panes as columns, as many rows per sheet as fit, with the boxes people
    drew on images. {item id: (sheet, row)}."""
    columns = len(panes)
    cell = (width - MARGIN * (columns + 1)) // columns
    if cell < MIN_CELL:
        raise SystemExit(f"{columns} image panes leave {cell} px per cell at width {width}; "
                         f"use a wider --size (cells need at least {MIN_CELL} px)")
    # Panes shown as text are named on every sheet, so a reader of the sheet alone knows they exist.
    text = [pane["label"] for pane in task["panes"] if pane["show"] not in gold.VISUAL]
    header = HEADER + (NOTE if text else 0)
    ratio = aspect(folder, panes, items)
    box = (cell, round(cell * ratio))
    fits = (height - header - MARGIN) / (box[1] + CAPTION + MARGIN)
    rows = int(fits)
    if rows < 1:
        raise SystemExit(f"a row needs {header + box[1] + CAPTION + 2 * MARGIN} px; use a taller --size")
    if fits - rows >= 0.5:  # more than half a row would stay empty: shrink the pictures to fill it
        tall = (height - header - MARGIN) // (rows + 1) - CAPTION - MARGIN
        if round(tall / ratio) >= MIN_CELL:
            rows, box = rows + 1, (round(tall / ratio), tall)
    font = ImageFont.load_default(size=16)
    placed = {}
    for start in range(0, len(items), rows):
        name = f"sheet-{start // rows + 1:02d}.png"
        sheet = Image.new("RGB", (width, height), "white")
        pen = ImageDraw.Draw(sheet)
        if text:
            pen.text((MARGIN, MARGIN), f"As text in sheets.md, not drawn here: {', '.join(text)}", fill="dimgrey", font=font)
        for column, pane in enumerate(panes):
            label = pane["label"] + ("  [target]" if pane["target"] else "") + ("  [hidden]" if pane["hidden"] else "")
            pen.text((MARGIN + column * (cell + MARGIN), header - HEADER + MARGIN), label, fill="black", font=font)
        for row, item in enumerate(items[start:start + rows]):
            top = header + row * (box[1] + CAPTION + MARGIN)
            for column, pane in enumerate(panes):
                left = MARGIN + column * (cell + MARGIN)
                picture = render(folder, pane, item.get(pane["field"]), box)
                if picture is None:
                    inset = left + (cell - box[0]) // 2
                    pen.rectangle((inset, top, inset + box[0] - 1, top + box[1] - 1), outline="grey")
                    pen.text((inset + 8, top + 8), f"no {pane['field']}", fill="grey", font=font)
                else:
                    origin = (left + (cell - picture.width) // 2, top + (box[1] - picture.height) // 2)
                    sheet.paste(picture, origin)
                    if pane["show"] == "image":
                        outline(pen, origin, picture.size, drawn(task, labels, item["id"], pane["field"]), font)
                pen.text((left, top + box[1] + 3), f"{item['id']} · {pane['label']}", fill="black", font=font)
            placed[item["id"]] = (name, row + 1)
        sheet.save(target / name, optimize=True)
    return placed


def drawn(task, labels, item_id, field):
    """(name, colour, box) for each box drawn on one image pane of one item, from each labeller's
    latest answer. A box is named and coloured as the page shows it: its number, its tag's label
    when there are several tags, and who drew it when several people did."""
    question = next((q for q in task["questions"] if q["kind"] == "region" and q["panes"] == [field]), None)
    if question is None:
        return []
    tags = question["tags"]
    drawers = [(who, record["answers"].get(question["id"], {}).get(field, []))
               for (item, who), record in sorted(labels.items()) if item == item_id and not record.get("skip")]
    drawers = [(who, boxes) for who, boxes in drawers if boxes]
    found = []
    for who, boxes in drawers:
        for number, box in enumerate(boxes, 1):
            place = next((index for index, tag in enumerate(tags) if tag["value"] == box.get("tag")), 0)
            label = tags[place]["label"] if len(tags) > 1 and "tag" in box else None
            parts = [who if len(drawers) > 1 else None, str(number), label]
            found.append((" · ".join(part for part in parts if part), COLOURS[place % len(COLOURS)], box))
    return found


def outline(pen, origin, size, boxes, font):
    """Boxes on a picture as pasted: an outline in the tag's colour, with its name."""
    for name, colour, box in boxes:
        x0, y0 = origin[0] + box["x"] * size[0], origin[1] + box["y"] * size[1]
        pen.rectangle((x0, y0, x0 + box["w"] * size[0], y0 + box["h"] * size[1]), outline=colour, width=3)
        left, top, right, bottom = pen.textbbox((x0 + 5, y0 + 4), name, font=font)
        pen.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=colour)
        pen.text((x0 + 5, y0 + 4), name, fill="white", font=font)


def aspect(folder, panes, items):
    """The median height-to-width ratio of the images shown, kept between 0.5 and 1.5."""
    ratios = []
    for item in items[:50]:
        for pane in panes:
            path = gold.media_path(folder, item.get(pane["field"]))
            if path is not None and path.is_file() and pane["show"] == "image":
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image)
                    ratios.append(image.height / image.width)
    ratios.sort()
    return min(1.5, max(0.5, ratios[len(ratios) // 2])) if ratios else 1.0


def render(folder, pane, value, box):
    path = gold.media_path(folder, value)
    if path is None or not path.is_file():
        return None
    if pane["show"] == "image":
        with Image.open(path) as image:
            picture = ImageOps.exif_transpose(image).convert("RGB")
        picture.thumbnail(box)
        return picture
    return frames(path, box)


def frames(path, box):
    """FRAMES frames spread over a video, two by two in the box."""
    video = cv2.VideoCapture(str(path))
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    pictures = []
    for index in range(FRAMES):
        video.set(cv2.CAP_PROP_POS_FRAMES, int(count * (index + 0.5) / FRAMES) if count > 0 else 0)
        ok, frame = video.read()
        if ok:
            pictures.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    video.release()
    if not pictures:
        return None
    grid = Image.new("RGB", box, "white")
    half = (box[0] // 2 - 2, box[1] // 2 - 2)
    for index, picture in enumerate(pictures):
        picture.thumbnail(half)
        grid.paste(picture, ((index % 2) * (box[0] // 2), (index // 2) * (box[1] // 2)))
    return grid


def write_index(folder, request, task, items, how, labels, placed, target):
    shown = {pane["field"] for pane in task["panes"]} | {pane["against"] for pane in task["panes"] if pane.get("against")}
    outputs = (f"{len(request['outputs'])} output files" if request.get("blind")  # their names may say which arm
               else ", ".join(f"`{p}`" for p in request["outputs"]))
    lines = [f"# {task['title']}", "",
             f"- Set: `{request['set']}`" + (f", outputs: {outputs}" if request["outputs"] else ""),
             f"- Items: {how}", f"- Sheets: {request['size'][0]} × {request['size'][1]} px, one item per row",
             f"- Made: {datetime.datetime.now().isoformat(timespec='seconds')}", ""]
    if request.get("blind"):
        lines[-1:-1] = ["- Blind: target panes withheld; shuffled panes named A, B, ... by place, in a new order "
                        "per item; no labels or other fields. The key is outside this folder."]
    for item in items:
        lines += [f"## {item['id']}", ""]
        if item["id"] in placed:
            lines += [f"Sheet `{placed[item['id']][0]}`, row {placed[item['id']][1]}.", ""]
        for pane in task["panes"]:
            if pane["show"] not in gold.VISUAL:
                lines += as_text(folder, pane, item)
        if other := {key: value for key, value in item.items() if key not in shown and key != "id"}:
            lines += ["| Field | Value |", "|---|---|"]
            lines += [f"| {key} | {cell(value)} |" for key, value in other.items()] + [""]
        for (item_id, labeller), record in sorted(labels.items()):
            if item_id == item["id"]:
                answer = "skipped" if record.get("skip") else json.dumps(record["answers"], ensure_ascii=False)
                lines += [f"- Label by {labeller} (revision {record['revision']}): {answer}"]
        lines += [""]
    (target / "sheets.md").write_text("\n".join(lines))


def as_text(folder, pane, item):
    value = item.get(pane["field"])
    head = f"**{pane['label']}**" + (" (target)" if pane["target"] else "")
    if value is None:
        return [f"{head}: missing", ""]
    show = pane["show"]
    if show == "audio":
        return [f"{head}: audio `{gold.media_path(folder, value)}`", ""]
    if show == "fields" and isinstance(value, dict):
        return [head, "", "| Field | Value |", "|---|---|"] + [f"| {k} | {cell(v)} |" for k, v in value.items()] + [""]
    if show == "chat" and isinstance(value, list):
        return [head, ""] + [f"- **{turn.get('role', '?')}**: {turn.get('content', '')}" for turn in value] + [""]
    if show == "diff":
        diff = difflib.unified_diff(str(item.get(pane["against"], "")).splitlines(), str(value).splitlines(),
                                    pane["against"], pane["field"], lineterm="")
        return [head, "", "```diff", *diff, "```", ""]
    text = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)
    return [head, "", text if show == "markdown" else "> " + text.replace("\n", "\n> "), ""]


def cell(value):
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    try:
        print(json.dumps(main(json.loads(sys.argv[1]))))
    except gold.Invalid as error:
        raise SystemExit(f"modellab look: {error}")
