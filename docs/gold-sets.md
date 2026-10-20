# Gold sets, sheets and labelling

Status, 2026-10-01: built (all five build steps below). Build step 1 of `docs/on-the-data.md`.

## Adopt or build (checked 2026-09-30)

- Gold-set format: keep JSONL, media by content hash and a manifest whose hash
  is the set id. Items map one to one onto Inspect AI's `Sample` (id, input,
  target, metadata) for evaluation. A Croissant export can come later, for
  sharing. Hugging Face `datasets` fingerprints and DVC hashes are not stable
  content ids.
- Labelling: build, on Python's own web server. Label Studio (usage data on, not mobile friendly), Argilla
  (no release since 2025-03), Potato (GPL; four advisories in 2026-08 and 09,
  one a critical remote code execution), FiftyOne (spatial labels only;
  MongoDB; usage data on), CVAT (spatial only; about 20 services) and Prodigy
  (paid per seat, so every fork must buy it) do not fit.
- Sheets: build a thin script on Pillow (layout, ids) and OpenCV (video
  frames). Both are in the environment; ImageMagick is not, and Homebrew's
  ffmpeg cannot draw text.

## A gold set is a folder

| File | Holds |
|---|---|
| `task.toml` | What an item shows and what is asked (below) |
| `items.jsonl` | One item per line: an `id` and named fields. A media field is `{"media": "<sha256>.<ext>"}` |
| `media/` | Images and videos, named by content hash |
| `labels.jsonl` | Answers, appended: item, labeller, revision, time, answers. Nothing is overwritten |
| `versions/<id>/` | Written by `modellab gold freeze`: copies of the files above and a `manifest.json` of their hashes and the media. Its hash is the id |

While a set is being built, a media field can name a file by path,
`{"file": "sources/a.jpg"}`, relative to the set. Freezing clones each such
file into `media/` (an APFS clone takes no space and does not change when the
original does), writes the content-hash name into `items.jsonl`, and keeps a
version. A version never changes and is read as `<set>@<id>`; new items or
labels make a new version with a new id. Targets are fields of an item (existing human work, such as a person's
own edit) or labels collected with `modellab label`. A project keeps its gold
sets in its own folders; ModelLab provides the format and the tools.

## task.toml

One file configures a task in three parts: what is shown (panes), what is
asked (questions) and how the work runs (queue). The page and `modellab look`
both read it, so a person and an agent see the same view.

```toml
title = "Edit feedback"
instructions = "guide.md"            # shown before the first item and on ?

[layout]
arrange = "row"                      # row, column or grid
compare = ["model_edit", "user_edit"]  # these panes zoom and pan together; F flicks between them

[[pane]]
field = "default_render"
show = "image"                       # image, video, audio, text, markdown, chat, fields or diff
label = "Default"
hidden = true                        # shown while its key is held
key = "o"

[[pane]]
field = "model_edit"
show = "image"
label = "Model"

[[pane]]
field = "user_edit"
show = "image"
label = "Target"
target = true                        # the goal's target

[[question]]
id = "verdict"
kind = "choice"                      # choice, multi, scale, number, text, rank, span or region
options = ["keep", "close", "rework", "not mine"]
pane = "model_edit"                  # asked about one pane; omit to ask about the whole item
help = "Keep: you would publish it as it is."

[[question]]
id = "issues"
kind = "multi"
options = ["crop", "tone", "colour", "black and white"]
pane = "model_edit"
unless = {verdict = ["keep"]}        # asked only then; `when` asks only when it holds
required = false

[[question]]
id = "quality"
kind = "scale"
range = [1, 5]
labels = { 1 = "unusable", 5 = "excellent" }

[queue]
order = "random"                     # file, random, or field:<name> (sorted by that field)
blind = false                        # true hides target panes while labelling (below)
shuffle = []                         # panes shown in a random order as A, B, ..., e.g. ["edit_a", "edit_b"]
labellers = 2                        # each item answered by two people, to measure agreement
auto_save = {verdict = ["keep"], quality = [5]}  # save and move on without Enter
```

- Pane displays: `image` and `video` (images of `compare` panes open side by
  side and zoom and scroll together; their videos play, pause and seek
  together; F flicks between the first two images), `audio`, `text`,
  `markdown`, `chat` (a conversation, turn by turn), `fields` (a table of
  values, such as metrics) and `diff` (two text panes, differences marked).
- Question kinds: `choice` (one option; yes/no is a choice), `multi` (several
  options), `scale` (any range, with labels), `number` (with a range and a
  unit), `text` (a note), `rank` (order several panes; a pairwise choice is a
  rank of two), `span` (highlight parts of a text and tag them) and `region`
  (mark boxes on an image or a video frame and tag them).
- `span` and `region` ask about one pane (`text` or `markdown` for a span;
  `image` or `video` for a region), and a pane takes one of them: `tags` mark
  different things. `tags` are like options (a name, or `{value, label,
  key}`); without tags, marks have no tag. The answer is the list of marks,
  so no marks is an answer; `required = true` asks for at least one.
  - A span is `{"start", "end", "text", "tag"}`: offsets in characters (code
    points, as Python counts them) of the field's text, markdown source
    included. `text` repeats the text marked, and a save is refused when it
    does not match the item. Spans are in order and do not overlap.
  - A region is `{"x", "y", "w", "h", "tag"}` in fractions of the picture as
    shown (after its EXIF rotation), and on a video also `"time"`, in
    seconds, of the frame it was drawn on.
- Every question can take `help`, `required`, `when`, `unless`, a `default`
  and `details` (shown under D; the default for `text`). An option or a
  hidden pane can take its own `key`.
- A condition (`when`, `unless`, `auto_save`) is a table of question = list of
  answers; every question in it must have one of its answers. `when` and
  `unless` may name only questions above their own: answers are checked in
  order, the same way on the page and on the server, and an answer to a
  question that is not asked is refused.
- A question kind can come from the project: `[plugins] kinds = {mood = {page
  = "mood.js", check = "mood.py"}}`, where the script registers
  `ModelLab.kinds.mood` on the page and `check(question, value)` returns a
  problem or None. A pane display is a script only: `displays = {name =
  "file.js"}`. Neither needs a change to ModelLab. The page's content policy
  applies to plugins: no `eval`, no inline styles, and nothing loaded from
  elsewhere or from `data:` and `blob:` addresses.
- Shuffled panes are shown in a random order per item, recorded with the
  answer, and labelled by position, A, B, ..., without a target mark. They
  must show the same way, none may be hidden, and a question about panes must
  cover all of them or none, so nothing on the page tells them apart.
- Blind mode never sends a target pane to the page, and refuses a task where
  something else would show it: a diff against a target, `order` by a target
  field, or a question about a target pane.
- The page stays one screen: panes, questions, keys. Configuration adds to
  the file, not to the screen.

## `modellab look`

```sh
modellab look <gold set> [--outputs run/out/outputs.jsonl] [--lowest FIELD | --highest FIELD]
              [--sample 24] [--seed 1] [--size 1600x1600] [--to folder] [--blind]
```

- Joins outputs to items by `id`. Each item is one row; the panes follow
  `task.toml`, with the target marked, and each cell carries the item id.
  Hidden panes are shown too. Text, markdown, chat, fields and diffs go into
  `sheets.md` as text; audio is listed by file. Boxes people drew on images
  are drawn on the sheet, named as the page names them; every answer,
  spans and video boxes included, is in `sheets.md`. Each sheet names, at
  the top, the panes that are only in `sheets.md`.
- Rows fill the sheet: when more than half a row would stay empty, the
  pictures shrink to fit one more row (never below 200 px wide).
- `--blind` gives an agent judge the view a person labelling blind gets:
  target panes withheld, the task's `shuffle` panes in a new order per item
  and named A, B, ... by place, and no labels, boxes or other fields. The key
  (which letter is which pane, per item) goes beside the folder as
  `<folder>-key.json`, never inside it. A diff against a target, or sorting
  by a target field, is refused.
- Writes PNG sheets of 1600 × 1600 by default (seen unscaled by current
  Claude and GPT vision models; `--size` changes it), cells at least 200 px,
  and a `sheets.md` that lists each item's id, fields and labels. Text fields
  go in `sheets.md` as text, not as pictures of text. A video shows four
  frames.
- The default is a random sample with the seed recorded; `--lowest` and
  `--highest` take the items with the lowest or highest number in a field,
  such as a score or a distance. Other fields of an item (output scores, for
  example) go in `sheets.md` as a table, with every label given.

## `modellab label`

```sh
modellab label <gold set> [--labeller name] [--port N] [--no-open]
```

- One page on `127.0.0.1` only, served by Python's own web server. It answers
  only requests addressed to it by that name (no DNS rebinding), accepts
  answers only with a header that pages from other sites cannot send, and
  serves only the page, plugin scripts the task names, and files items name
  (with byte ranges, so video can seek). Use from a phone on another device
  comes later, as an explicit option with a token.
- The panes as `task.toml` arranges them, large. An image opens larger on
  click; the `compare` panes open side by side, zoom together and scroll
  together; F flicks the first compare pane to the second and back. A hidden
  pane shows while its key is held, or stays shown after a tap.
- Keys, in this order: keys the task gives; a scale's own digits (5 means 5);
  a single choice's first letter when no other option on screen shares it;
  then free digits. Enter saves and moves on, Backspace goes back, D opens
  the questions under `details`, S skips, ? shows the instructions and the
  keys. With marks, Backspace removes the chosen mark (or a box waiting for
  its tag) and Esc lets it go; with a region video, Space plays and pauses,
  and `,` and `.` step a frame (1/30 s). Those three keys are then the
  page's own; a task cannot give them to an option. Keys show on the buttons (and in `aria-keyshortcuts`); a key typed
  into a text field stays text. `auto_save` saves without Enter when its
  condition holds for every pane, and not while D is open. Keys ignore case.
- Every save is checked against the task and appended as a new revision; a
  stale page cannot overwrite a newer answer, and an answer that fails to
  save stays on the page with the reason and a retry.
- Marks. A span: select text with the mouse, and it is marked; where there
  are several tags, it waits for a tag's key or button. On a phone, select
  text, then tap its tag (or "Mark the selection"). A region: drag on the
  picture (a click still opens it large); where there are several tags, the
  box waits for one. A region video has its own controls, so a box is drawn
  on a paused frame, and shows only on that frame. A mark or box clicked (or
  text selected inside a mark) is chosen until something else is answered:
  a tag's key retags it, Backspace removes it. A new span that overlaps
  others replaces them. Save waits while a box waits for its tag. Each mark is listed under its pane, with ×
  to remove it and, for a video box, its time.
- Checked in a browser on desktop and phone, in light and dark mode
  (2026-10-01).

## Build

1. Declare Pillow and OpenCV in the environment (they used to arrive only
   through MLX-VLM). The page needs no packages.
2. The folder format, `modellab gold check` and `modellab gold freeze`.
3. `modellab look`.
4. `modellab label` with all pane displays and the question kinds `choice`,
   `multi`, `scale`, `number`, `text` and `rank`.
5. `span` and `region`, which need the most page work.

Proof by use: a photo-editing project's archive as a gold set (default render
next to the photographer's own edit), and a sheet of a model's edits beside
those edits.
