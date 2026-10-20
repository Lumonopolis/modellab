"""Training recipes. The tests with MLX-VLM need an installed environment and a Qwen-family
vision model folder (ChatML template), such as qwen3-vl-8b-4bit:

    MODELLAB_TEST_MLX_ENV=<root>/envs/mlx MODELLAB_TEST_VLM=<model folder> python3 -m pytest tests/test_recipe.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

from conftest import HOME

MLX_ENV = os.environ.get("MODELLAB_TEST_MLX_ENV")
VLM = os.environ.get("MODELLAB_TEST_VLM")
needs_vlm = pytest.mark.skipif(not (MLX_ENV and VLM), reason="set MODELLAB_TEST_MLX_ENV and MODELLAB_TEST_VLM")

# Synthetic examples in the shape of a photo-editing project's examples: two pictures, a JSON reply.
SCRIPT = '''
import os, sys, types
from pathlib import Path
import numpy as np
from PIL import Image
from mlx_vlm import load, stream_generate
from mlx_vlm.generate.dispatch import _prepare_generation_inputs
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer.datasets import VisionDataset
from modellab import recipe

model_path, mode = sys.argv[1], sys.argv[2]
for name, (w, h) in (("a.png", (96, 64)), ("b.png", (64, 64))):
    Image.fromarray(np.linspace(0, 255, w * h * 3).reshape(h, w, 3).astype("uint8")).save(name)
training = mode in ("fit", "train", "measure", "failing", "straight", "interrupted", "torn", "other", "small",
                    "undeclared", "nolook", "twice", "hidden", "noallowed")
count = {"fit": 1, "straight": 3, "interrupted": 3, "torn": 3, "other": 3}.get(mode, 4) if training else 1
model, processor = load(model_path, lazy=not training)
config = model.config.__dict__
reply = '{"crop":{"box":[24,89,997,956],"angle":0.0},"treatment":"colour"}'
items = [{"id": f"ex{n}", "images": ["a.png", "b.png"], "messages": [
    {"role": "user", "content": f"Image 1 is photograph {n} and image 2 its day. Reply with the edit as JSON."},
    {"role": "assistant", "content": reply}]} for n in range(count)]
if mode == "image":
    items = [{"id": "ex0", "image": "a.png", "messages": items[0]["messages"]}]
dataset = VisionDataset(items, config, processor, train_on_completions=True)
pictures = 1 if mode == "image" else 2
base = lambda item: apply_chat_template(processor, config, [item["messages"][0]], num_images=pictures)
head = reply[:reply.index(',"treatment')]
prompts = {"good": lambda item: [base(item) + '{"', base(item) + head + ',"'],
           "split": lambda item: [base(item) + "{", base(item) + head + ","],
           "short": lambda item: [base(item).rsplit("<|im_start|>", 1)[0]],
           "whole": lambda item: [apply_chat_template(processor, config, item["messages"], add_generation_prompt=False,
                                                      num_images=pictures)],
           "long": lambda item: [base(item) + '{"'],
           "image": lambda item: [base(item) + '{"'],
           }.get(mode, lambda item: [base(item) + '{"'])
if mode == "fit":
    # The check's stand-in makes as many prompt tokens as real generation does.
    made = _prepare_generation_inputs(types.SimpleNamespace(config=model.config), processor, base(items[0]) + '{"',
                                      ["a.png", "b.png"], None, None, {})[0]
    generated = list(stream_generate(model, processor, base(items[0]) + '{"', image=["a.png", "b.png"], max_tokens=1))
    assert np.array(made).size == generated[-1].prompt_tokens, (np.array(made).size, generated[-1].prompt_tokens)
    recipe.train(model, processor, dataset, prompts, "adapter", fit=True)
elif mode == "train":  # 4 examples, 2 epochs, accumulate 3: 9 steps; checkpoints at 6 (the default) and 9
    Path("adapter").mkdir()
    Path("adapter/unrelated.safetensors").write_text("not the recipe's")
    recipe.train(model, processor, dataset, prompts, "adapter", epochs=2, accumulate=3, full=True, look=None)
elif mode == "measure":  # NaN never counts, and a tie keeps the earlier checkpoint
    values = iter([float("nan"), 0.4, 0.4, 0.2])
    recipe.train(model, processor, dataset, prompts, "adapter", save_every=1, measure=lambda model: next(values),
                 full=True, look=None)
elif mode == "failing":  # the measure fails after the first checkpoint, then works when run again
    first = not Path("adapter/recipe-state.json").exists()
    recipe.train(model, processor, dataset, prompts, "adapter", save_every=2,
                 measure=lambda model: 1 / 0 if first else 0.5, full=True, look=None)
elif mode in ("straight", "interrupted", "torn", "other"):
    # 3 examples, 2 epochs, accumulate 2, dropout: 6 steps, checkpoints at 4 and 6 (a short last segment).
    folder = {"straight": "a", "other": "a", "interrupted": "b", "torn": "c"}[mode]
    if mode in ("interrupted", "torn") and not Path(folder, "started").exists():
        Path(folder).mkdir()
        Path(folder, "started").write_text("")
        write = recipe._write_state
        if mode == "interrupted":  # stopped right after the first checkpoint is complete
            recipe._write_state = lambda out, state: (write(out, state), os._exit(3))
        else:  # stopped inside the first checkpoint, before the state names it
            recipe._write_state = lambda out, state: os._exit(3)
    recipe.train(model, processor, dataset, prompts, folder, epochs=2, accumulate=2, save_every=4, dropout=0.1,
                 learning_rate=2e-4 if mode == "other" else 1e-4, full=True, look=None)
elif mode in ("small", "undeclared", "nolook", "twice", "hidden", "noallowed"):
    calibration = {"id": "cal0", "images": ["b.png", "a.png"], "messages": [
        {"role": "user", "content": "Image 1 is a calibration photograph and image 2 its day. Reply with the edit as JSON."},
        {"role": "assistant", "content": '{"crop":{"box":[0,0,1000,1000],"angle":0.0},"treatment":"black and white"}'}]}
    declared = ["ex0", "ex1"] if mode == "undeclared" else ["ex0", "ex1", "cal0"]
    chosen = [items[0], items[0]] if mode == "twice" else [items[0], items[1], calibration]
    if mode == "hidden":  # ex1 may not be shown, and its picture has moved
        chosen = [items[0], {**items[1], "images": ["moved.png"]}]
    # parse gives every reply (and target) a dict of two parts, so sheets.md shows them part by part.
    look = recipe.Look(items=chosen, declared=declared, allowed=[] if mode == "noallowed" else ["ex0", "cal0"],
                       max_tokens=2, compare=8,
                       parse=lambda text: {"start": text[:4], "length": len(text)},
                       picture=lambda item, reply, value: "a.png" if item["id"] == "cal0" else None)
    if mode == "nolook":
        recipe.train(model, processor, dataset, prompts, "adapter", small=2)
    recipe.train(model, processor, dataset, prompts, "adapter", small=2, look=look)
elif mode == "compare":
    import mlx.core as mx
    a, b = mx.load("a/adapters.safetensors"), mx.load(f"{sys.argv[3]}/adapters.safetensors")
    print("largest difference", max(mx.abs(a[k] - b[k]).max().item() for k in a), "of", len(a), "weights")
else:
    recipe.check_prompts(model, processor, dataset, prompts, max_seq_length=50 if mode == "long" else None)
'''


@pytest.fixture
def vlm_job(lab, tmp_path):
    (lab.root / "envs").mkdir(parents=True)
    (lab.root / "envs" / "mlx").symlink_to(Path(MLX_ENV))
    work = tmp_path / "work"
    work.mkdir()
    (work / "train.py").write_text(SCRIPT)

    def run(mode, *more, budget="4G"):
        path = lab.run("python", "train.py", VLM, mode, *more, budget=budget, env="mlx", extra=("--cwd", str(work)))
        _, status = lab.wait(path, timeout=600)
        return path, status, (path / "log").read_text()
    return run


def test_differences_find_the_first_token_that_differs():
    from modellab import recipe
    found = recipe._differences({"a": [1, 2, 3], "b": [1, 2], "c": [5], "d": [1]}, {"a": [1, 9, 3], "b": [1, 2, 7],
                                                                                    "c": [5], "e": [2]})
    assert found == {"a": 1, "b": 2, "c": None}


def test_jobs_can_import_modellab(lab, tmp_path):
    path = lab.run(sys.executable, "-c", "import modellab, pathlib; print(pathlib.Path(modellab.__file__).parent.parent)",
                   extra=("--cwd", str(tmp_path)))
    state, status = lab.wait(path)
    assert (state, status["exit_code"]) == ("exited", 0), (path / "log").read_text()
    assert (path / "log").read_text().split()[-1] == str(HOME)


@needs_vlm
def test_prompts_made_like_training_pass(vlm_job):
    path, status, log = vlm_job("good")
    assert status["exit_code"] == 0, log
    assert "Prompt check passed: 2 prompts for 1 examples." in log
    summary = json.loads((path / "out" / "prompt-check.json").read_text())
    assert (summary["examples"], summary["prompts"], summary["mismatches"]) == (1, 2, 0)


@needs_vlm
def test_a_prompt_that_splits_a_token_training_kept_whole_stops_the_job(vlm_job):
    # 2026-10-02: a project's sampler ended with '{' and joined with ','; training had '{"' and '},"'.
    path, status, log = vlm_job("split")
    assert status["exit_code"] == 1, log
    assert "Prompt check failed: 2 of 2 prompts" in log
    assert "ex0, prompt 1 of 2: differs at token" in log and "training has '{\"', the prompt has '{'" in log
    assert "ex0, prompt 2 of 2: differs at token" in log and "training has '},\"', the prompt has '},'" in log
    assert json.loads((path / "out" / "prompt-check.json").read_text())["mismatches"] == 2


@needs_vlm
def test_a_prompt_that_stops_short_of_the_reply_stops_the_job(vlm_job):
    _, status, log = vlm_job("short")
    assert status["exit_code"] == 1, log
    assert "before the first trained token" in log and "training never taught it" in log


@needs_vlm
def test_an_example_longer_than_max_seq_length_stops_the_job(vlm_job):
    _, status, log = vlm_job("long")
    assert status["exit_code"] == 1, log
    assert "ex0: " in log and "more than max_seq_length 50" in log


@needs_vlm
def test_a_prompt_that_is_the_whole_example_stops_the_job(vlm_job):
    _, status, log = vlm_job("whole")
    assert status["exit_code"] == 1, log
    assert "nothing is left for the model to write" in log


@needs_vlm
def test_an_example_with_one_image_under_image_is_checked_with_it(vlm_job):
    _, status, log = vlm_job("image")
    assert status["exit_code"] == 0, log


@needs_vlm
def test_the_fit_step_checks_prompts_then_trains_one_step_and_keeps_no_adapter(vlm_job):
    path, status, log = vlm_job("fit", budget="16G")
    assert status["exit_code"] == 0, log
    assert "Prompt check passed" in log
    fit = json.loads((path / "out" / "fit.json").read_text())
    assert fit["fit_example"] == "ex0" and fit["steps"] == 1 and fit["peak_memory_gb"] > 0 and fit["tokens"] > 0
    assert not (Path(json.loads((path / "run.json").read_text())["cwd"]) / "adapter").exists()


@needs_vlm
def test_training_keeps_the_latest_checkpoint_and_leaves_other_files_alone(vlm_job):
    path, status, log = vlm_job("train", budget="16G")
    assert status["exit_code"] == 0, log
    result = json.loads((path / "out" / "train.json").read_text())
    assert (result["steps"], result["latest"], result["best"]) == (9, "step-0000009", None)
    adapter = Path(json.loads((path / "run.json").read_text())["cwd"]) / "adapter"
    assert [f.name for f in (adapter / "checkpoints").iterdir()] == ["step-0000009"]
    assert {f.name for f in (adapter / "checkpoints" / "step-0000009").iterdir()} == {
        "adapters.safetensors", "adapter_config.json", "optimizer.safetensors"}
    assert (adapter / "adapters.safetensors").read_bytes() == (
        adapter / "checkpoints" / "step-0000009" / "adapters.safetensors").read_bytes()
    assert (adapter / "unrelated.safetensors").exists()


@needs_vlm
def test_training_keeps_the_best_checkpoint_by_the_measure(vlm_job):
    path, status, log = vlm_job("measure", budget="16G")
    assert status["exit_code"] == 0, log
    result = json.loads((path / "out" / "train.json").read_text())
    assert result["measures"] == [[1, None], [2, 0.4], [3, 0.4], [4, 0.2]]
    assert result["best"] == {"checkpoint": "step-0000002", "measure": 0.4} and result["latest"] == "step-0000004"
    adapter = Path(json.loads((path / "run.json").read_text())["cwd"]) / "adapter"
    assert sorted(f.name for f in (adapter / "checkpoints").iterdir()) == ["step-0000002", "step-0000004"]


@needs_vlm
def test_a_failing_measure_loses_no_training(vlm_job):
    path, status, log = vlm_job("failing", budget="16G")
    assert status["exit_code"] == 1 and "ZeroDivisionError" in log, log
    cwd = Path(json.loads((path / "run.json").read_text())["cwd"])
    assert json.loads((cwd / "adapter" / "recipe-state.json").read_text())["latest"] == "step-0000002"
    path, status, log = vlm_job("failing", budget="16G")
    assert status["exit_code"] == 0 and "Resuming from step-0000002 (step 2 of 4)." in log, log
    result = json.loads((path / "out" / "train.json").read_text())  # step 2 is measured on resuming
    assert result["measures"] == [[2, 0.5], [4, 0.5]]
    assert result["best"] == {"checkpoint": "step-0000002", "measure": 0.5}


@needs_vlm
def test_a_stopped_run_resumes_exactly(vlm_job):
    # With dropout, accumulation across the stop and a short last segment.
    _, status, log = vlm_job("straight", budget="16G")
    assert status["exit_code"] == 0, log
    path, status, log = vlm_job("interrupted", budget="16G")
    assert status["exit_code"] == 3, log
    cwd = Path(json.loads((path / "run.json").read_text())["cwd"])
    assert json.loads((cwd / "b" / "recipe-state.json").read_text())["latest"] == "step-0000004"
    assert not (cwd / "b" / "adapters.safetensors").exists()  # stopped before the copy; resuming does it
    _, status, log = vlm_job("interrupted", budget="16G")
    assert status["exit_code"] == 0 and "Resuming from step-0000004 (step 4 of 6)." in log, log
    _, status, log = vlm_job("compare", "b")
    assert status["exit_code"] == 0 and "largest difference 0.0 " in log, log
    _, status, log = vlm_job("interrupted", budget="16G")  # a finished run, run again, does nothing
    assert status["exit_code"] == 0 and "Resuming from step-0000006 (step 6 of 6)." in log, log
    _, status, log = vlm_job("other", budget="16G")  # a different learning rate into the same folder
    assert status["exit_code"] == 1 and "holds a run with other learning_rate (recipe-state.json)" in log, log


@needs_vlm
def test_a_run_stopped_inside_its_first_checkpoint_starts_again_cleanly(vlm_job):
    _, status, log = vlm_job("straight", budget="16G")
    assert status["exit_code"] == 0, log
    path, status, log = vlm_job("torn", budget="16G")
    assert status["exit_code"] == 3, log
    cwd = Path(json.loads((path / "run.json").read_text())["cwd"])
    assert not (cwd / "c" / "recipe-state.json").exists()
    _, status, log = vlm_job("torn", budget="16G")
    assert status["exit_code"] == 0 and "Resuming" not in log, log
    assert [f.name for f in (cwd / "c" / "checkpoints").iterdir()] == ["step-0000006"]
    _, status, log = vlm_job("compare", "c")
    assert status["exit_code"] == 0 and "largest difference 0.0 " in log, log


@needs_vlm
def test_a_small_run_ends_with_a_look_at_declared_items(vlm_job):
    path, status, log = vlm_job("small", budget="16G")
    assert status["exit_code"] == 0, log
    assert "A small run: 2 examples, one pass" in log
    assert "Replies: 3 of 3 parse; 3 stopped at max_tokens (2)" in log
    assert "Adapter against the untrained model:" in log and "of 2 replies differ" in log  # ex0 and cal0
    result = json.loads((path / "out" / "train.json").read_text())
    assert (result["full"], result["steps"], result["look"]["not_shown"]) == (False, 2, 1)
    assert result["look"]["sheets"] == {"items": 2, "files": ["sheet-01.png"]}
    folder = Path(json.loads((path / "run.json").read_text())["cwd"]) / "adapter" / "small"
    text = (folder / "look" / "sheets.md").read_text()
    assert "## ex0" in text and "## cal0" in text and "| Part | Target | Reply |" in text and "| length **≠** |" in text
    # ex1 is declared but not allowed: it counts in the health line and appears nowhere by id or reply.
    for written in (text, (folder / "look" / "look.json").read_text(), (folder / "look" / "replies.jsonl").read_text(),
                    (folder / "look-base.json").read_text(), (path / "out" / "train.json").read_text()):
        assert "ex1" not in written
    assert not [line for line in log.splitlines() if "ex1" in line]
    assert len((folder / "look" / "replies.jsonl").read_text().splitlines()) == 2
    _, status, log = vlm_job("small", budget="16G")  # run again: it looks again, against the kept replies
    assert status["exit_code"] == 0 and "Resuming from step-0000002 (step 2 of 2)." in log, log
    assert "of 2 replies differ" in log


@needs_vlm
def test_an_undeclared_look_item_is_refused_before_training(vlm_job):
    path, status, log = vlm_job("undeclared", budget="16G")
    assert status["exit_code"] == 1 and "1 are not in the declared list (such as cal0)" in log, log
    assert not (Path(json.loads((path / "run.json").read_text())["cwd"]) / "adapter").exists()
    _, status, log = vlm_job("twice", budget="16G")
    assert status["exit_code"] == 1 and "1 ids appear more than once (such as ex0)" in log, log
    _, status, log = vlm_job("hidden", budget="16G")  # a refusal never names an item that may not be shown
    assert status["exit_code"] == 1 and "1 look items have a picture that is not there (1 not allowed to be named)" in log
    assert "ex1" not in log and "moved.png" not in log, log


@needs_vlm
def test_a_look_with_nothing_allowed_says_so_rather_than_alarming(vlm_job):
    _, status, log = vlm_job("noallowed", budget="16G")
    assert status["exit_code"] == 0 and "Adapter against the untrained model: no allowed items to compare" in log, log


@needs_vlm
def test_training_needs_a_look_or_an_explicit_none(vlm_job):
    _, status, log = vlm_job("nolook", budget="16G")
    assert status["exit_code"] == 1 and "train needs look=recipe.Look" in log, log
