"""Reports: numbers that come with their items, and a judge no better than guessing marked so."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import CLI
from modellab import report

EVAL_ENV = os.environ.get("MODELLAB_TEST_EVAL_ENV")
MLX_ENV = os.environ.get("MODELLAB_TEST_MLX_ENV")


def rows(pairs, scorer="parsed"):
    """(target, answer) pairs as an Inspect log's rows, scored right when the answer is the target."""
    return [{"id": f"i{n}", "epoch": 1, "targets": [target], "target": target, "reply": f"I think {answer}.",
             "scores": {scorer: {"value": float(answer == target), "values": None, "raw": "C" if answer == target else "I",
                                 "answer": answer}}, "error": None} for n, (target, answer) in enumerate(pairs)]


def figures(log, scorer="parsed", seed=1):
    items, failed = report.per_item(log, scorer)
    return report.numbers(items, seed), failed


def test_a_constant_answer_is_no_better_than_trivial():
    result, _ = figures(rows([("yes", "no")] * 20 + [("no", "no")] * 19 + [("no", "yes")]))
    assert result["n"] == 40 and result["result"] == pytest.approx(0.475)
    assert result["baselines"] == {"no": 0.5, "yes": 0.5}
    assert result["better_than_trivial"] is False
    assert result["targets"] == {"no": 20, "yes": 20} and result["answers"] == {"no": 39, "yes": 1}
    low, high = result["interval"]
    assert low < 0.475 < high


def test_a_clearly_better_answerer_is_shown_better_with_its_interval():
    log = rows([("yes", "yes")] * 18 + [("yes", "no")] * 2 + [("no", "no")] * 19 + [("no", "yes")])
    result, _ = figures(log)
    assert result["result"] == pytest.approx(37 / 40) and result["better_than_trivial"] is True
    assert result["difference"][0] > 0  # the paired interval of (result - best trivial) is above zero
    assert figures(log)[0] == result  # repeatable by its seed


def test_imbalanced_targets_raise_the_trivial_bar():
    result, _ = figures(rows([("colour", "colour")] * 36 + [("bw", "colour")] * 4))
    assert result["result"] == pytest.approx(0.9) and result["baselines"]["colour"] == pytest.approx(0.9)
    assert result["better_than_trivial"] is False


def test_spellings_of_one_target_are_one_answer():
    log = rows([("Yes.", "yes")] * 5 + [("YES", "yes")] * 5 + [(" yes!", "yes")] * 5 + [("no", "no")] * 5)
    for row in log:
        row["scores"]["parsed"]["value"] = 1.0  # a scorer that ignores case and end punctuation, as targets differ only so
    result, _ = figures(log)
    assert result["baselines"] == {"no": 0.25, "yes": 0.75} and result["targets"] == {"yes": 15, "no": 5}


def test_no_baseline_when_the_scorer_is_not_exact_match():
    several = rows([("yes", "yes")] * 10 + [("no", "no")] * 2)
    for row in several[:5]:
        row["targets"] = ["yes", "y"]  # several targets: "always yes" would score what the scorer says, not 1/3
    result, _ = figures(several)
    assert result["baselines"] is None and "several targets" in result["no_baseline"]
    loose = rows([("yes", "yes, it is")] * 6 + [("no", "no")] * 6)  # right without the answer being the target
    for row in loose[:6]:
        row["scores"]["parsed"]["value"] = 1.0
    result, _ = figures(loose)
    assert result["baselines"] is None and "exactly when it equals the target" in result["no_baseline"]
    assert result["better_than_trivial"] is None


def test_tied_best_answers_must_each_be_beaten():
    result, _ = figures(rows([("a", "a")] * 6 + [("b", "a")] * 6))
    assert result["baselines"] == {"a": 0.5, "b": 0.5} and result["better_than_trivial"] is False


def test_failed_items_are_listed_not_fatal_and_epochs_keep_right_or_wrong():
    log = rows([("a", "a"), ("a", "b"), ("b", "b")])
    log[1]["id"], log[1]["epoch"] = "i0", 2  # a second epoch of i0, wrong this time
    log[2]["error"], log[2]["scores"] = "model timed out", {}
    items, failed = report.per_item(log, "parsed")
    [item] = items
    assert item["score"] == 0.5 and item["epochs"] == 2 and failed == ["i2"]
    result = report.numbers(items, seed=1)
    assert result["baselines"] == {"a": 1.0}  # still right or wrong per epoch, so a baseline applies


def test_graded_and_several_valued_scores():
    graded = [{"id": f"g{n}", "epoch": 1, "targets": [""], "target": "", "reply": "", "error": None,
               "scores": {"distance": {"value": value, "values": None, "raw": value, "answer": None}}}
              for n, value in enumerate([0.2, 0.4, 0.9])]
    result, _ = figures(graded, "distance")
    assert result["result"] == pytest.approx(0.5) and result["baselines"] is None and "graded" in result["no_baseline"]
    two = [{**row, "scores": {"two": {"value": None, "values": {"a": 1.0, "b": 0.0}, "raw": "{}", "answer": None}}} for row in graded]
    with pytest.raises(SystemExit, match="--scorer two:a"):
        report.per_item(two, "two")
    assert figures(two, "two:a")[0]["result"] == 1.0


@pytest.mark.skipif(not (EVAL_ENV and MLX_ENV), reason="set MODELLAB_TEST_EVAL_ENV and MODELLAB_TEST_MLX_ENV")
def test_the_report_command_puts_items_before_numbers(lab, tmp_path):
    (lab.root / "envs").mkdir(parents=True)
    (lab.root / "envs" / "mlx").symlink_to(Path(MLX_ENV))
    (lab.root / "envs" / "eval").symlink_to(Path(EVAL_ENV))
    folder = tmp_path / "set"
    folder.mkdir()
    (folder / "task.toml").write_text('[[pane]]\nfield = "question"\nshow = "text"\n')
    (folder / "items.jsonl").write_text("".join(json.dumps({"id": f"q{n}", "question": f"Is {n} even?"}) + "\n" for n in range(6)))
    task = tmp_path / "evenness.py"
    task.write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import Sample\n"
        "from inspect_ai.scorer import match\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def evenness():\n"
        "    return Task(dataset=[Sample(id=f'q{n}', input=f'Is {n} even?', target='yes' if n % 2 == 0 else 'no')\n"
        "                         for n in range(6)], solver=generate(), scorer=match())\n")
    made = subprocess.run([str(Path(EVAL_ENV) / "bin" / "inspect"), "eval", f"{task}@evenness", "--model", "mockllm/model",
                           "--log-dir", str(tmp_path / "logs"), "--display", "none"], capture_output=True, text=True)
    assert made.returncode == 0, made.stdout + made.stderr
    [log] = (tmp_path / "logs").glob("*.eval")
    result = subprocess.run([sys.executable, CLI, "report", str(log), "--set", str(folder), "--to", str(tmp_path / "report")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "report" / "report.md").read_text()
    assert text.index("## Look first") < text.index("## Numbers")
    assert "no better than trivial" in text and "mockllm/model" in text
    misses = (tmp_path / "report" / "lowest" / "sheets.md").read_text()
    assert "Default output from mockllm" in misses  # the raw reply sits next to the parsed answer
    assert str(tmp_path / "report" / "outputs.jsonl") in misses and not list(tmp_path.glob(".report*"))

    def refused(set_folder, to):
        done = subprocess.run([sys.executable, CLI, "report", str(log), "--set", str(set_folder), "--to", str(to)],
                              capture_output=True, text=True)
        assert done.returncode != 0 and not to.exists() and not list(tmp_path.glob(f".{to.name}*"))
        return done.stderr
    clashing = tmp_path / "clashing"
    clashing.mkdir()
    (clashing / "task.toml").write_text('[[pane]]\nfield = "eval_answer"\nshow = "text"\n')
    (clashing / "items.jsonl").write_text("".join(json.dumps({"id": f"q{n}", "eval_answer": "gold"}) + "\n" for n in range(6)))
    assert "would replace" in refused(clashing, tmp_path / "clash-report")
    fewer = tmp_path / "fewer"
    fewer.mkdir()
    (fewer / "task.toml").write_text('[[pane]]\nfield = "question"\nshow = "text"\n')
    (fewer / "items.jsonl").write_text(json.dumps({"id": "q0", "question": "Is 0 even?"}) + "\n")
    assert "not in" in refused(fewer, tmp_path / "fewer-report")
