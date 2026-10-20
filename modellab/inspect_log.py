"""One Inspect AI log as plain rows: what was asked of each item, what came back, how it scored.

Runs inside the eval environment (Inspect AI): `modellab report` starts it. The rows carry the
raw reply next to each scorer's parsed answer, so a report can show both.
"""

import json
import sys

from inspect_ai.log import read_eval_log
from inspect_ai.scorer import value_to_float

to_float = value_to_float()


def main(request):
    log = read_eval_log(request["log"])
    if log.status != "success":
        raise SystemExit(f"{request['log']}: the evaluation ended {log.status!r}; report a finished one")
    rows, scorers = [], {}
    for sample in log.samples or []:
        targets = [sample.target] if isinstance(sample.target, str) else list(sample.target)
        scores = {}
        for name, score in (sample.scores or {}).items():
            scorers[name] = None  # names as the items carry them, in order
            several = isinstance(score.value, dict)
            scores[name] = {"value": None if several else number(score.value),
                            "values": {key: number(value) for key, value in score.value.items()} if several else None,
                            "raw": score.value if isinstance(score.value, (str, int, float, bool)) else json.dumps(score.value),
                            "answer": score.answer}
        rows.append({"id": str(sample.id), "epoch": sample.epoch, "targets": targets, "target": " | ".join(targets),
                     "reply": sample.output.completion if sample.output else "", "scores": scores,
                     "error": sample.error.message if sample.error else None})
    return {"task": log.eval.task, "model": log.eval.model, "created": log.eval.created,
            "inspect": log.eval.packages.get("inspect_ai"), "metadata": log.eval.metadata or {},
            "scorers": list(scorers), "rows": rows}


def number(value):
    """A score as a number (C, I, P, N and booleans as Inspect counts them), or None."""
    return to_float(value) if isinstance(value, (str, int, float, bool)) else None


if __name__ == "__main__":
    print(json.dumps(main(json.loads(sys.argv[1]))))
