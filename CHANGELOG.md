# Changelog

## Unreleased

- Continuous integration: the tests run on macOS (Apple Silicon) with Python 3.11 and 3.14
  for every pull request and every push to `main`.

## 0.1.0, 2026-10-20

First public release.

- `examples/finetune_text.py` and `examples/ask.py`: a first LoRA fine-tune of a small text
  model with MLX-LM, as a guarded job, and a way to ask it with or without the adapter. The
  README walks through them.
- `modellab run` and `serve` print the run id alone on standard output; the first state goes
  to standard error.
- Documentation: a Limits section for the guard, budget guidance, and scoped claims for exact
  resume. The recipe is for vision-language models, and now says so.
- Known limit recorded: MLX-LM 0.31.3's server ignores `--adapter-path`.

## Initial import, 2026-10-20

ModelLab's code, environments, documentation and tests, imported with fresh history.
