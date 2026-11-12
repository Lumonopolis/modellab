# Changelog

## Unreleased

- Continuous integration: the tests run on macOS (Apple Silicon) with Python 3.11 and 3.14
  for every pull request and every push to `main`.
- `run` on a host whose reserve leaves jobs no memory, such as an 8 GB Mac with the default
  reserve, says so and how to fix it, instead of reporting a negative allowance.
- The MLX environment moves to MLX 0.32.4, MLX-LM 0.32.2 and MLX-VLM 0.7.12; the evaluation
  environment to Inspect AI 0.3.289. Run `modellab env sync` and `modellab env sync eval` on
  every host. Results are not bit-identical across the move: MLX 0.32.3 and 0.32.4 changed its
  kernels, and a model's logits move with them.
  - The patch for MLX-VLM 0.7.3's mixture-of-experts routers is deleted: 0.7.5 carries the fix.
  - `modellab serve --adapter` now applies the adapter on text models, for requests to
    `default_model` (MLX-LM 0.32.0 fixed `--adapter-path`).
  - With Qwen's tokenizers, MLX-VLM's streamed text arrives per token. Gemma-family tokenizers
    still hold it back.
  - The recipe lets MLX-VLM choose the adapter layers. The same layers as before, in module
    order; naming them as a list gave each process different starting values.
  - Inspect AI accepts an absolute task-file path.
- Project code written against MLX-VLM 0.7.3's trainer must change:
  - `mlx_vlm.trainer.sft_trainer` is now `mlx_vlm.trainer.vlm.sft.trainer`.
    `mlx_vlm.trainer.lora` and `mlx_vlm.trainer.orpo_trainer` are gone.
  - `find_all_linear_names`, `get_peft_model`, `grad_checkpoint` and `save_adapter` come from
    `mlx_vlm.trainer`, no longer `mlx_vlm.trainer.utils`.
  - `vision_language_loss_fn` returns `(loss, metrics)`, not a scalar. A `loss` given to the
    recipe may return either.
  - `TrainingArgs` has no `learning_rate`, `warmup_steps` or `min_learning_rate`; set the
    learning rate on the optimizer.
  - The trainer's report names its fields (`iterations_per_second`); its `tokens_per_second`
    and `trained_tokens` count only the trained tokens.

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
