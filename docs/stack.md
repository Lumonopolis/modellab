# ModelLab stack

What ModelLab uses for each layer, and why. Checked against upstream source and
releases on 2026-09-28; the gold-set rows again on 2026-09-30; the MLX and Inspect AI
rows, Known limits and Patches again on 2026-11-12. Update a row when a choice
changes; keep the reason.

## Choices

| Layer | Piece | Call | Choice | Reason |
|---|---|---|---|---|
| Machines and models | Environments | Adopt | uv, one lockfile | Pins Python and packages; exports to `pylock.toml`. pixi and conda lag PyPI for MLX. |
| | Model store | Adopt + glue | Hugging Face cache and `models.lock` | MLX-LM, MLX-VLM, Transformers and llama.cpp read it. The lock records repo, commit, quantization and weight hashes. |
| | Serving | Adopt + glue | `mlx_lm.server` (text), `mlx_vlm.server` (vision) | Log-probabilities, batching, per-request adapters, same store. |
| | Jobs | Build (thin) | ModelLab runner | Must queue by memory, guard each job, and work inside a host sandbox that blocks `/bin/sh`. pueue has no memory admission and needs a daemon and a shell. SkyPilot, dstack and Ray target Linux. |
| | Memory guard | Build (thin) | Budgets, in-process caps, per-job supervisor | No mature upstream tool. mlx-guard is a design reference. |
| | Two-Mac pooling | Defer | Two independent workers | JACCL needs recoveryOS access on each Mac. No memory gain for training. |
| | Run records | Build (tiny) | Plain run folders; Trackio considered as a viewer, not installed | MLflow sends usage data by default. Aim is stale. |
| Gold sets | Labelling | Build | Config-driven labelling page (`modellab label`), standard library only | Must cover standard labelling work with a minimal, keyboard-first interface. Label Studio: usage data on by default, not mobile friendly, an account-takeover advisory in 2026-01. Argilla: no release since 2025-03, no new features planned. Potato: GPL; four advisories in 2026-08 and 09, one a critical remote code execution. FiftyOne: spatial labels only; MongoDB; usage data on; a 2026-07 advisory (any website could read its local server). CVAT: spatial only, about 20 services. Prodigy: the closest fit, but paid per seat, so every fork must buy it. |
| | Versioning | Build (tiny) | JSONL, media by content hash, `versions/<id>/` with a manifest | The set ID is the manifest hash. Parquet bytes, `datasets` fingerprints and DVC hashes are not stable content IDs. Items map one to one onto Inspect AI's `Sample`; a Croissant export can come later, for sharing. |
| | Sheets for agents | Build (thin) | `modellab look` on Pillow and OpenCV | PNG sheets sized for current Claude and GPT image inputs (1600 × 1600 by default). ImageMagick is not assumed; Homebrew's ffmpeg cannot draw text. |
| Evaluation | Harness | Adopt | Inspect AI, pinned in its own environment (`envs/eval`) | Local by default (telemetry only through a hook you supply); vision; stores log-probabilities; best run provenance. 86 locked packages and near-daily releases, so it stays apart from the MLX lock. |
| | Reports | Build (thin) | `modellab report` over Inspect logs, sheets by `look` | Items before numbers, constant-answer baselines and a paired bootstrap; Inspect's viewer shows numbers and transcripts, not the items beside their targets. |
| | Prime Intellect path | Later | Verifiers, for export only | Defaults to hosted inference, result upload, a remote sandbox and a hosted judge. |
| | Calibration | Later | Kev metrics (Apache-2.0), not installed | Calibration error, temperature fit, risk-coverage, paired bootstrap. |
| | Colour difference | Build (thin) | `modellab colour`: CIEDE2000 on numpy, profiles converted with Pillow | Tested against Sharma et al. (2005), all 34 pairs. colour-science and scikit-image would add a dependency for one formula. Pillow's own LAB mode stores a and b unsigned (negative values read near 255); one project hit this on 2026-10-01 and wrote its own conversion. |
| Training | Text | Adopt | MLX-LM's trainer (`examples/finetune_text.py`); mlx-lm-lora for DPO and GRPO when needed, not installed | MLX-LM has no preference or RL training. |
| | Vision | Adopt | MLX-VLM | Loading, preprocessing, SFT and ORPO. |
| | Custom heads | Glue | Own loss and batching through MLX-LM `train()` | No forks; no patches beyond the rule under Patches. Store custom tensors apart from LoRA weights. |
| | Fallback | Later | TRL and PEFT on PyTorch MPS, not installed | For PyTorch-only objectives. |

## Defaults to override

- Verifiers: set a local `base_url`, `push=false`, a local runtime and a local judge.
- `mlx_vlm.server`: binds `0.0.0.0`. Always pass `--host 127.0.0.1`.
- Label Studio: `COLLECT_ANALYTICS=false`. MLflow: `MLFLOW_DISABLE_TELEMETRY=true`.
  vllm-metal: `VLLM_NO_USAGE_STATS=1`.
- Hugging Face at run time: `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`.
- MLX: the memory limit is soft and the cache limit defaults to it. Set both per
  job. MLX-LM, MLX-VLM and mlx-lm-lora raise the wired limit themselves; ModelLab
  clamps these calls.
- MLX-LM 0.32.2: `--clear-cache-threshold` is parsed but not applied
  (ml-explore/mlx-lm#2000, merged 2026-10-29, not yet released). Set it through the Python API.

## Known limits

- MLX-LM went five months without a release (0.31.3, 2026-04-22, to 0.32.0, 2026-10-01).
  Pin git commits when needed.
- MLX trainers "resume" by reloading weights only; optimizer state restarts (MLX-VLM 0.7.12's
  `--resume-adapter-file` says so). ModelLab's training recipe resumes exactly
  (`docs/training.md`).
- mlx-lm-lora, mlx-guard and Kev each have one maintainer or are very young.
- macOS has no kernel per-process memory cap without root. `taskpolicy -m` and a
  spawn-time jetsam limit did not stop a 409 MB child (2026-09-29).
- MLX GPU buffers count in a process's physical footprint, also with the wired
  limit raised (2026-09-29, M4 Max, macOS 26.4.1, MLX 0.30.3). An outside guard
  can see them.
- MLX 0.32.1 and later refuse a gradient through gather indices ("[gather_axis] Cannot
  calculate VJP with respect to indices"). A mixture-of-experts router that picks experts with
  `argpartition` needs `mx.stop_gradient` on the indices to train. MLX-LM 0.32.0 and MLX-VLM
  0.7.5 fixed their MoE routers; check a new MoE architecture with a fit step.
- MLX-VLM 0.7.12 `SPMStreamingDetokenizer` (Gemma-family tokenizers): `stream_generate`'s
  `.text` holds back everything since the last space until `finalize`; compact JSON arrives at
  the end. A sampler that stops on a pattern in `.text` stops late or never; decode the token
  ids. Qwen's byte-level BPE tokenizers stream per token since 0.7.11.
- MLX-LM 0.32.2 `server`: `--adapter-path` applies only to requests for `default_model`. A
  request that names the model's path loads the base model.

## Patches

ModelLab does not fork or patch upstream packages, with one exception: a small fix that the
upstream project or its sibling project has already accepted. Such a patch lives in
`envs/<env>/patches/<distribution>-<version>-<what>.patch`; `modellab env sync` applies it only
to that exact version, on every host, and stops when the version changes; every run records it
beside the library version (`libraries.jsonl`); it is deleted once a release carries the fix.

None at present.

## Open checks

1. Done 2026-10-01: Inspect AI runs real image evaluations against `modellab serve` (two
   images per item, four requests in flight), logs per item without images, and re-scores
   saved logs. It needs its own pinned environment (`envs/eval`); see `docs/evaluation.md`.
2. Server top-k log-probabilities against direct forward-pass probabilities.
3. A vision model through MLX-LM `train()` with a custom loss.

