# ModelLab stack

What ModelLab uses for each layer, and why. Checked against upstream source and
releases on 2026-09-28; the gold-set rows again on 2026-09-30. Update a row when
a choice changes; keep the reason.

## Choices

| Layer | Piece | Call | Choice | Reason |
|---|---|---|---|---|
| Machines and models | Environments | Adopt | uv, one lockfile | Pins Python and packages; exports to `pylock.toml`. pixi and conda lag PyPI for MLX. |
| | Model store | Adopt + glue | Hugging Face cache and `models.lock` | MLX-LM, MLX-VLM, Transformers and llama.cpp read it. The lock records repo, commit, quantization and weight hashes. |
| | Serving | Adopt + glue | `mlx_lm.server` (text), `mlx_vlm.server` (vision) | Log-probabilities, batching, per-request adapters, same store. |
| | Jobs | Build (thin) | ModelLab runner | Must queue by memory, guard each job, and work inside a host sandbox that blocks `/bin/sh`. pueue has no memory admission and needs a daemon and a shell. SkyPilot, dstack and Ray target Linux. |
| | Memory guard | Build (thin) | Budgets, in-process caps, per-job supervisor | No mature upstream tool. mlx-guard is a design reference. |
| | Two-Mac pooling | Defer | Two independent workers | JACCL needs recoveryOS access on each Mac. No memory gain for training. |
| | Run records | Adopt (light) | Plain run folders; Trackio as optional viewer | MLflow sends usage data by default. Aim is stale. |
| Gold sets | Labelling | Build | Config-driven labelling page (`modellab label`), standard library only | Must cover standard labelling work with a minimal, keyboard-first interface. Label Studio: usage data on by default, not mobile friendly, an account-takeover advisory in 2026-01. Argilla: no release since 2025-03, no new features planned. Potato: GPL; four advisories in 2026-08 and 09, one a critical remote code execution. FiftyOne: spatial labels only; MongoDB; usage data on; a 2026-07 advisory (any website could read its local server). CVAT: spatial only, about 20 services. Prodigy: the closest fit, but paid per seat, so every fork must buy it. |
| | Versioning | Build (tiny) | JSONL, media by content hash, `versions/<id>/` with a manifest | The set ID is the manifest hash. Parquet bytes, `datasets` fingerprints and DVC hashes are not stable content IDs. Items map one to one onto Inspect AI's `Sample`; a Croissant export can come later, for sharing. |
| | Sheets for agents | Build (thin) | `modellab look` on Pillow and OpenCV | PNG sheets sized for current Claude and GPT image inputs (1600 × 1600 by default). ImageMagick is not installed; Homebrew's ffmpeg cannot draw text. |
| Evaluation | Harness | Adopt | Inspect AI, pinned in its own environment (`envs/eval`) | Local by default (telemetry only through a hook you supply); vision; stores log-probabilities; best run provenance. 86 locked packages and near-daily releases, so it stays apart from the MLX lock. |
| | Reports | Build (thin) | `modellab report` over Inspect logs, sheets by `look` | Items before numbers, constant-answer baselines and a paired bootstrap; Inspect's viewer shows numbers and transcripts, not the items beside their targets. |
| | Prime Intellect path | Later | Verifiers, for export only | Defaults to hosted inference, result upload, a remote sandbox and a hosted judge. |
| | Calibration | Adopt | Kev metrics (Apache-2.0) | Calibration error, temperature fit, risk-coverage, paired bootstrap. |
| | Colour difference | Build (thin) | `modellab colour`: CIEDE2000 on numpy, profiles converted with Pillow | Tested against Sharma et al. (2005), all 34 pairs. colour-science and scikit-image would add a dependency for one formula. Pillow's own LAB mode stores a and b unsigned (negative values read near 255); one project hit this on 2026-10-01 and wrote its own conversion. |
| Training | Text | Adopt | MLX-LM `train()`; mlx-lm-lora for DPO and GRPO | MLX-LM has no preference or RL training. |
| | Vision | Adopt | MLX-VLM | Loading, preprocessing, SFT and ORPO. |
| | Custom heads | Glue | Own loss and batching through MLX-LM `train()` | No forks; no patches beyond the rule under Patches. Store custom tensors apart from LoRA weights. |
| | Fallback | Keep | TRL and PEFT on PyTorch MPS | For PyTorch-only objectives. |

## Defaults to override

- Verifiers: set a local `base_url`, `push=false`, a local runtime and a local judge.
- `mlx_vlm.server`: binds `0.0.0.0`. Always pass `--host 127.0.0.1`.
- Label Studio: `COLLECT_ANALYTICS=false`. MLflow: `MLFLOW_DISABLE_TELEMETRY=true`.
  vllm-metal: `VLLM_NO_USAGE_STATS=1`.
- Hugging Face at run time: `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`.
- MLX: the memory limit is soft and the cache limit defaults to it. Set both per
  job. MLX-LM, MLX-VLM and mlx-lm-lora raise the wired limit themselves; ModelLab
  clamps these calls.
- MLX-LM: `--clear-cache-threshold` is parsed but not applied (0.31.3). Set it
  through the Python API.

## Known limits

- MLX-LM went five months without a release (0.31.3, 2026-04-22, to 0.32.0, 2026-10-01).
  Pin git commits when needed. ModelLab's environment still has 0.31.3.
- MLX trainers "resume" by reloading weights only; optimizer state restarts. ModelLab's
  training recipe resumes exactly (`docs/training.md`).
- mlx-lm-lora, mlx-guard and Kev each have one maintainer or are very young.
- macOS has no kernel per-process memory cap without root. `taskpolicy -m` and a
  spawn-time jetsam limit did not stop a 409 MB child (2026-09-29).
- MLX GPU buffers count in a process's physical footprint, also with the wired
  limit raised (2026-09-29, M4 Max, macOS 26.4.1, MLX 0.30.3). An outside guard
  can see them.
- 2026-10-01: LoRA training of mixture-of-experts models fails on the first backward step
  with "[gather_axis] Cannot calculate VJP with respect to indices" (Qwen3.6-35B-A3B,
  on the M5).
  - Cause: MLX 0.32.1 (PR #3820) turned the silent zero gradient through gather indices into
    this error, as intended. MoE routers that pick experts with `argpartition` and gather
    their scores now need `mx.stop_gradient` on the indices. MLX 0.32.0 trains the same code.
  - MLX-LM fixed its routers that way (#1787, #1856), released in 0.32.0 on 2026-10-01.
    ModelLab's environment still has MLX-LM 0.31.3, so its MoE text models fail too.
  - MLX-VLM 0.7.3 and 0.7.4 have not; reported as Blaizzy/mlx-vlm#2407, fixed in 0.7.5
    (2026-10-05, #2409). ModelLab's patch for 0.7.3 was applied on both Macs on 2026-10-01;
    runs record it in `libraries.jsonl`.
  - Checked: on the real model's architecture, shrunk, one LoRA step fails as shipped and
    trains with only the router line changed; the forward pass is unchanged. On MLX 0.32.2
    with the fix, gradients equal MLX 0.32.0's without it (bit for bit) and match finite
    differences.
  - ModelLab patches MLX-VLM 0.7.3's Qwen MoE routers (see Patches). With the patch, the
    real Qwen3.6-35B-A3B 8-bit trains (a fit check on the M4 Max: 5,529 tokens with two
    images, rank 16, about 96 s per step at 57.7 tokens/s, 60.4 GB peak, loss 0.659). The
    dense 27B 8-bit trains at about 20 tokens/s on the M4 Max and 33.5 on the M5, 55.5 GB
    peak on the same example (42 GB at 4-bit).
- 2026-10-01, MLX-VLM 0.7.3 `sft_trainer`: the last, partial report of a run divides
  `steps_per_report` (not the steps actually run) by the elapsed time, so a one-step fit check
  shows `It/sec` ten times too high with the default report interval of 10. Estimate step time
  from `Trained Tokens` / `Tokens/sec` instead.
- 2026-10-02, MLX-VLM 0.7.3 `BPEStreamingDetokenizer` (Qwen's tokenizers): `add_token` passes
  text on only when the next token starts with a space, so `stream_generate`'s `.text` holds
  back everything since the last space until `finalize`. Compact JSON has almost no spaces:
  after 200 tokens of a training reply, 175 of 453 characters had arrived. A sampler that
  stops on a pattern in `.text` stops late or never. Decode the collected `result.token` ids
  instead.

## Patches

ModelLab does not fork or patch upstream packages, with one exception: a small fix that the
upstream project or its sibling project has already accepted. Such a patch lives in
`envs/<env>/patches/<distribution>-<version>-<what>.patch`; `modellab env sync` applies it only
to that exact version, on every host, and stops when the version changes; every run records it
beside the library version (`libraries.jsonl`); it is deleted once a release carries the fix.

| Patch | Fixes | Accepted upstream as | Delete when |
|---|---|---|---|
| `mlx-vlm-0.7.3-moe-router-stop-gradient.patch` | LoRA training of Qwen MoE models (qwen3_5_moe, qwen3_moe, qwen3_next, qwen3_vl_moe, qwen3_omni_moe) on MLX 0.32.1 and later; see Known limits | MLX-LM #1787 and #1856; MLX-VLM #2409, released in 0.7.5 | the environment moves to MLX-VLM 0.7.5 or later |

## Open checks

1. Done 2026-10-01: Inspect AI runs real image evaluations against `modellab serve` (two
   images per item, four requests in flight), logs per item without images, and re-scores
   saved logs. It needs its own pinned environment (`envs/eval`); see `docs/evaluation.md`.
2. Server top-k log-probabilities against direct forward-pass probabilities.
3. A vision model through MLX-LM `train()` with a custom loss.

