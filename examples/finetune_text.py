"""A first fine-tune: LoRA on a small text model with MLX-LM's own trainer, as a guarded job.

Run it through ModelLab, from this checkout, with a pulled text model:

    modellab run --name first-finetune --budget 4G --model smol -- python examples/finetune_text.py

The model is the one given with `--model` ($MODELLAB_MODEL). The data are chat examples, one per
line, in examples/data/train.jsonl and valid.jsonl; pass another folder of the same shape to
train on your own: `... -- python examples/finetune_text.py path/to/data`. The adapter goes to
the run's out/adapters, and MLX-LM's report in the log shows the loss and the peak memory.

MLX-LM resumes only from the adapter's weights, so this job is not --resumable.
"""

import os
import subprocess
import sys
from pathlib import Path

data = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data").resolve()
adapters = Path(os.environ["MODELLAB_RUN_DIR"]) / "out" / "adapters"
# A child process, not exec: this process keeps the job hook's watch on the supervisor.
sys.exit(subprocess.call([sys.executable, "-m", "mlx_lm", "lora", "--model", os.environ["MODELLAB_MODEL"],
                          "--train", "--data", str(data), "--adapter-path", str(adapters),
                          "--iters", "200", "--learning-rate", "2e-4", "--batch-size", "1", "--mask-prompt",
                          "--steps-per-report", "50"]))
