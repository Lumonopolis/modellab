"""Ask a pulled text model one question, with or without a LoRA adapter, as a guarded job:

    modellab run --budget 2G --model smol -- python examples/ask.py "When does the Copper Kettle open?"
    modellab run --budget 2G --model smol -- python examples/ask.py "When does the Copper Kettle open?" \
        ~/modellab/runs/<run-id>/out/adapters

The answer is in the run's log: `modellab wait <run-id>`, then `modellab logs <run-id>`.
"""

import os
import subprocess
import sys

question, *adapter = sys.argv[1:]
command = [sys.executable, "-m", "mlx_lm", "generate", "--model", os.environ["MODELLAB_MODEL"],
           "--prompt", question, "--max-tokens", "100", "--temp", "0"]
if adapter:
    command += ["--adapter-path", adapter[0]]
sys.exit(subprocess.call(command))
