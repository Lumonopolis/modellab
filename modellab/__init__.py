"""ModelLab: a forkable workbench for local model work on Apple Silicon Macs."""

import subprocess
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent



def revision():
    """Which ModelLab code is running: REVISION where it was pushed, else the last Git commit
    that changed ModelLab, plus "+dirty" when its code differs from that commit."""
    pushed = HOME / "REVISION"
    if pushed.exists():
        return pushed.read_text().strip()
    last = subprocess.run(["git", "-C", str(HOME), "log", "-1", "--format=%H", "--", "."],
                          capture_output=True, text=True)
    if last.returncode != 0 or not last.stdout.strip():
        return "unknown"
    dirty = subprocess.run(["git", "-C", str(HOME), "status", "--porcelain", "--", "."], capture_output=True, text=True)
    return last.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
