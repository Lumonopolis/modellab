"""Hugging Face Hub operations. They need huggingface_hub, so they run inside a ModelLab
environment, one request per process:

    <env>/bin/python -m modellab.hub '{"op": "pull", "repo": "...", "revision": "..."}'

The result is one line of JSON on stdout. Progress bars and errors go to stderr.
"""

import json
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError


def resolve(repo, revision):
    """The repository's canonical name, the commit a revision points to, and its config.json."""
    info = HfApi().model_info(repo, revision=revision)
    with open(hf_hub_download(info.id, "config.json", revision=info.sha)) as handle:
        return {"repo": info.id, "revision": info.sha, "config": json.load(handle)}


def pull(repo, revision):
    path = snapshot_download(repo, revision=revision)
    check = HfApi().verify_repo_checksums(repo, revision=revision)
    if check.mismatches or check.missing_paths or check.extra_paths:
        raise SystemExit(f"{repo}@{revision} failed verification: mismatches {check.mismatches}, "
                         f"missing {check.missing_paths}, extra {check.extra_paths}")
    return {"path": path, "checked": check.checked_count}


if __name__ == "__main__":
    request = json.loads(sys.argv[1])
    operation = {"resolve": resolve, "pull": pull}[request.pop("op")]
    try:
        print(json.dumps(operation(**request)))
    except HfHubHTTPError as error:
        sys.exit(str(error).splitlines()[0])
