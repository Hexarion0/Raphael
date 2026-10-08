"""Fetch the pinned production Chatterbox Turbo model files."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

TURBO_REVISION = "749d1c1a46eb10492095d68fbcf55691ccf137cd"

CANDIDATES = {
    "chatterbox-turbo": (
        "ResembleAI/chatterbox-turbo",
        [
            "t3_turbo_v1.safetensors",
            "s3gen_meanflow.safetensors",
            "ve.safetensors",
            "*.json",
            "*.txt",
        ],
    ),
}


def main() -> None:
    """Pin repository revision and inventory sizes before a bounded download."""
    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=CANDIDATES)
    parser.add_argument("--root", type=Path, default=Path("data/voice/models"))
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true", help="Fetch just speaker encoder")
    parser.add_argument("--max-bytes", type=int, default=6_000_000_000)
    args = parser.parse_args()
    repo, patterns = CANDIDATES[args.candidate]
    directory = args.root / args.candidate
    directory.mkdir(parents=True, exist_ok=True)
    inventory_path = directory / "inventory.json"
    inventory = {}
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
    if inventory.get("repository") != repo or inventory.get("revision") != TURBO_REVISION:
        info = HfApi().model_info(repo, revision=TURBO_REVISION, files_metadata=True)
        files = [
            {"path": s.rfilename, "bytes": s.size or 0}
            for s in info.siblings
            if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)
        ]
        inventory = {
            "repository": repo,
            "revision": info.sha,
            "files": files,
            "selected_bytes": sum(f["bytes"] for f in files),
        }
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory, indent=2), flush=True)
    if args.inventory_only:
        return
    files = inventory["files"]
    if args.analysis_only:
        files = [f for f in files if f["path"] == "ve.safetensors"]
    if sum(f["bytes"] for f in files) > args.max_bytes:
        parser.error(
            "Selected files exceed the download budget; inspect inventory before proceeding"
        )
    for file in files:
        hf_hub_download(
            repo,
            file["path"],
            revision=inventory["revision"],
            local_dir=directory,
            cache_dir=args.root.parent / "cache/huggingface",
        )
        print(f"Verified download: {file['path']}", flush=True)


if __name__ == "__main__":
    main()
