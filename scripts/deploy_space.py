"""Build and upload the public demo Space (iPwnds/lexeuropa-demo).

    python scripts/deploy_space.py            # stage only, print what would be uploaded
    python scripts/deploy_space.py --upload   # create the Space if needed and upload

Needs the ingested corpus (data/processed, data/index) and a Hugging Face login with write access. The Space gets
the Dockerfile (pinned to the current git commit, which must be pushed), a README with the OAuth settings, and the
library as three files; it never receives a token or any other secret (Hugging Face injects the OAuth client
id and secret itself because the README says `hf_oauth: true`).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_ID = "iPwnds/lexeuropa-demo"


def stage(dest: Path) -> None:
    sys.path.insert(0, str(ROOT))
    import numpy as np

    from lexeuropa.config import CHUNKS_PATH, DENSE_INDEX_PATH, INSTRUMENTS_PATH
    from lexeuropa.ingest.pipeline import load_chunks
    from lexeuropa.retrieval.dense_index import DenseIndex

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dockerfile = (ROOT / "space" / "Dockerfile").read_text().replace("ARG LEXEUROPA_REF=main", f"ARG LEXEUROPA_REF={sha}")
    (dest / "Dockerfile").write_text(dockerfile)
    shutil.copy(ROOT / "space" / "README.md", dest / "README.md")
    (dest / "data" / "processed").mkdir(parents=True)
    (dest / "data" / "index").mkdir(parents=True)
    shutil.copy(CHUNKS_PATH, dest / "data" / "processed" / "chunks.jsonl")
    shutil.copy(INSTRUMENTS_PATH, dest / "data" / "processed" / "instruments.jsonl")
    # Re-save the dense index so the ids are plain unicode (loadable without pickle).
    chunks = load_chunks()
    DenseIndex.load(DENSE_INDEX_PATH, {c.id: c for c in chunks}).save(dest / "data" / "index" / "dense.npz")
    with np.load(dest / "data" / "index" / "dense.npz", allow_pickle=False) as z:
        assert z["ids"].dtype.kind == "U", "ids must be unicode"
    print(f"staged commit {sha[:10]}: {len(chunks):,} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        stage(dest)
        files = sorted(p.relative_to(dest) for p in dest.rglob("*") if p.is_file())
        print("\n".join(f"  {f}  {(dest / f).stat().st_size / 1e6:.1f} MB" for f in files))
        if args.upload:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(SPACE_ID, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
            api.upload_folder(folder_path=str(dest), repo_id=SPACE_ID, repo_type="space", commit_message="Deploy the LexEuropa live demo")
            print(f"uploaded: https://huggingface.co/spaces/{SPACE_ID}")
