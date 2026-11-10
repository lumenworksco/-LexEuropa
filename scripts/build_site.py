"""Assemble the GitHub Pages site into `_site/`.

    python scripts/build_site.py                # assemble from site/ and the real interface page
    python scripts/build_site.py --refresh-info # also regenerate site/demo/info.json from the ingested corpus

The demo is the real interface (lexeuropa/web_static/index.html) served from a public host, where it
switches itself into static mode: a built-in example answer, no backend, nothing sent anywhere.
`info.json` (the corpus listing the interface's "What it covers" dialog reads) is committed, so the
Pages workflow needs no ingested data; refresh it whenever the corpus changes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = ROOT / "_site"


def refresh_info() -> None:
    sys.path.insert(0, str(ROOT))
    from lexeuropa.web import build_info

    info = build_info()
    (SITE / "demo" / "info.json").write_text(json.dumps(info, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"info.json: {len(info['instruments'])} instruments, {info['chunks']:,} chunks")


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "demo").mkdir(parents=True)
    for name in ("index.html", "og.png", "CNAME"):
        if (SITE / name).exists():
            shutil.copy(SITE / name, OUT / name)
    shutil.copy(ROOT / "lexeuropa" / "web_static" / "index.html", OUT / "demo" / "index.html")
    shutil.copy(SITE / "demo" / "info.json", OUT / "demo" / "info.json")
    if (SITE / "demo" / "gallery.json").exists():            # the recorded answers (see scripts/build_gallery.py)
        shutil.copy(SITE / "demo" / "gallery.json", OUT / "demo" / "gallery.json")
    (OUT / ".nojekyll").write_text("")
    print(f"built {OUT} ({sum(1 for p in OUT.rglob('*') if p.is_file())} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-info", action="store_true")
    args = parser.parse_args()
    if args.refresh_info:
        refresh_info()
    build()
