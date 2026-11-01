"""Download the Open Cities AI Challenge imagery and building labels.

The dataset is published as a STAC catalogue, so nothing here is hard-coded
beyond the catalogue root: the script walks catalog -> collection -> item and
reads each asset's href from the item itself. A city added to the catalogue, or
a scene renamed, is therefore picked up without editing this file.

Two licences apply and they are not the same (see DATA.md):
  imagery  CC BY 4.0
  labels   ODbL-1.0
Both are recorded per file in the manifest this script writes.

Usage
-----
    python -m src.fetch_data --dry-run              # sizes only, no download
    python -m src.fetch_data                        # Dar es Salaam, tier 1
    python -m src.fetch_data --city znz --city acc  # other cities
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urljoin

import requests

CATALOG_ROOT = "https://data.source.coop/open-cities/ai-challenge/"
DEFAULT_TIER = "train_tier_1"
CHUNK = 1 << 20  # 1 MiB
TIMEOUT = 60

# Source Cooperative serves the catalogue anonymously; a session gives us
# connection reuse across the ~25 requests a city needs.
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "dar-building-segmentation/1.0 (research; +https://github.com/WMahendra)"


@dataclass
class Asset:
    """One file to fetch, with the provenance needed to cite it."""

    scene: str
    kind: str  # "image" or "labels"
    url: str
    dest: str  # path relative to the data root
    licence: str
    datetime: str | None
    bbox: list[float] | None


def get_json(url: str) -> dict:
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def child_links(doc: dict, rel: str = "child") -> list[str]:
    return [l["href"] for l in doc.get("links", []) if l.get("rel") == rel]


def discover(tier: str, cities: list[str]) -> list[Asset]:
    """Walk the STAC catalogue and return every asset for the chosen cities."""
    tier_url = urljoin(CATALOG_ROOT, f"{tier}/catalog.json")
    catalog = get_json(tier_url)

    available: dict[str, str] = {}
    for href in child_links(catalog):
        # "./dar/collection.json" -> "dar"
        name = href.strip("./").split("/")[0]
        available[name] = urljoin(tier_url, href)

    unknown = [c for c in cities if c not in available]
    if unknown:
        raise SystemExit(
            f"unknown city code(s) {unknown}; catalogue offers {sorted(available)}"
        )

    assets: list[Asset] = []
    for city in cities:
        collection_url = available[city]
        collection = get_json(collection_url)

        for href in child_links(collection):
            item_url = urljoin(collection_url, href)
            item = get_json(item_url)
            props = item.get("properties", {})
            licence = props.get("license", "unspecified")

            # Imagery items carry an "image" asset, label items a "labels" one.
            for kind in ("image", "labels"):
                asset = (item.get("assets") or {}).get(kind)
                if not asset:
                    continue
                url = urljoin(item_url, asset["href"])
                scene = Path(url).stem
                sub = "images" if kind == "image" else "labels"
                assets.append(
                    Asset(
                        scene=scene,
                        kind=kind,
                        url=url,
                        dest=f"{city}/{sub}/{Path(url).name}",
                        licence=licence,
                        datetime=props.get("datetime"),
                        bbox=item.get("bbox"),
                    )
                )
    return assets


def remote_size(url: str) -> int | None:
    try:
        r = SESSION.head(url, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return int(r.headers["Content-Length"])
    except Exception:
        return None


def human(n: int | None) -> str:
    if n is None:
        return "unknown"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GiB"


def download(url: str, dest: Path) -> tuple[int, str]:
    """Fetch url to dest, returning (bytes, sha256). Resumes nothing: a partial
    file from an interrupted run is written to .part and only moved into place
    once complete, so dest is never a truncated raster."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    total = 0
    with SESSION.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with part.open("wb") as fh:
            for block in r.iter_content(CHUNK):
                fh.write(block)
                digest.update(block)
                total += len(block)
                print(f"\r    {human(total)}", end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)
    part.replace(dest)
    return total, digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--city", action="append", default=None,
                    help="city code, repeatable (default: dar)")
    ap.add_argument("--tier", default=DEFAULT_TIER)
    ap.add_argument("--out", default="data/raw", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="list files and sizes, download nothing")
    args = ap.parse_args()

    cities = args.city or ["dar"]
    print(f"catalogue : {CATALOG_ROOT}{args.tier}", file=sys.stderr)
    print(f"cities    : {', '.join(cities)}", file=sys.stderr)

    assets = discover(args.tier, cities)
    images = [a for a in assets if a.kind == "image"]
    labels = [a for a in assets if a.kind == "labels"]
    print(f"found     : {len(images)} scenes, {len(labels)} label files",
          file=sys.stderr)

    if args.dry_run:
        grand = 0
        for a in assets:
            size = remote_size(a.url)
            grand += size or 0
            print(f"{human(size):>10}  {a.licence:<10}  {a.dest}")
        print(f"{human(grand):>10}  TOTAL")
        return 0

    manifest = []
    for i, a in enumerate(assets, 1):
        dest = args.out / a.dest
        print(f"[{i}/{len(assets)}] {a.dest}", file=sys.stderr)
        if dest.exists():
            size = dest.stat().st_size
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            print(f"    already present ({human(size)})", file=sys.stderr)
        else:
            size, sha = download(a.url, dest)
        manifest.append({**asdict(a), "bytes": size, "sha256": sha})

    args.out.mkdir(parents=True, exist_ok=True)
    mpath = args.out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest  : {mpath}", file=sys.stderr)
    print("Imagery is CC BY 4.0, labels are ODbL-1.0 - see DATA.md before "
          "redistributing either.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
