"""Build the results table and figures from whatever arms have been run.

Reads results/*/metrics.json and writes:

  results/results_table.md      the table the README embeds
  results/results_table.csv     same numbers, for reuse
  figures/pr_curves.png         precision-recall, all arms on one axis
  figures/threshold_tradeoff.png  precision and recall vs threshold
  figures/qualitative_<scene>.png imagery, reference, and predictions

Every number is read from a metrics.json produced by an actual run. Nothing in
this module invents, interpolates or fills a value, so an arm that has not been
run is simply absent from the table rather than estimated into it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORDER_HINT = [
    "bce", "weighted_bce", "dice", "tversky", "focal", "bce_dice",
    "oversample", "oversample_dice",
    "pixel_plain", "pixel_smote", "pixel_smote_adaboost",
]


def load_results(results_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(results_dir.glob("*/metrics.json")):
        out[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
    if not out:
        raise SystemExit(f"no metrics.json under {results_dir}; run the arms first")
    return out


def ordered(names) -> list[str]:
    known = [n for n in ORDER_HINT if n in names]
    return known + sorted(n for n in names if n not in known)


def results_table(res: dict[str, dict]) -> tuple[str, str]:
    header = ("| arm | AP | best F1 | threshold | precision | recall | IoU | "
              "F1 @ 0.5 |")
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    lines, csv = [header, rule], ["arm,ap,best_f1,threshold,precision,recall,iou,f1_at_0.5"]
    for name in ordered(res):
        r = res[name]
        b, h = r["best_f1"], r["at_0.5"]
        lines.append(
            f"| `{name}` | {r['average_precision']:.3f} | {b['f1']:.3f} | "
            f"{b['threshold']:.2f} | {b['precision']:.3f} | {b['recall']:.3f} | "
            f"{b['iou']:.3f} | {h['f1']:.3f} |")
        csv.append(f"{name},{r['average_precision']:.4f},{b['f1']:.4f},"
                   f"{b['threshold']:.2f},{b['precision']:.4f},{b['recall']:.4f},"
                   f"{b['iou']:.4f},{h['f1']:.4f}")
    return "\n".join(lines), "\n".join(csv)


def pr_curves(res: dict[str, dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=160)
    for name in ordered(res):
        s = res[name]["sweep"]
        r = np.asarray(s["recall"])
        p = np.asarray(s["precision"])
        o = np.argsort(r)
        style = "--" if name.startswith("pixel_") else "-"
        ax.plot(r[o], p[o], style, lw=1.6, label=
                f"{name} (AP {res[name]['average_precision']:.2f})")

    base = res[ordered(res)[0]].get("positive_pixel_fraction")
    if base:
        ax.axhline(base, color="0.6", lw=0.9, ls=":")
        ax.annotate(f"random classifier (P = {base:.3f})", (0.99, base + 0.015),
                    fontsize=7, color="0.45", ha="right", va="bottom")

    ax.set_xlabel("recall (building class)")
    ax.set_ylabel("precision (building class)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    ax.set_title("Precision-recall on the held-out scene", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def threshold_tradeoff(res: dict[str, dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ordered(res) if not n.startswith("pixel_")]
    if not names:
        names = ordered(res)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=160, sharey=True)
    for name in names:
        s = res[name]["sweep"]
        t = np.asarray(s["threshold"])
        axes[0].plot(t, np.asarray(s["precision"]), lw=1.4, label=name)
        axes[1].plot(t, np.asarray(s["recall"]), lw=1.4, label=name)
    for ax, title in zip(axes, ("precision", "recall")):
        ax.set_xlabel("decision threshold")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("score (building class)")
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("What each arm trades away as the threshold moves", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def qualitative(res: dict[str, dict], cfg: dict, out_dir: Path,
                max_arms: int = 3, side: int = 1024) -> list[Path]:
    """Imagery, reference mask and predictions over the same window.

    One crop from the held-out scene, taken where the reference has the most
    buildings so the panel shows the hard part rather than empty ground.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import Window

    from .raster import scale_to_unit

    written = []
    scenes = sorted({s for r in res.values() for s in r.get("test_scenes", [])})
    arms = [n for n in ordered(res)][:max_arms]

    for scene in scenes:
        img = Path(cfg["raw_dir"]) / cfg["city"] / "images" / f"{scene}.tif"
        msk = Path(cfg["processed_dir"]) / cfg["city"] / "masks" / f"{scene}.tif"
        if not img.exists() or not msk.exists():
            continue

        with rasterio.open(msk) as m:
            side = min(side, m.height, m.width)
            # Coarse search for the densest window, on a decimated read so this
            # stays cheap on a large scene.
            step = max(side // 2, 1)
            best, best_win = -1.0, Window(0, 0, side, side)
            for row in range(0, m.height - side + 1, step):
                for col in range(0, m.width - side + 1, step):
                    w = Window(col, row, side, side)
                    frac = float((m.read(1, window=w) > 0).mean())
                    if frac > best:
                        best, best_win = frac, w
            ref = m.read(1, window=best_win)

        with rasterio.open(img) as s:
            rgb = scale_to_unit(
                np.transpose(s.read(indexes=[1, 2, 3], window=best_win), (1, 2, 0)),
                s.dtypes[0])

        panels = [("imagery", rgb, None), ("reference", ref, "ref")]
        for name in arms:
            p = Path(cfg["results_dir"]) / name / f"prediction_{scene}.tif"
            if not p.exists():
                continue
            with rasterio.open(p) as pr:
                q = pr.read(1, window=best_win).astype("float32")
            q = np.where(q == 255, np.nan, q / 254.0)
            panels.append((name, q, "prob"))

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), dpi=160)
        axes = np.atleast_1d(axes)
        for ax, (title, arr, kind) in zip(axes, panels):
            if kind is None:
                ax.imshow(np.clip(arr, 0, 1))
            elif kind == "ref":
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(arr, cmap="magma", vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{scene} - held-out scene, {best:.1%} building pixels in "
                     f"this {side}x{side} crop", fontsize=9)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"qualitative_{scene}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config/config.yaml", type=Path)
    ap.add_argument("--figures", default="figures", type=Path)
    ap.add_argument("--no-qualitative", action="store_true",
                    help="skip the image panels (they need the rasters present)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results_dir = Path(cfg["results_dir"])
    res = load_results(results_dir)

    prints = {n: r.get("config_fingerprint", {}).get("hash") for n, r in res.items()}
    distinct = {h for h in prints.values() if h}
    if len(distinct) > 1:
        print("WARNING: these arms were not all run under the same settings, so "
              "the table below is not a like-for-like comparison:")
        for n, h in sorted(prints.items()):
            print(f"  {n:<24} config {h or 'unrecorded'}")
        print()

    md, csv = results_table(res)
    (results_dir / "results_table.md").write_text(md + "\n", encoding="utf-8")
    (results_dir / "results_table.csv").write_text(csv + "\n", encoding="utf-8")
    print(md)

    pr_curves(res, args.figures / "pr_curves.png")
    threshold_tradeoff(res, args.figures / "threshold_tradeoff.png")
    print(f"\nfigures: {args.figures}/pr_curves.png, "
          f"{args.figures}/threshold_tradeoff.png")

    if not args.no_qualitative:
        for p in qualitative(res, cfg, args.figures):
            print(f"figures: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
