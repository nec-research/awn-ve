#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/make_paper_figures.py
#   Authors:  Federico Errica (federico.errica@neclab.eu)
#             Nithya Bhasker (nithya.bhasker@neclab.eu)
#             Raman Siarheyeu (raman.siarheyeu@neclab.eu)
#             Catherine Chaput (catherine.chaput@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2026, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#

"""Compose paper-ready figures from the experiment result artifacts.

Reads the CSV/JSON/embedding outputs written under ``results/paper/`` (and the
cached embeddings under ``embeddings/``) and renders:

* ``blosum_correlation.png``      - one hexbin per AWN config: embedding pairwise
                                    distance vs true BLOSUM distance (held-out
                                    test fold, N=2000 subsample), Pearson r
                                    annotated against the ESM-2 teacher.
* ``lineage_classification.png``  - left: per-variant F1 for every encoder;
                                    right: confusion matrix of the best encoder.
* ``mutation_heatmap_<cfg>_{best,worst}.png`` - decode mutation-site heatmaps
                                    (Source vs ESM-2 vs AWN) for each config's
                                    best and worst reconstructed test sequence.

Missing inputs are skipped with a warning, so the script is safe to run while
experiments are still completing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Canonical display order and labels.
DEFAULT_ENCODERS = ["esm2", "config_3", "config_1", "config_2"]
DISPLAY = {
    "esm2": "ESM-2 650M",
    "config_1": "AWN-ESM2 65.1M",
    "config_2": "AWN-ESM2 27.6M",
    "config_3": "AWN-ESM2 39.0M",
}
# AWN configs in ascending parameter-count order (compression story).
CONFIG_ORDER = ["config_2", "config_3", "config_1"]
BAR_COLORS = {"esm2": "#444444", "config_1": "#1f77b4", "config_2": "#d62728", "config_3": "#2ca02c"}

# Embedding-cache fingerprint for the current fp32 run (pins the right files).
CACHE_TAG = "be08f5d3db00"
SEED = 42
PLOT_N = 2000
SEQ_LEN = 1300


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# BLOSUM hexbins
# --------------------------------------------------------------------------- #
def _find_cache(cache_root: str, pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(cache_root, "**", pattern), recursive=True))
    return hits[0] if hits else None


def blosum_hexbin_figure(
    fasta: str,
    cache_root: str,
    configs: list[str],
    out: Path,
    *,
    metric: str = "l2",
    n: int = PLOT_N,
    seed: int = SEED,
) -> bool:
    """One hexbin per AWN config: embedding distance vs true BLOSUM distance.

    Recomputed on CPU from the cached pooled embeddings + the held-out test
    sequences (exactly reproduces the evaluate_pairwise correlations).
    """
    import torch  # noqa: WPS433 (local, heavy import)
    from downstream_validation.evaluate_pairwise import (
        correlate,
        cosine_distance_matrix,
        l2_distance_matrix,
        pairwise_blosum_distance_matrix,
        parse_fasta,
        upper_triangle_flat,
    )
    from downstream_validation.splits import aligned_splits

    cfgs = [c for c in CONFIG_ORDER if c in configs]
    if not cfgs:
        print("[blosum] no AWN configs requested; skipping.")
        return False

    labels, seqs = parse_fasta(fasta)
    test = aligned_splits(labels)["test_idxs"]
    rng = random.Random(seed)
    sub = sorted(rng.sample(list(test), min(n, len(test))))
    seqs_sub = [seqs[i][:SEQ_LEN] for i in sub]
    x = upper_triangle_flat(pairwise_blosum_distance_matrix(seqs_sub)).numpy()
    dist_fn = l2_distance_matrix if metric == "l2" else cosine_distance_matrix
    ylab = "L2 embedding distance" if metric == "l2" else "cosine embedding distance"

    def _emb_r(pattern: str):
        path = _find_cache(cache_root, pattern)
        if path is None:
            return None, None, None
        emb = torch.load(path, map_location="cpu")["embeddings"][sub]
        y = upper_triangle_flat(dist_fn(emb)).numpy()
        c = correlate(x, y)
        return y, c["pearson_r"], c["spearman_rho"]

    _, r_teacher, rho_teacher = _emb_r(f"filtered_msa.esm2.*{CACHE_TAG}.pt")

    fig, axes = plt.subplots(1, len(cfgs), figsize=(5.2 * len(cfgs), 4.9), squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        y, r, rho = _emb_r(f"filtered_msa.awn.{cfg}.*{CACHE_TAG}.pt")
        if y is None:
            ax.text(0.5, 0.5, f"no cache for {cfg}", ha="center", va="center")
            ax.axis("off")
            continue
        hb = ax.hexbin(x, y, gridsize=60, mincnt=1, cmap="viridis")
        subtitle = f"Pearson $r$ = {r:.3f}   Spearman $\\rho$ = {rho:.3f}"
        if r_teacher is not None:
            subtitle += f"\n(teacher: $r$ {r_teacher:.3f}, $\\rho$ {rho_teacher:.3f})"
        ax.set_title(f"{DISPLAY.get(cfg, cfg)}\n{subtitle}", fontsize=10)
        ax.set_xlabel("true BLOSUM pairwise distance")
        ax.set_ylabel(ylab)
        fig.colorbar(hb, ax=ax, label="pair count")
    fig.suptitle(
        f"Biological distance preservation ({'L2' if metric == 'l2' else 'cosine'}, "
        f"held-out test fold, N={len(sub)})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[blosum] wrote {out}  (teacher r={r_teacher:.3f})")
    return True


# --------------------------------------------------------------------------- #
# Lineage: per-variant F1 + confusion matrix
# --------------------------------------------------------------------------- #
def _per_class_f1(conf: np.ndarray) -> np.ndarray:
    """Per-class F1 from a confusion matrix (rows=true, cols=pred)."""
    conf = conf.astype(float)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / denom, 0.0)


def lineage_figure(results_dir: Path, encoders: list[str], out: Path) -> bool:
    """Left: per-variant F1 for each encoder. Right: confusion matrix per encoder."""
    f1s: dict[str, np.ndarray] = {}
    confs: dict[str, np.ndarray] = {}
    classes: list[str] | None = None
    for enc in encoders:
        sdir = results_dir / "lineage" / enc
        sjson = sdir / "summary.json"
        cpath = sdir / "confusion_matrix.npy"
        if not sjson.is_file() or not cpath.is_file():
            continue
        s = json.loads(sjson.read_text(encoding="utf-8"))
        conf = np.load(cpath)
        classes = s.get("classes") or classes
        f1s[enc] = _per_class_f1(conf)
        confs[enc] = conf

    present = [e for e in encoders if e in f1s]
    if not present or classes is None:
        print("[lineage] no metrics found; skipping.")
        return False

    n_cls = len(classes)
    # Left half = F1 bars (spans both rows); right 2x2 = per-encoder confusions.
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1.1, 1, 1], hspace=0.55, wspace=0.45)
    ax_f1 = fig.add_subplot(gs[:, 0:2])

    # Per-variant F1 as an annotated encoder x variant heatmap (teacher first).
    row_order = [e for e in ["esm2", "config_1", "config_3", "config_2"] if e in present]
    mat = np.array([f1s[e][:n_cls] for e in row_order])
    im0 = ax_f1.imshow(mat, cmap="RdYlGn", vmin=0.25, vmax=1.0, aspect="auto")
    ax_f1.set_xticks(range(n_cls))
    ax_f1.set_xticklabels(classes, rotation=30, ha="right")
    ax_f1.set_yticks(range(len(row_order)))
    ax_f1.set_yticklabels([DISPLAY.get(e, e) for e in row_order])
    for i in range(len(row_order)):
        for j in range(n_cls):
            ax_f1.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                       fontsize=9, color="black")
    ax_f1.set_title("Lineage classification: per-variant F1 (held-out test fold)")
    fig.colorbar(im0, ax=ax_f1, fraction=0.035, pad=0.02, label="F1")

    # Confusion matrices for every encoder (teacher first, then AWN by size desc).
    conf_order = [e for e in ["esm2", "config_1", "config_3", "config_2"] if e in confs]
    slots = [gs[0, 2], gs[0, 3], gs[1, 2], gs[1, 3]]
    im = None
    for enc, slot in zip(conf_order, slots):
        ax = fig.add_subplot(slot)
        conf = confs[enc].astype(float)
        norm = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(DISPLAY.get(enc, enc), fontsize=9)
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
    if im is not None:
        fig.colorbar(im, ax=fig.axes[1:], fraction=0.025, pad=0.02,
                     label="row-normalized fraction")

    fig.suptitle("Lineage classification (held-out test fold, N=16,318)", y=1.01)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[lineage] wrote {out}")
    return True


# --------------------------------------------------------------------------- #
# Reconstruction mutation heatmaps (best + worst per config)
# --------------------------------------------------------------------------- #
def _load_recons(results_dir: Path, enc: str):
    """Return (per_sequence rows, {source_index: reconstructed_seq})."""
    from downstream_validation.evaluate_pairwise import parse_fasta

    pcsv = results_dir / "decoder" / enc / "per_sequence.csv"
    rfa = results_dir / "decoder" / enc / "reconstructions.fasta"
    if not pcsv.is_file() or not rfa.is_file():
        return None, {}
    rows = _read_csv_rows(pcsv)
    _, recon_seqs = parse_fasta(str(rfa))
    mapping = {int(r["source_index"]): rec for r, rec in zip(rows, recon_seqs)}
    return rows, mapping


def reconstruction_heatmaps(results_dir: Path, fasta: str, configs: list[str], images_dir: Path) -> bool:
    """Render a typical all-encoder heatmap plus per-config best/worst appendix panels."""
    from downstream_validation.evaluate_pairwise import parse_fasta
    from downstream_validation.plot_decode_mutation_heatmaps import plot_reconstruction_heatmap
    from downstream_validation.splits import raw_lineage_from_label, collapse_lineage

    labels, seqs = parse_fasta(fasta)

    def _seq_tag(idx: int) -> str:
        lin = raw_lineage_from_label(labels[idx])
        return f"lineage: {lin}  (WHO: {collapse_lineage(lin)})"
    recon_maps: dict[str, dict[int, str]] = {}
    mism_maps: dict[str, dict[int, int]] = {}
    for enc in ["esm2", *configs]:
        rows, mapping = _load_recons(results_dir, enc)
        recon_maps[enc] = mapping
        mism_maps[enc] = {int(r["source_index"]): int(r["n_mismatches"]) for r in (rows or [])}

    # All encoders in display order (teacher + AWN ascending size) for the main figure.
    all_encs = [("esm2", DISPLAY["esm2"])] + [
        (c, DISPLAY.get(c, c)) for c in CONFIG_ORDER if c in configs
    ]

    # ---- Typical example: median total (across encoders) non-gap mismatches ----
    present_encs = [e for e, _ in all_encs if mism_maps.get(e)]
    if present_encs:
        common = set.intersection(*(set(mism_maps[e]) for e in present_encs))
        totals = {si: sum(mism_maps[e].get(si, 0) for e in present_encs) for si in common}
        nonzero = sorted(si for si in totals if totals[si] > 0)
        if nonzero:
            med_val = float(np.median([totals[si] for si in nonzero]))
            si = min(nonzero, key=lambda s: abs(totals[s] - med_val))
            source = seqs[si][:SEQ_LEN]
            recons = {e: recon_maps[e][si] for e, _ in all_encs
                      if e in recon_maps and si in recon_maps[e]}
            out = images_dir / "mutation_heatmap_typical.png"
            plot_reconstruction_heatmap(
                source, recons, [lbl for _, lbl in all_encs], out,
                title="Reconstruction mismatches vs source \u2014 typical held-out sequence",
                seq_label=_seq_tag(si), encoders=tuple(all_encs),
            )
            print(f"[recon] wrote {out}  (typical: {totals[si]} total mismatches)")

    wrote_any = False
    for cfg in configs:
        rows, _ = _load_recons(results_dir, cfg)
        if not rows or cfg not in recon_maps or not recon_maps[cfg]:
            print(f"[recon] no reconstructions for {cfg}; skipping.")
            continue
        recs = [(int(r["source_index"]), int(r["n_mismatches"])) for r in rows]
        worst = max(recs, key=lambda t: t[1])
        nonzero = [t for t in recs if t[1] > 0]
        best = min(nonzero or recs, key=lambda t: t[1])
        enc_tuple = (("esm2", DISPLAY["esm2"]), (cfg, DISPLAY.get(cfg, cfg)))
        for tag, (si, nm) in [("best", best), ("worst", worst)]:
            source = seqs[si][:SEQ_LEN]
            recons = {
                e: recon_maps[e][si]
                for e, _ in enc_tuple
                if e in recon_maps and si in recon_maps[e]
            }
            out = images_dir / f"mutation_heatmap_{cfg}_{tag}.png"
            plot_reconstruction_heatmap(
                source,
                recons,
                [lbl for _, lbl in enc_tuple],
                out,
                title=f"Reconstruction mismatches vs source \u2014 {DISPLAY.get(cfg, cfg)} "
                      f"({tag}: {nm} non-gap mismatches)",
                seq_label=_seq_tag(si),
                encoders=enc_tuple,
            )
            print(f"[recon] wrote {out}")
            wrote_any = True
    return wrote_any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for paper-figure rendering."""
    p = argparse.ArgumentParser(description="Render paper figures from result artifacts.")
    p.add_argument("--results-dir", default="results/paper")
    p.add_argument("--images-dir", default="results/figures")
    p.add_argument("--fasta", default="data/filtered_msa.fasta")
    p.add_argument("--cache-root", default="embeddings")
    p.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    p.add_argument("--blosum-metric", choices=["l2", "cosine"], default="l2")
    p.add_argument("--skip-blosum", action="store_true")
    p.add_argument("--skip-lineage", action="store_true")
    p.add_argument("--skip-reconstruction", action="store_true")
    return p.parse_args()


def main() -> None:
    """CLI entry point: render all paper figures from result artifacts."""
    args = parse_args()
    results_dir = Path(args.results_dir)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    configs = [e for e in args.encoders if e != "esm2"]

    if not args.skip_blosum:
        blosum_hexbin_figure(
            args.fasta, args.cache_root, configs,
            images_dir / "blosum_correlation.png", metric=args.blosum_metric,
        )
    if not args.skip_lineage:
        lineage_figure(results_dir, args.encoders, images_dir / "lineage_classification.png")
    if not args.skip_reconstruction:
        reconstruction_heatmaps(results_dir, args.fasta, configs, images_dir)


if __name__ == "__main__":
    main()
