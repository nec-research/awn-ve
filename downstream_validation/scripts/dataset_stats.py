#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/scripts/dataset_stats.py
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

"""Compute curated-dataset characteristics and render the appendix artifacts.

Reads the aligned Spike MSA (``data/filtered_msa.fasta``), reconstructs the
leakage-free protein-ID split with :mod:`splits`, and writes the tables and
figures used by the paper's dataset appendix:

Tables (LaTeX fragments, ``--tables-dir``)
    * ``corpus_stats.tex``   - size, length, gap, lineage/country counts, splits.
    * ``geography.tex``      - top-N isolation countries plus an aggregated tail.

Figures (PNG, ``--images-dir``)
    * ``dataset_temporal_by_variant.png``   - monthly collection-date histogram
                                              stacked by WHO variant group.
    * ``dataset_alignment_occupancy.png``   - per-column non-gap fraction across
                                              the 1,300 alignment positions.
    * ``dataset_variant_composition.png``   - WHO variant composition, full
                                              corpus vs. held-out test fold.

Example::

    .venv/bin/python scripts/dataset_stats.py \
        --fasta data/filtered_msa.fasta \
        --images-dir results/figures \
        --tables-dir results/tables
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ``splits.py`` lives in the downstream_validation package root, one level above scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import splits as S

ALIGN_LEN = 1300
GAP_CHARS = frozenset({"-", "."})

# WHO variant display order and a colour-blind-friendly palette.
VARIANT_ORDER = ["Alpha", "Beta", "Gamma", "Delta", "Omicron", "Other"]
VARIANT_COLORS = {
    "Alpha": "#4c72b0",
    "Beta": "#55a868",
    "Gamma": "#c44e52",
    "Delta": "#dd8452",
    "Omicron": "#8172b3",
    "Other": "#999999",
}
TOP_COUNTRIES = 15


# --------------------------------------------------------------------------- #
# Single streaming pass over the FASTA.
# --------------------------------------------------------------------------- #
def scan_fasta(path: Path) -> dict:
    """One pass: collect labels, per-column occupancy, and residue lengths."""
    labels: list[str] = []
    occupancy = np.zeros(ALIGN_LEN, dtype=np.int64)
    res_lengths: list[int] = []
    seq_parts: list[str] = []

    def flush() -> None:
        if not seq_parts:
            return
        seq = "".join(seq_parts)
        residues = 0
        for i, ch in enumerate(seq):
            if ch not in GAP_CHARS:
                residues += 1
                if i < ALIGN_LEN:
                    occupancy[i] += 1
        res_lengths.append(residues)

    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                labels.append(line[1:].strip() or f"seq_{len(labels)}")
                seq_parts = []
            else:
                seq_parts.append(line)
        flush()

    if not labels:
        raise ValueError(f"No FASTA records found in {path}")
    return {
        "labels": labels,
        "occupancy": occupancy / max(len(labels), 1),
        "res_lengths": np.asarray(res_lengths, dtype=np.int64),
    }


def month_key(raw: str) -> str | None:
    """Return a ``YYYY-MM`` bucket from a header date, or ``None``."""
    if not raw or len(raw) < 7:
        return None
    return raw[:7]


# --------------------------------------------------------------------------- #
# LaTeX tables.
# --------------------------------------------------------------------------- #
def write_corpus_table(stats: dict, out: Path) -> None:
    res = stats["res_lengths"]
    lines = [
        "% Auto-generated by scripts/dataset_stats.py -- do not edit by hand.",
        "\\begin{table}[htbp]",
        "  \\caption{Curated full-length SARS-CoV-2 Spike corpus. Sequences are",
        "  aligned to a fixed reference and split by protein identifier"
        " (Sec.~\\ref{par:splits}).}",
        "  \\label{tab:corpus-stats}",
        "  \\centering",
        "  \\begin{tabular}{ll}",
        "  \\hline",
        "  Characteristic & Value\\\\",
        "  \\hline",
        f"  Records / unique identifiers & {stats['n']:,} / {stats['n_unique']:,}\\\\",
        f"  Alignment length & {ALIGN_LEN:,} columns\\\\",
        f"  Residues per sequence (non-gap) & {int(res.mean())} mean"
        f" ({int(res.min())}--{int(res.max())})\\\\",
        f"  Mean gap fraction & {stats['mean_gap_frac'] * 100:.1f}\\%\\\\",
        f"  Distinct Pango lineages & {stats['n_lineages']:,}\\\\",
        f"  Isolation countries & {stats['n_countries']:,}\\\\",
        f"  Collection span & {stats['date_min']} to {stats['date_max']}\\\\",
        f"  Train / validation / test & {stats['splits']['train']:,} /"
        f" {stats['splits']['val']:,} / {stats['splits']['test']:,}\\\\",
        "  \\hline",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table] wrote {out}")


def _pretty_country(name: str) -> str:
    if name in {"", "nan", "NaN", "NA", "unknown", "Unknown"}:
        return "Unknown"
    return name.replace("_", " ")


def write_geography_table(country_counts: Counter, total: int, out: Path) -> None:
    top = country_counts.most_common(TOP_COUNTRIES)
    covered = sum(c for _, c in top)
    tail = total - covered
    caption = (
        f"Top {TOP_COUNTRIES} isolation countries by record count "
        f"(of {len(country_counts)} countries represented; "
        "\\emph{Unknown} denotes records without a reported country)."
    )
    lines = [
        "% Auto-generated by scripts/dataset_stats.py -- do not edit by hand.",
        "\\begin{table}[htbp]",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:geography}",
        "  \\centering",
        "  \\begin{tabular}{lrr}",
        "  \\hline",
        "  Country & Records & Share\\\\",
        "  \\hline",
    ]
    for name, count in top:
        pretty = _pretty_country(name)
        lines.append(f"  {pretty} & {count:,} & {count / total * 100:.1f}\\%\\\\")
    lines += [
        "  \\hline",
        f"  Other countries & {tail:,} & {tail / total * 100:.1f}\\%\\\\",
        "  \\hline",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table] wrote {out}")


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def fig_temporal(month_variant: dict, out: Path) -> None:
    months = sorted(month_variant)
    x = np.arange(len(months))
    bottom = np.zeros(len(months))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for variant in VARIANT_ORDER:
        heights = np.array([month_variant[m].get(variant, 0) for m in months])
        if heights.sum() == 0:
            continue
        ax.bar(x, heights, bottom=bottom, width=1.0,
               label=variant, color=VARIANT_COLORS[variant])
        bottom += heights
    # Year ticks at each January.
    tick_pos = [i for i, m in enumerate(months) if m.endswith("-01")]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([months[i][:4] for i in tick_pos])
    ax.set_xlabel("collection date (month)")
    ax.set_ylabel("sequences")
    ax.set_title("Temporal coverage by WHO variant group")
    ax.margins(x=0.01)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out}")


def fig_occupancy(occupancy: np.ndarray, out: Path) -> None:
    x = np.arange(1, len(occupancy) + 1)
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.fill_between(x, occupancy, color="#4c72b0", alpha=0.35)
    ax.plot(x, occupancy, color="#1f3b73", linewidth=0.8)
    ax.set_xlabel("alignment column")
    ax.set_ylabel("non-gap fraction")
    ax.set_title("Per-column occupancy of the Spike alignment")
    ax.set_xlim(1, len(occupancy))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out}")


def fig_variant_composition(full: Counter, test: Counter, out: Path) -> None:
    variants = [v for v in VARIANT_ORDER if full.get(v, 0) > 0]
    full_share = np.array([full.get(v, 0) / sum(full.values()) for v in variants])
    test_share = np.array([test.get(v, 0) / sum(test.values()) for v in variants])
    x = np.arange(len(variants))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width / 2, full_share * 100, width,
           label="full corpus", color="#4c72b0")
    ax.bar(x + width / 2, test_share * 100, width,
           label="held-out test", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("share of records (\\%)" if False else "share of records (%)")
    ax.set_title("WHO variant composition: full corpus vs. test fold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out}")


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", default="data/filtered_msa.fasta")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--images-dir", default="results/figures")
    p.add_argument("--tables-dir", default="results/tables")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fasta = Path(args.fasta)
    images_dir = Path(args.images_dir)
    tables_dir = Path(args.tables_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    scan = scan_fasta(fasta)
    labels = scan["labels"]
    fields = [S.parse_header_fields(label) for label in labels]

    sp = S.aligned_splits(labels, seed=args.seed)
    splits = {k.replace("_idxs", ""): len(v) for k, v in sp.items()}

    variants = [S.collapse_lineage(f.get("lineage", "")) for f in fields]
    full_variant = Counter(variants)
    test_variant = Counter(variants[i] for i in sp["test_idxs"])

    countries = Counter(f.get("isolation_country", "nan") for f in fields)

    dates = sorted(f.get("collection_date", "") for f in fields
                   if len(f.get("collection_date", "")) == 10)

    month_variant: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f, variant in zip(fields, variants):
        mk = month_key(f.get("collection_date", ""))
        if mk is not None:
            month_variant[mk][variant] += 1

    stats = {
        "n": len(labels),
        # Protein ID = the token before the first whitespace (matches
        # esm_surrogate.data.protein_id_from_label without importing MLWiz).
        "n_unique": len({label.split(maxsplit=1)[0] for label in labels}),
        "res_lengths": scan["res_lengths"],
        "mean_gap_frac": float(1.0 - scan["occupancy"].mean()),
        "n_lineages": len({f.get("lineage", "Unassigned") for f in fields}),
        "n_countries": len(countries),
        "date_min": dates[0] if dates else "n/a",
        "date_max": dates[-1] if dates else "n/a",
        "splits": splits,
    }

    # Tables.
    write_corpus_table(stats, tables_dir / "corpus_stats.tex")
    write_geography_table(countries, len(labels), tables_dir / "geography.tex")

    # Figures.
    fig_temporal(month_variant, images_dir / "dataset_temporal_by_variant.png")
    fig_occupancy(scan["occupancy"], images_dir / "dataset_alignment_occupancy.png")
    fig_variant_composition(full_variant, test_variant,
                            images_dir / "dataset_variant_composition.png")

    # Console summary.
    print("\n=== dataset summary ===")
    print(f"records: {stats['n']:,} (unique IDs {stats['n_unique']:,})")
    print(f"splits:  {splits}")
    print(f"variants (full): {full_variant.most_common()}")
    print(f"variants (test): {test_variant.most_common()}")


if __name__ == "__main__":
    main()
