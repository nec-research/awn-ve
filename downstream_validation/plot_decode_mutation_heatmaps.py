#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/plot_decode_mutation_heatmaps.py
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

"""Mutation heatmaps after decode-from-latent (ESM-2 vs AWN configs).

Embeds held-out MSA sequences with uncompressed ESM-2 and exported AWN
models, decodes each latent with the ESM-2–trained sequence decoder, and
plots mismatch sites vs the source alignment (cont_dev / virus-evolution
layout: Source + one column per encoder).

Example::

    uv run python -m downstream_validation.plot_decode_mutation_heatmaps \\
        --fasta data/filtered_msa.fasta \\
        --decoder-checkpoint checkpoints/decoder_esm2/best.pt \\
        --use-checkpoint-val \\
        --limit 256 \\
        --n-heatmaps 8 \\
        --output-dir results/decode_mutation_heatmaps
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from downstream_validation.decoder import detokenize
from downstream_validation.evaluate_decoder import load_sequence_decoder, resolve_eval_indices
from downstream_validation.evaluate_pairwise import DEFAULT_SEED, embed_awn, embed_raw_esm, parse_fasta
from downstream_validation.loader import load_model

ENCODERS = (
    ("esm2", "ESM-2"),
    ("config_1", "AWN-1"),
    ("config_2", "AWN-2"),
    ("config_3", "AWN-3"),
)
_GAP = frozenset("-.")


def _identity(source: str, recon: str, *, ignore_gap: bool = False) -> float:
    n = min(len(source), len(recon))
    if n == 0:
        return float("nan")
    matches = 0
    counted = 0
    for i in range(n):
        s, r = source[i], recon[i]
        if ignore_gap and (s in _GAP or r in _GAP):
            continue
        counted += 1
        if s == r:
            matches += 1
    return matches / counted if counted else float("nan")


def _mismatch_positions(
    source: str,
    recons: dict[str, str],
    *,
    include_gap_mismatches: bool = False,
) -> list[int]:
    positions: set[int] = set()
    n = len(source)
    for recon in recons.values():
        for i in range(min(n, len(recon))):
            if source[i] in _GAP:
                continue
            if not include_gap_mismatches and recon[i] in _GAP:
                continue
            if source[i] != recon[i]:
                positions.add(i)
    return sorted(positions)


def _status_vs_source(source: str, recon: str, positions: list[int]) -> np.ndarray:
    """0 = matches source, 1 = mutated."""
    out = np.zeros(len(positions), dtype=int)
    for row, i in enumerate(positions):
        aa = recon[i] if i < len(recon) else "-"
        if aa != source[i]:
            out[row] = 1
    return out


def plot_reconstruction_heatmap(
    source: str,
    recons: dict[str, str],
    column_labels: list[str],
    output_path: Path,
    *,
    title: str,
    max_positions: int = 80,
    seq_label: str = "",
    encoders: tuple[tuple[str, str], ...] = ENCODERS,
    include_gap_mismatches: bool = False,
) -> int:
    """Plot Source + reconstruction columns for mismatch positions only."""
    positions = _mismatch_positions(
        source,
        recons,
        include_gap_mismatches=include_gap_mismatches,
    )
    if not positions:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No mismatches vs source", ha="center", va="center")
        ax.axis("off")
        ax.set_title(title)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return 0

    if len(positions) > max_positions:
        positions = positions[:max_positions]

    n_rows = len(positions)
    col_keys = ["source"] + [k for k, _ in encoders if k in recons]
    col_names = ["Source"] + [
        label for key, label in encoders if key in recons
    ]
    n_cols = len(col_names)

    aa_matrix = np.empty((n_rows, n_cols), dtype=object)
    status = np.zeros((n_rows, n_cols), dtype=int)
    for row, i in enumerate(positions):
        aa_matrix[row, 0] = source[i]
        status[row, 0] = 0
        for col, key in enumerate(col_keys[1:], start=1):
            aa = recons[key][i] if i < len(recons[key]) else "-"
            aa_matrix[row, col] = aa
            status[row, col] = 0 if aa == source[i] else 1

    cmap = ListedColormap(["#e8e8e8", "#f4a460"])
    fig_w = max(7, 0.55 * n_cols + 2)
    fig_h = max(4, 0.28 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(status, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    y_labels = [
        f"{i} ({source[i]})" for i in positions
    ]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_names, rotation=30, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Encoder → decoder")
    ax.set_ylabel("Position (source aa)")
    ax.set_title(title if not seq_label else f"{title}\n{seq_label}")

    for row in range(n_rows):
        for col in range(n_cols):
            aa = aa_matrix[row, col]
            color = "white" if status[row, col] == 1 else "black"
            ax.text(col, row, aa, ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    legend = [
        Patch(facecolor="#e8e8e8", edgecolor="k", label="Matches source"),
        Patch(facecolor="#f4a460", edgecolor="k", label="Mutated vs source"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return n_rows


@torch.inference_mode()
def decode_latents(
    decoder,
    latents: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> list[str]:
    """Greedily decode latent vectors into amino-acid strings, batched."""
    out: list[str] = []
    for start in range(0, latents.shape[0], batch_size):
        z = latents[start : start + batch_size].to(device)
        tokens = decoder.generate(z, greedy=True)
        out.extend(detokenize(tokens))
    return out


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mutation-heatmap plotting."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", default="data/filtered_msa.fasta")
    p.add_argument("--decoder-checkpoint", default="checkpoints/decoder_esm2/best.pt")
    p.add_argument("--use-checkpoint-val", action="store_true", default=True)
    p.add_argument("--no-use-checkpoint-val", action="store_false", dest="use_checkpoint_val")
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--token-budget", type=int, default=16384)
    p.add_argument("--esm-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--decode-batch-size", type=int, default=8)
    p.add_argument("--n-heatmaps", type=int, default=8, help="Individual sequence heatmaps to save.")
    p.add_argument("--max-positions", type=int, default=80)
    p.add_argument("--output-dir", type=Path, default=Path("results/decode_mutation_heatmaps"))
    return p.parse_args()


def main() -> None:
    """CLI entry point: decode latents and render mutation heatmaps."""
    args = parse_args()
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    labels, seqs = parse_fasta(args.fasta)
    decoder, dec_meta = load_sequence_decoder(args.decoder_checkpoint, device=device)

    cfg_path = Path(args.decoder_checkpoint).parent / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else dec_meta
    eval_idxs = resolve_eval_indices(
        len(seqs),
        cfg,
        use_checkpoint_val=args.use_checkpoint_val,
        limit=args.limit,
        seed=args.seed,
    )
    if eval_idxs is not None:
        labels = [labels[i] for i in eval_idxs]
        seqs = [seqs[i] for i in eval_idxs]
        print(f"Evaluating checkpoint val split N={len(seqs)}")
    elif args.limit and args.limit < len(seqs):
        rng = random.Random(args.seed)
        pick = sorted(rng.sample(range(len(seqs)), args.limit))
        labels = [labels[i] for i in pick]
        seqs = [seqs[i] for i in pick]
        print(f"Subsampled to N={len(seqs)} (seed={args.seed})")

    seq_length = int(decoder.seq_len)
    seqs = [s[:seq_length] for s in seqs]

    print("Embedding with ESM-2...")
    esm_latents, esm_name, _ = embed_raw_esm(
        seqs,
        device=device,
        model_name="esm2_t33_650M_UR50D",
        repr_layer=33,
        token_budget=args.token_budget,
        seq_length=seq_length,
        esm_dtype=args.esm_dtype,
    )

    awn_latents: dict[str, torch.Tensor] = {}
    for config_id in ("config_1", "config_2", "config_3"):
        print(f"Embedding with AWN {config_id}...")
        awn_model, metadata, target_mean = load_model(config_id, device=device)
        enc_len = min(seq_length, int(metadata["max_sequence_length"]))
        awn_latents[config_id] = embed_awn(
            awn_model,
            target_mean,
            seqs,
            device=device,
            token_budget=args.token_budget,
            seq_length=enc_len,
        )
        del awn_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Decoding...")
    decoded: dict[str, list[str]] = {
        "esm2": decode_latents(decoder, esm_latents, device, args.decode_batch_size),
    }
    for config_id, latents in awn_latents.items():
        decoded[config_id] = decode_latents(decoder, latents, device, args.decode_batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "decode_identity_summary.csv"
    summary_rows: list[dict] = []

    for idx, (label, source) in enumerate(zip(labels, seqs)):
        row: dict = {"index": idx, "label": label}
        for key, _ in ENCODERS:
            recon = decoded[key][idx]
            id_all = _identity(source, recon, ignore_gap=False)
            id_ng = _identity(source, recon, ignore_gap=True)
            row[f"{key}_identity"] = f"{id_all:.6f}"
            row[f"{key}_nongap_identity"] = f"{id_ng:.6f}"
            row[f"{key}_n_mismatches"] = sum(
                1 for i in range(min(len(source), len(recon)))
                if source[i] not in _GAP and recon[i] not in _GAP and source[i] != recon[i]
            )
        summary_rows.append(row)

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        fields = ["index", "label"]
        for key, _ in ENCODERS:
            fields.extend([f"{key}_identity", f"{key}_nongap_identity", f"{key}_n_mismatches"])
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")

    # Aggregate stats
    print("\nMean identity (all positions):")
    for key, name in ENCODERS:
        vals = [float(r[f"{key}_identity"]) for r in summary_rows]
        print(f"  {name:8s}: {np.mean(vals):+.4f}  (min={np.min(vals):.4f}, max={np.max(vals):.4f})")

    # Pick heatmap examples: highest/lowest AWN-2 spread vs ESM, plus random
    rng = random.Random(args.seed)
    scored = []
    for idx, row in enumerate(summary_rows):
        esm_id = float(row["esm2_identity"])
        for cfg in ("config_1", "config_2", "config_3"):
            awn_id = float(row[f"{cfg}_identity"])
            scored.append((abs(esm_id - awn_id), idx, cfg))
    scored.sort(reverse=True)
    chosen: set[int] = set()
    heatmap_plan: list[tuple[int, str]] = []
    for _, idx, cfg in scored:
        if len(heatmap_plan) >= args.n_heatmaps:
            break
        if idx not in chosen:
            chosen.add(idx)
            heatmap_plan.append((idx, f"large_gap_vs_esm_{cfg}"))
    for idx in rng.sample(range(len(seqs)), k=min(args.n_heatmaps, len(seqs))):
        if len(heatmap_plan) >= args.n_heatmaps:
            break
        if idx not in chosen:
            chosen.add(idx)
            heatmap_plan.append((idx, "random"))

    per_seq_dir = args.output_dir / "per_sequence"
    for idx, tag in heatmap_plan:
        source = seqs[idx]
        recons = {key: decoded[key][idx] for key, _ in ENCODERS}
        safe_label = labels[idx].replace("/", "-").replace(" ", "_")[:60]
        out_png = per_seq_dir / f"seq{idx:04d}_{tag}_{safe_label}.png"
        n_rows = plot_reconstruction_heatmap(
            source,
            recons,
            [name for _, name in ENCODERS],
            out_png,
            title="Decode mutation sites vs source",
            max_positions=args.max_positions,
            seq_label=f"{labels[idx]}  (idx={idx})",
        )
        print(f"Saved heatmap ({n_rows} rows): {out_png}")

    # Mean identity bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [name for _, name in ENCODERS]
    means = [
        np.mean([float(r[f"{key}_identity"]) for r in summary_rows])
        for key, _ in ENCODERS
    ]
    bars = ax.bar(names, means, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean reconstruction identity")
    ax.set_title(f"Decoder identity by encoder (N={len(seqs)}, ESM-2 decoder)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    summary_plot = args.output_dir / "mean_identity_by_encoder.png"
    fig.savefig(summary_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot: {summary_plot}")


if __name__ == "__main__":
    main()
