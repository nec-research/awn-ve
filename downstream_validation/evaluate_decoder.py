#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/evaluate_decoder.py
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

"""Evaluate an ESM-2–conditioned sequence decoder checkpoint.

Reports teacher-forced accuracy, autoregressive reconstruction identity,
embedding cycle consistency, and an optional latent-lerp smoke test.

Example::

    uv run python -m downstream_validation.evaluate_decoder \\
        --fasta data/filtered_msa.fasta \\
        --decoder-checkpoint checkpoints/decoder_esm2/best.pt \\
        --limit 256 \\
        --token-budget 16384 \\
        --esm-dtype bf16 \\
        --output results/decoder_esm2/eval_decoder.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from downstream_validation.decoder import (
    DECODER_RESIDUES,
    SequenceDecoder,
    _CHAR_TO_ID,
    detokenize,
    tokenize_sequences,
)
from downstream_validation.evaluate_pairwise import DEFAULT_SEED, embed_awn, embed_raw_esm, parse_fasta
from downstream_validation.loader import load_model
from downstream_validation.train_decoder import (
    DEFAULT_ESM_MODEL,
    DecoderSplitDataset,
    run_eval,
)


def load_sequence_decoder(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[SequenceDecoder, dict]:
    """Load a trained ``SequenceDecoder`` and its checkpoint metadata from disk."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = state.get("decoder_state_dict", state)
    latent_dim = state.get("latent_dim")
    seq_len = state.get("seq_len")
    if latent_dim is None or seq_len is None:
        raise KeyError(
            "Decoder checkpoint missing latent_dim/seq_len. "
            f"Keys: {sorted(state.keys())}"
        )
    model = SequenceDecoder(
        latent_dim=int(latent_dim),
        seq_len=int(seq_len),
        d_model=int(state.get("d_model", 256)),
        n_layers=int(state.get("n_layers", 4)),
        n_heads=int(state.get("n_heads", 4)),
        ff_dim=int(state.get("ff_dim", 1024)),
        dropout=float(state.get("dropout", 0.1)),
        vocab_size=int(state.get("vocab_size", 22)),
        bos_id=int(state.get("bos_id", 21)),
    )
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model, state


def _identity(pred: torch.Tensor, true: torch.Tensor, *, ignore_gap: bool) -> float:
    if ignore_gap:
        gap_id = _CHAR_TO_ID["-"]
        scores: list[float] = []
        for i in range(true.shape[0]):
            mask = true[i] != gap_id
            if not mask.any():
                continue
            scores.append(float((pred[i, mask] == true[i, mask]).float().mean().item()))
        return float(np.mean(scores)) if scores else float("nan")
    return float((pred == true).float().mean(dim=1).mean().item())


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=1)
    b = F.normalize(b.float(), dim=1)
    return (1.0 - (a * b).sum(dim=1)).clamp(min=0.0, max=2.0)


@torch.inference_mode()
def autoregressive_metrics(
    model: SequenceDecoder,
    latents: torch.Tensor,
    tokens: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
) -> dict:
    """Greedily decode each latent and summarize autoregressive identity metrics."""
    model.eval()
    preds: list[torch.Tensor] = []
    for start in range(0, latents.shape[0], batch_size):
        z = latents[start : start + batch_size].to(device)
        preds.append(model.generate(z, greedy=True).cpu())
    pred = torch.cat(preds, dim=0)
    return {
        "ar_identity": _identity(pred, tokens, ignore_gap=False),
        "ar_nongap_identity": _identity(pred, tokens, ignore_gap=True),
        "pred_tokens": pred,
    }


def per_sequence_reconstruction_rows(
    pred: torch.Tensor,
    true: torch.Tensor,
    labels: list[str],
    source_indices: list[int],
) -> list[dict]:
    """Return exact per-sequence reconstruction metrics.

    ``nongap_identity`` and ``n_mismatches`` match the legacy decode-heatmap
    convention: positions are counted only when both source and reconstruction
    are non-gap. ``source_nongap_*`` uses every source amino-acid position and
    therefore counts a predicted gap as an error.
    """
    if pred.shape != true.shape:
        raise ValueError(f"Prediction shape {pred.shape} != target shape {true.shape}.")
    if pred.shape[0] != len(labels) or len(labels) != len(source_indices):
        raise ValueError("Predictions, labels, and source indices must have equal length.")

    gap_id = _CHAR_TO_ID["-"]
    rows: list[dict] = []
    for eval_index, (pred_i, true_i, label, source_index) in enumerate(
        zip(pred, true, labels, source_indices)
    ):
        mismatch = pred_i != true_i
        source_nongap = true_i != gap_id
        pair_nongap = source_nongap & (pred_i != gap_id)

        n_positions = int(true_i.numel())
        n_source_nongap = int(source_nongap.sum().item())
        n_pair_nongap = int(pair_nongap.sum().item())
        n_mismatches = int((mismatch & pair_nongap).sum().item())
        n_source_nongap_errors = int((mismatch & source_nongap).sum().item())

        rows.append({
            "eval_index": eval_index,
            "source_index": source_index,
            "label": label,
            "n_positions": n_positions,
            "identity": 1.0 - float(mismatch.sum().item()) / n_positions,
            "n_source_nongap": n_source_nongap,
            "source_nongap_identity": (
                1.0 - n_source_nongap_errors / n_source_nongap
                if n_source_nongap
                else float("nan")
            ),
            "n_source_nongap_errors": n_source_nongap_errors,
            "n_pair_nongap": n_pair_nongap,
            "nongap_identity": (
                1.0 - n_mismatches / n_pair_nongap
                if n_pair_nongap
                else float("nan")
            ),
            "n_mismatches": n_mismatches,
            "n_predicted_gaps_at_source_nongap": int(
                (source_nongap & (pred_i == gap_id)).sum().item()
            ),
        })
    return rows


def write_per_sequence_csv(path: Path, rows: list[dict]) -> None:
    """Write per-sequence reconstruction metric rows to ``path`` as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def cycle_metrics(
    latents: torch.Tensor,
    decoded_seqs: list[str],
    *,
    device: torch.device,
    esm_model: str,
    esm_repr_layer: int | None,
    token_budget: int,
    seq_length: int,
    esm_dtype: str,
) -> dict:
    """Re-embed decoded sequences and measure encode-decode cycle consistency."""
    re_emb, _, _ = embed_raw_esm(
        decoded_seqs,
        device=device,
        model_name=esm_model,
        repr_layer=esm_repr_layer,
        token_budget=token_budget,
        seq_length=seq_length,
        esm_dtype=esm_dtype,
    )
    d = _cosine_distance(latents.cpu(), re_emb.cpu())
    return {
        "cycle_cos_dist_mean": float(d.mean().item()),
        "cycle_cos_dist_median": float(d.median().item()),
        "cycle_cos_dist_p90": float(torch.quantile(d, 0.9).item()),
        "cycle_distances": d.numpy(),
    }


@torch.inference_mode()
def lerp_smoke_test(
    model: SequenceDecoder,
    latents: torch.Tensor,
    *,
    device: torch.device,
    esm_model: str,
    esm_repr_layer: int | None,
    token_budget: int,
    seq_length: int,
    esm_dtype: str,
    n_pairs: int = 8,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Decode interpolations between latent pairs to probe latent-space smoothness."""
    n = latents.shape[0]
    if n < 2:
        return []
    rng = random.Random(seed)
    pairs = [tuple(rng.sample(range(n), 2)) for _ in range(min(n_pairs, n // 2))]
    rows: list[dict] = []
    for pair_i, (ia, ib) in enumerate(pairs):
        za = latents[ia]
        zb = latents[ib]
        for alpha in alphas:
            z = ((1.0 - alpha) * za + alpha * zb).unsqueeze(0)
            tokens = model.generate(z.to(device), greedy=True)
            seq = detokenize(tokens)[0]
            cycle = cycle_metrics(
                z.cpu(),
                [seq],
                device=device,
                esm_model=esm_model,
                esm_repr_layer=esm_repr_layer,
                token_budget=token_budget,
                seq_length=seq_length,
                esm_dtype=esm_dtype,
            )
            rows.append({
                "pair": pair_i,
                "i": ia,
                "j": ib,
                "alpha": alpha,
                "cycle_cos_dist": cycle["cycle_cos_dist_mean"],
                "seq": seq,
            })
            print(
                f"  lerp pair={pair_i} alpha={alpha:.2f} "
                f"cycle_cos={cycle['cycle_cos_dist_mean']:.4f}"
            )
    return rows


def resolve_eval_indices(
    n_fasta: int,
    meta: dict,
    *,
    use_checkpoint_val: bool,
    limit: int | None,
    seed: int,
    fold: str = "val",
) -> list[int] | None:
    """Return indices into the full FASTA, or None to use all (then optional limit)."""
    if use_checkpoint_val:
        raw_idxs = meta.get(f"{fold}_idxs")
        if not raw_idxs:
            # Fall back to config.json next to the checkpoint if present later.
            return None
        idxs = [int(i) for i in raw_idxs]
        if any(i < 0 or i >= n_fasta for i in idxs):
            raise SystemExit(
                f"Checkpoint {fold}_idxs out of range for this FASTA "
                f"(N={n_fasta}). Pass matching --fasta or omit the --use-checkpoint flag."
            )
        if limit is not None and limit > 0 and limit < len(idxs):
            rng = random.Random(seed)
            idxs = sorted(rng.sample(idxs, limit))
        return idxs
    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for decoder evaluation."""
    p = argparse.ArgumentParser(
        description="Evaluate ESM-2–conditioned AR sequence decoder.",
    )
    p.add_argument("--fasta", default="data/filtered_msa.fasta")
    p.add_argument("--decoder-checkpoint", required=True)
    p.add_argument(
        "--use-checkpoint-val",
        action="store_true",
        help="Evaluate on val_idxs stored in the checkpoint (training holdout).",
    )
    p.add_argument(
        "--use-checkpoint-test",
        action="store_true",
        help=(
            "Evaluate on the aligned, leakage-free test_idxs stored in the "
            "checkpoint (the AWN surrogate's held-out protein IDs)."
        ),
    )
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--token-budget", type=int, default=16384)
    p.add_argument("--seq-length", type=int, default=None)
    p.add_argument(
        "--encoder",
        choices=["esm2", "awn"],
        default=None,
        help="Override embedding source (default: read from checkpoint/config).",
    )
    p.add_argument(
        "--awn-model",
        default=None,
        help="AWN bundle id when --encoder awn (default: checkpoint value).",
    )
    p.add_argument(
        "--esm-model",
        default=None,
        help="Override encoder (default: checkpoint esm_model or esm2_t33_650M_UR50D).",
    )
    p.add_argument(
        "--esm-repr-layer",
        type=int,
        default=None,
        help="Override layer (default: checkpoint value or last layer).",
    )
    p.add_argument("--esm-dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--decode-batch-size", type=int, default=4)
    p.add_argument("--tf-batch-size", type=int, default=8)
    p.add_argument("--output", default=None)
    p.add_argument(
        "--per-sequence-output",
        default=None,
        help=(
            "Optional CSV path for exact per-sequence identities and mismatch "
            "counts (requires autoregressive decoding)."
        ),
    )
    p.add_argument(
        "--save-recon-fasta",
        default=None,
        help="Optional path to write greedy reconstructions.",
    )
    p.add_argument("--lerp-pairs", type=int, default=8)
    p.add_argument("--skip-ar", action="store_true")
    p.add_argument("--skip-cycle", action="store_true")
    return p.parse_args()


def main() -> None:
    """CLI entry point: evaluate a trained sequence decoder on a FASTA split."""
    args = parse_args()
    if not os.path.isfile(args.fasta):
        raise SystemExit(f"FASTA not found: {args.fasta}")
    if not os.path.isfile(args.decoder_checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.decoder_checkpoint}")

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model, meta = load_sequence_decoder(args.decoder_checkpoint, device=device)
    cfg_path = Path(args.decoder_checkpoint).parent / "config.json"
    cfg = {}
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    encoder = args.encoder or meta.get("encoder") or cfg.get("encoder") or "esm2"
    awn_model = args.awn_model or meta.get("awn_model") or cfg.get("awn_model") or "config_2"
    esm_model = args.esm_model or meta.get("esm_model") or cfg.get("esm_model") or DEFAULT_ESM_MODEL
    esm_layer = (
        args.esm_repr_layer
        if args.esm_repr_layer is not None
        else meta.get("esm_repr_layer")
    )
    if esm_layer is not None:
        esm_layer = int(esm_layer)
        if esm_layer < 0:
            esm_layer = None

    print(
        f"Loaded decoder: latent_dim={model.latent_dim} seq_len={model.seq_len} "
        f"input_mode={meta.get('input_mode', 'raw')}"
    )
    if encoder == "awn":
        print(f"Encoder: AWN {awn_model}")
    else:
        print(f"Encoder: {esm_model} layer={esm_layer if esm_layer is not None else 'last'}")

    labels, seqs = parse_fasta(args.fasta)
    source_indices = list(range(len(seqs)))
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise SystemExit(f"Need equal-length MSA sequences, got {sorted(lengths)}")
    fasta_len = lengths.pop()

    if args.use_checkpoint_val and args.use_checkpoint_test:
        raise SystemExit("Pass only one of --use-checkpoint-val / --use-checkpoint-test.")
    use_checkpoint_fold = args.use_checkpoint_val or args.use_checkpoint_test
    fold = "test" if args.use_checkpoint_test else "val"

    eval_idxs = resolve_eval_indices(
        len(seqs),
        meta,
        use_checkpoint_val=use_checkpoint_fold,
        limit=args.limit,
        seed=args.seed,
        fold=fold,
    )
    if eval_idxs is None and use_checkpoint_fold:
        # Try sibling config.json written by train_decoder.
        cfg_path = Path(args.decoder_checkpoint).parent / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            eval_idxs = resolve_eval_indices(
                len(seqs),
                cfg,
                use_checkpoint_val=True,
                limit=args.limit,
                seed=args.seed,
                fold=fold,
            )
        if eval_idxs is None:
            raise SystemExit(
                f"--use-checkpoint-{fold} set but no {fold}_idxs in "
                "checkpoint/config.json."
            )

    if eval_idxs is not None:
        labels = [labels[i] for i in eval_idxs]
        seqs = [seqs[i] for i in eval_idxs]
        source_indices = eval_idxs
        print(f"Evaluating checkpoint {fold} split N={len(seqs)}")
    elif args.limit is not None and args.limit > 0 and args.limit < len(seqs):
        rng = random.Random(args.seed)
        pick = sorted(rng.sample(range(len(seqs)), args.limit))
        labels = [labels[i] for i in pick]
        seqs = [seqs[i] for i in pick]
        source_indices = pick
        print(f"Subsampled to N={len(seqs)} (seed={args.seed})")
    else:
        print(f"Evaluating N={len(seqs)}")

    if any(len(s) != model.seq_len for s in seqs):
        seqs = [s[: model.seq_len] for s in seqs]
        if any(len(s) != model.seq_len for s in seqs):
            raise SystemExit(
                f"Sequences shorter than model seq_len={model.seq_len} after truncate."
            )

    enc_len = args.seq_length or max(fasta_len, model.seq_len)
    if encoder == "awn":
        print(f"Embedding N={len(seqs)} via AWN {awn_model}...")
        awn, metadata, target_mean = load_model(awn_model, device=device)
        if "max_sequence_length" in metadata:
            enc_len = min(enc_len, int(metadata["max_sequence_length"]))
        latents = embed_awn(
            awn,
            target_mean,
            seqs,
            device=device,
            token_budget=args.token_budget,
            seq_length=enc_len,
        )
        used_layer = None
        del awn
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"Embedding N={len(seqs)} via {esm_model}...")
        latents, _, used_layer = embed_raw_esm(
            seqs,
            device=device,
            model_name=esm_model,
            repr_layer=esm_layer,
            token_budget=args.token_budget,
            seq_length=enc_len,
            esm_dtype=args.esm_dtype,
        )
    if latents.shape[1] != model.latent_dim:
        raise SystemExit(
            f"Embed dim {latents.shape[1]} != decoder latent_dim {model.latent_dim}."
        )

    tokens = tokenize_sequences(seqs)
    if tokens.shape[1] != model.seq_len:
        raise SystemExit(
            f"Token length {tokens.shape[1]} != model seq_len {model.seq_len}."
        )

    ds = DecoderSplitDataset(latents, tokens)
    loader = DataLoader(
        ds,
        batch_size=min(args.tf_batch_size, len(ds)),
        shuffle=False,
        collate_fn=DecoderSplitDataset.collate,
    )
    tf_loss, tf_acc, tf_nongap = run_eval(model, loader, device)
    print(
        f"Teacher-forced: loss={tf_loss:.4f} acc={tf_acc:.4f} "
        f"nongap_acc={tf_nongap:.4f}"
    )

    results: dict = {
        "n_seqs": len(seqs),
        "input_mode": meta.get("input_mode", "raw"),
        "encoder": encoder,
        "tf_loss": tf_loss,
        "tf_acc": tf_acc,
        "tf_nongap_acc": tf_nongap,
        "alphabet": DECODER_RESIDUES,
        "decoder_checkpoint": args.decoder_checkpoint,
    }
    if encoder == "awn":
        results["awn_model"] = awn_model
    else:
        results["esm_model"] = esm_model
        results["esm_repr_layer"] = used_layer

    pred_tokens = None
    if not args.skip_ar:
        print("Autoregressive greedy decode...")
        ar = autoregressive_metrics(
            model, latents, tokens, device, batch_size=args.decode_batch_size,
        )
        pred_tokens = ar["pred_tokens"]
        results["ar_identity"] = ar["ar_identity"]
        results["ar_nongap_identity"] = ar["ar_nongap_identity"]
        print(
            f"AR identity={ar['ar_identity']:.4f} "
            f"nongap={ar['ar_nongap_identity']:.4f}"
        )
        if args.per_sequence_output:
            per_sequence_rows = per_sequence_reconstruction_rows(
                pred_tokens,
                tokens,
                labels,
                source_indices,
            )
            per_sequence_path = Path(args.per_sequence_output)
            write_per_sequence_csv(per_sequence_path, per_sequence_rows)

            substitution_counts = np.asarray(
                [row["n_mismatches"] for row in per_sequence_rows],
                dtype=np.float64,
            )
            source_nongap_errors = np.asarray(
                [row["n_source_nongap_errors"] for row in per_sequence_rows],
                dtype=np.float64,
            )
            for prefix, values in (
                ("ar_substitutions_per_seq", substitution_counts),
                ("ar_source_nongap_errors_per_seq", source_nongap_errors),
            ):
                results[f"{prefix}_mean"] = float(np.mean(values))
                results[f"{prefix}_median"] = float(np.median(values))
                results[f"{prefix}_min"] = int(np.min(values))
                results[f"{prefix}_p05"] = float(np.quantile(values, 0.05))
                results[f"{prefix}_p25"] = float(np.quantile(values, 0.25))
                results[f"{prefix}_p75"] = float(np.quantile(values, 0.75))
                results[f"{prefix}_p95"] = float(np.quantile(values, 0.95))
                results[f"{prefix}_max"] = int(np.max(values))
            print(
                "Per-sequence substitutions: "
                f"mean={np.mean(substitution_counts):.2f} "
                f"median={np.median(substitution_counts):.1f} "
                f"range={int(np.min(substitution_counts))}-"
                f"{int(np.max(substitution_counts))}"
            )
            print(f"Wrote per-sequence metrics to {per_sequence_path}")
        if args.save_recon_fasta:
            out = Path(args.save_recon_fasta)
            out.parent.mkdir(parents=True, exist_ok=True)
            recon = detokenize(pred_tokens)
            with open(out, "w", encoding="utf-8") as fh:
                for lab, seq in zip(labels, recon):
                    fh.write(f">{lab}\n{seq}\n")
            print(f"Wrote reconstructions to {out}")
    elif args.per_sequence_output:
        raise SystemExit("--per-sequence-output requires AR decoding; omit --skip-ar.")

    if not args.skip_cycle and pred_tokens is not None and encoder == "esm2":
        print("Cycle consistency (re-embed decoded sequences)...")
        decoded = detokenize(pred_tokens)
        cycle = cycle_metrics(
            latents,
            decoded,
            device=device,
            esm_model=esm_model,
            esm_repr_layer=used_layer,
            token_budget=args.token_budget,
            seq_length=enc_len,
            esm_dtype=args.esm_dtype,
        )
        results["cycle_cos_dist_mean"] = cycle["cycle_cos_dist_mean"]
        results["cycle_cos_dist_median"] = cycle["cycle_cos_dist_median"]
        results["cycle_cos_dist_p90"] = cycle["cycle_cos_dist_p90"]
        print(
            f"Cycle cos dist: mean={cycle['cycle_cos_dist_mean']:.4f} "
            f"median={cycle['cycle_cos_dist_median']:.4f} "
            f"p90={cycle['cycle_cos_dist_p90']:.4f}"
        )
    elif not args.skip_cycle and pred_tokens is not None and encoder == "awn":
        print("Skipping cycle check for AWN encoder (use --skip-cycle to silence).")
    elif not args.skip_cycle and pred_tokens is None:
        print("Skipping cycle check (requires AR decode; omit --skip-ar).")

    lerp_rows: list[dict] = []
    if args.lerp_pairs > 0 and encoder == "esm2":
        print(f"Latent lerp smoke test ({args.lerp_pairs} pairs)...")
        lerp_rows = lerp_smoke_test(
            model,
            latents,
            device=device,
            esm_model=esm_model,
            esm_repr_layer=used_layer,
            token_budget=args.token_budget,
            seq_length=enc_len,
            esm_dtype=args.esm_dtype,
            n_pairs=args.lerp_pairs,
            seed=args.seed,
        )
        if lerp_rows:
            results["lerp_cycle_cos_dist_mean"] = float(
                np.mean([r["cycle_cos_dist"] for r in lerp_rows])
            )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["which", "key", "value"])
            for k, v in results.items():
                writer.writerow(["summary", k, v])
            for row in lerp_rows:
                writer.writerow([
                    "lerp",
                    f"pair{row['pair']}_a{row['alpha']}",
                    row["cycle_cos_dist"],
                ])
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
