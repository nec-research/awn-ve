#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/lineage_classification.py
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

"""Lineage classification probe on encoder embeddings (leakage-free split).

Fits a multinomial-logistic (softmax) linear probe in PyTorch on mean-pooled
embeddings and evaluates it on the aligned, held-out test fold. Fine-grained
Pango lineages are collapsed to WHO variant groups (see
:func:`downstream_validation.splits.collapse_lineage`).

Embeddings are reused from the decoder embedding caches when available (they are
pooled ``[N, D]`` in FASTA order), otherwise they are computed on the fly.

Examples::

    # From a cached pooled-embedding file (fast, no GPU embedding pass):
    uv run python -m downstream_validation.lineage_classification \\
        --fasta data/filtered_msa.fasta \\
        --encoder-name config_2 \\
        --embedding-cache embeddings/awn_config2/<cache>.pt \\
        --output-dir results/paper/lineage/config_2

    # Embedding on the fly with an AWN model:
    uv run python -m downstream_validation.lineage_classification \\
        --fasta data/filtered_msa.fasta \\
        --encoder awn --awn-model config_1 --encoder-name config_1 \\
        --output-dir results/paper/lineage/config_1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from downstream_validation.evaluate_pairwise import (
    DEFAULT_SEED,
    embed_awn,
    embed_raw_esm,
    parse_fasta,
)
from downstream_validation.loader import load_model
from downstream_validation.splits import aligned_splits, group_from_label

DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"


def load_cached_embeddings(path: str) -> tuple[torch.Tensor, list[str] | None]:
    """Load pooled embeddings (and optional labels) from a cached tensor file."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "embeddings" in payload:
        return payload["embeddings"].float(), payload.get("labels")
    if isinstance(payload, torch.Tensor):
        return payload.float(), None
    raise SystemExit(f"Unrecognized embedding cache format: {path}")


def embed_sequences(
    args: argparse.Namespace,
    seqs: list[str],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Mean-pool embeddings for ``seqs`` using the raw ESM-2 or AWN encoder."""
    if args.encoder == "awn":
        print(f"Embedding N={len(seqs)} with AWN {args.awn_model}...")
        model, metadata, target_mean = load_model(args.awn_model, device=device)
        enc_len = min(seq_len, int(metadata.get("max_sequence_length", seq_len)))
        emb = embed_awn(
            model, target_mean, seqs,
            device=device, token_budget=args.token_budget, seq_length=enc_len,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return emb
    print(f"Embedding N={len(seqs)} with {args.esm_model}...")
    emb, _, _ = embed_raw_esm(
        seqs,
        device=device,
        model_name=args.esm_model,
        repr_layer=args.esm_repr_layer,
        token_budget=args.token_budget,
        seq_length=seq_len,
        esm_dtype=args.esm_dtype,
    )
    return emb


class LinearProbe(nn.Module):
    """Single linear layer mapping frozen embeddings to lineage-class logits."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of embeddings."""
        return self.linear(x)


def macro_f1(conf: np.ndarray) -> float:
    """Compute the unweighted mean per-class F1 from a confusion matrix."""
    f1s = []
    for c in range(conf.shape[0]):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def train_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    class_weights: torch.Tensor | None,
    seed: int,
) -> LinearProbe:
    """Train a linear probe with AdamW, returning the best-validation model."""
    torch.manual_seed(seed)
    model = LinearProbe(x_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights.to(device) if class_weights is not None else None

    n = x_train.shape[0]
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = -1.0
    g = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx].to(device, non_blocking=True)
            yb = y_train[idx].to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val.to(device)).argmax(1).cpu()
        val_acc = float((val_pred == y_val).float().mean())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  epoch {epoch:03d}/{epochs}  loss={float(loss):.4f}  val_acc={val_acc:.4f}")
    model.load_state_dict(best_state)
    print(f"  best val_acc={best_val:.4f}")
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the lineage-probe experiment."""
    p = argparse.ArgumentParser(description="Lineage classification linear probe.")
    p.add_argument("--fasta", default="data/filtered_msa.fasta")
    p.add_argument("--encoder-name", required=True,
                   help="Short name for outputs (e.g. esm2, config_1, config_2, config_3).")
    p.add_argument("--embedding-cache", default=None,
                   help="Pooled-embedding cache .pt (reused if given).")
    p.add_argument("--encoder", choices=["esm2", "awn"], default="esm2",
                   help="On-the-fly embedding source when no cache is given.")
    p.add_argument("--awn-model", default="config_2")
    p.add_argument("--esm-model", default=DEFAULT_ESM_MODEL)
    p.add_argument("--esm-repr-layer", type=int, default=None)
    p.add_argument("--esm-dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--token-budget", type=int, default=16384)
    p.add_argument("--seq-length", type=int, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--class-weighting", action="store_true",
                   help="Use inverse-frequency class weights (imbalanced groups).")
    p.add_argument("--min-class-count", type=int, default=1,
                   help="Drop WHO groups with fewer than this many train examples.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    """CLI entry point: train and evaluate the lineage linear probe."""
    args = parse_args()
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    if not os.path.isfile(args.fasta):
        raise SystemExit(f"FASTA not found: {args.fasta}")

    labels, seqs = parse_fasta(args.fasta)
    seq_len = len(seqs[0])

    if args.embedding_cache:
        print(f"Loading cached embeddings: {args.embedding_cache}")
        embeddings, cache_labels = load_cached_embeddings(args.embedding_cache)
        if embeddings.shape[0] != len(labels):
            raise SystemExit(
                f"Cache N={embeddings.shape[0]} != FASTA N={len(labels)}."
            )
        if cache_labels is not None and list(cache_labels) != list(labels):
            raise SystemExit("Cache labels do not match FASTA order; cannot align.")
    else:
        embeddings = embed_sequences(args, seqs, args.seq_length or seq_len, device)

    splits = aligned_splits(labels, seed=args.seed)
    groups = np.array([group_from_label(l) for l in labels])

    # Class vocabulary from the training fold only (leakage-free).
    train_idx = np.array(splits["train_idxs"])
    val_idx = np.array(splits["val_idxs"])
    test_idx = np.array(splits["test_idxs"])

    train_groups = groups[train_idx]
    uniq, counts = np.unique(train_groups, return_counts=True)
    classes = sorted(u for u, c in zip(uniq, counts) if c >= args.min_class_count)
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    def select(split_idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, int]:
        keep = [i for i in split_idx if groups[i] in class_to_id]
        dropped = len(split_idx) - len(keep)
        keep = np.array(keep)
        y = torch.tensor([class_to_id[groups[i]] for i in keep], dtype=torch.long)
        x = embeddings[torch.from_numpy(keep)]
        return x, y, dropped

    x_train, y_train, drop_tr = select(train_idx)
    x_val, y_val, drop_va = select(val_idx)
    x_test, y_test, drop_te = select(test_idx)
    print(
        f"train={len(y_train)}(-{drop_tr})  val={len(y_val)}(-{drop_va})  "
        f"test={len(y_test)}(-{drop_te}) dropped (unseen classes)"
    )

    # Standardize features with train statistics.
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True).clamp(min=1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    class_weights = None
    if args.class_weighting:
        freq = torch.bincount(y_train, minlength=len(classes)).float()
        class_weights = (freq.sum() / (len(classes) * freq.clamp(min=1)))

    model = train_probe(
        x_train, y_train, x_val, y_val, len(classes),
        device=device, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, batch_size=args.batch_size,
        class_weights=class_weights, seed=args.seed,
    )

    model.eval()
    with torch.no_grad():
        test_pred = model(x_test.to(device)).argmax(1).cpu().numpy()
    y_test_np = y_test.numpy()

    conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y_test_np, test_pred):
        conf[t, p] += 1
    acc = float((test_pred == y_test_np).mean())
    # Balanced accuracy = mean per-class recall.
    recalls = [
        conf[c, c] / conf[c, :].sum() if conf[c, :].sum() else 0.0
        for c in range(len(classes))
    ]
    bal_acc = float(np.mean(recalls))
    f1 = macro_f1(conf)
    print(
        f"TEST  encoder={args.encoder_name}  acc={acc:.4f}  "
        f"balanced_acc={bal_acc:.4f}  macro_f1={f1:.4f}  N={len(y_test_np)}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "encoder": args.encoder_name,
        "n_classes": len(classes),
        "classes": classes,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test_np)),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": f1,
        "per_class_recall": {c: float(r) for c, r in zip(classes, recalls)},
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(out_dir / "confusion_matrix.npy", conf)
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["encoder", "metric", "value"])
        w.writerow([args.encoder_name, "accuracy", acc])
        w.writerow([args.encoder_name, "balanced_accuracy", bal_acc])
        w.writerow([args.encoder_name, "macro_f1", f1])
        for c, r in zip(classes, recalls):
            w.writerow([args.encoder_name, f"recall::{c}", r])
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
