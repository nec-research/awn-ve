#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/train_decoder.py
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

"""Train an AR sequence decoder conditioned on encoder embeddings.

Default encoder is ``esm2_t33_650M_UR50D`` (mean-pooled). Use ``--encoder awn``
to train on exported AWN latents (e.g. ``config_2``).

Example (ESM-2)::

    uv run python -m downstream_validation.train_decoder \\
        --fasta data/filtered_msa.fasta \\
        --test-fraction 0.15 \\
        --run-name decoder_esm2_fp32 \\
        --token-budget 16384 \\
        --esm-dtype fp32

Example (AWN config_2)::

    uv run python -m downstream_validation.train_decoder \\
        --fasta data/filtered_msa.fasta \\
        --encoder awn \\
        --awn-model config_2 \\
        --test-fraction 0.15 \\
        --seed 42 \\
        --run-name decoder_awn_config2 \\
        --token-budget 16384 \\
        --embedding-cache-dir embeddings/awn_config2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from downstream_validation.decoder import (
    DECODER_RESIDUES,
    SequenceDecoder,
    _CHAR_TO_ID,
    tokenize_sequences,
)
from downstream_validation.evaluate_pairwise import DEFAULT_SEED, embed_awn, embed_raw_esm, parse_fasta
from downstream_validation.loader import load_model
from downstream_validation.splits import aligned_splits

DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"


class DecoderSplitDataset(Dataset):
    """Pairs per-sequence latent embeddings with their target token ids."""

    def __init__(self, embeddings: torch.Tensor, tokens: torch.Tensor):
        if embeddings.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"embeddings rows ({embeddings.shape[0]}) != "
                f"tokens rows ({tokens.shape[0]})."
            )
        self.embeddings = embeddings.float()
        self.tokens = tokens.long()

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.tokens[idx]

    @staticmethod
    def collate(batch):
        z = torch.stack([b[0] for b in batch])
        tokens = torch.stack([b[1] for b in batch])
        return z, tokens


def token_accuracy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    ignore_gap: bool = False,
) -> float:
    """Return mean per-position token accuracy, optionally ignoring gap targets."""
    pred = logits.argmax(dim=-1)
    if ignore_gap:
        gap_id = _CHAR_TO_ID["-"]
        mask = tokens != gap_id
        if not mask.any():
            return float("nan")
        return float((pred[mask] == tokens[mask]).float().mean().item())
    return float((pred == tokens).float().mean().item())


@torch.no_grad()
def run_eval(
    model: SequenceDecoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Teacher-forced evaluation; returns mean (loss, accuracy, non-gap accuracy)."""
    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    gap_accs: list[float] = []
    for z, tokens in loader:
        z = z.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        logits = model(z, tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            tokens.reshape(-1),
        )
        losses.append(float(loss.item()))
        accs.append(token_accuracy(logits, tokens, ignore_gap=False))
        gap_accs.append(token_accuracy(logits, tokens, ignore_gap=True))
    return (
        float(np.mean(losses)) if losses else float("nan"),
        float(np.mean(accs)) if accs else float("nan"),
        float(np.mean(gap_accs)) if gap_accs else float("nan"),
    )


def run_epoch(
    model: SequenceDecoder,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Run one teacher-forced training epoch; returns mean (loss, accuracy)."""
    model.train()
    losses: list[float] = []
    accs: list[float] = []
    for z, tokens in loader:
        z = z.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        logits = model(z, tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            tokens.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.item()))
        accs.append(token_accuracy(logits.detach(), tokens, ignore_gap=False))
    return float(np.mean(losses)), float(np.mean(accs))


def save_checkpoint(
    path: str,
    model: SequenceDecoder,
    optimizer,
    epoch: int,
    loss: float,
    **extra,
) -> None:
    """Save decoder weights, optimizer state, and hyperparameters to ``path``."""
    payload = {
        "epoch": epoch,
        "loss": loss,
        "decoder_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "latent_dim": model.latent_dim,
        "seq_len": model.seq_len,
        "d_model": model.d_model,
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "ff_dim": model.ff_dim,
        "dropout": model.dropout_p,
        "vocab_size": model.vocab_size,
        "bos_id": model.bos_id,
        "alphabet": DECODER_RESIDUES,
        "input_mode": "raw",
    }
    payload.update(extra)
    torch.save(payload, path)


def embedding_cache_path(
    cache_dir: str,
    *,
    fasta: str,
    encoder: str,
    encoder_id: str,
    repr_layer: int | None,
    n_seqs: int,
    seq_len: int,
    limit: int | None,
    seed: int,
    content_fingerprint: str,
) -> Path:
    """Build the deterministic cache filename for a pooled-embedding tensor."""
    if encoder == "awn":
        layer_tag = "awn"
    else:
        layer_tag = "last" if repr_layer is None else str(repr_layer)
    lim = "all" if limit is None else f"N{limit}_s{seed}"
    safe_id = encoder_id.replace("/", "_")
    name = (
        f"{Path(fasta).stem}.{encoder}.{safe_id}.L{layer_tag}.{lim}."
        f"Lseq{seq_len}.N{n_seqs}.{content_fingerprint}.pt"
    )
    return Path(cache_dir) / name


def make_splits(
    n: int,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Shuffle ``n`` indices with ``seed`` and split into (train, test) lists."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"--test-fraction must be in (0,1), got {test_fraction}.")
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_test = max(1, int(round(n * test_fraction)))
    n_test = min(n_test, n - 1)
    test_idxs = sorted(idxs[:n_test])
    train_idxs = sorted(idxs[n_test:])
    return train_idxs, test_idxs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for decoder training."""
    p = argparse.ArgumentParser(
        description="Train AR decoder conditioned on ESM-2 or AWN embeddings.",
    )
    p.add_argument("--fasta", default="data/filtered_msa.fasta")
    p.add_argument(
        "--encoder",
        choices=["esm2", "awn"],
        default="esm2",
        help="Embedding source: raw fair-esm (esm2) or exported AWN (awn).",
    )
    p.add_argument(
        "--awn-model",
        default="config_2",
        help="AWN bundle id or path (default: config_2). Used when --encoder awn.",
    )
    p.add_argument(
        "--esm-model",
        default=DEFAULT_ESM_MODEL,
        help="fair-esm pretrained loader name (default: esm2_t33_650M_UR50D).",
    )
    p.add_argument(
        "--esm-repr-layer",
        type=int,
        default=None,
        help="Representation layer (default: last layer).",
    )
    p.add_argument("--esm-dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--token-budget", type=int, default=16384)
    p.add_argument("--seq-length", type=int, default=None,
                   help="Encoder truncation length (default: full MSA length).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional subsample before train/val split (random split only).")
    p.add_argument(
        "--split",
        choices=["aligned", "random"],
        default="aligned",
        help=(
            "aligned: leakage-free protein-ID hold-out matched to the AWN "
            "surrogate (train/val/test); random: legacy index shuffle (train/val)."
        ),
    )
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--embedding-cache-dir", default="embeddings")
    p.add_argument("--no-embedding-cache", action="store_true")
    p.add_argument(
        "--build-cache-only",
        action="store_true",
        help="Compute and cache pooled embeddings, then exit (no decoder training).",
    )

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--best-metric", choices=["val_loss", "val_acc"], default="val_acc")
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--no-tensorboard", action="store_true")
    return p.parse_args()


def main() -> None:
    """CLI entry point: train a sequence decoder on cached encoder embeddings."""
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    if not os.path.isfile(args.fasta):
        raise SystemExit(f"FASTA not found: {args.fasta}")

    labels, seqs = parse_fasta(args.fasta)
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise SystemExit(f"Need equal-length MSA sequences, got {sorted(lengths)}")
    seq_len = lengths.pop()
    print(f"Loaded N={len(seqs)} sequences, L={seq_len} from {args.fasta}")

    if args.split == "aligned":
        if args.limit is not None and args.limit > 0:
            raise SystemExit(
                "--limit is incompatible with --split aligned (it reorders "
                "records and breaks the protein-ID alignment). Use --split random."
            )
        splits = aligned_splits(labels, seed=args.seed)
        train_idxs = splits["train_idxs"]
        val_idxs = splits["val_idxs"]
        test_idxs = splits["test_idxs"]
        print(
            f"Aligned split (leakage-free): train={len(train_idxs)} "
            f"val={len(val_idxs)} test={len(test_idxs)} (held out for eval)"
        )
    else:
        if args.limit is not None and args.limit > 0 and args.limit < len(seqs):
            rng = random.Random(args.seed)
            pick = sorted(rng.sample(range(len(seqs)), args.limit))
            labels = [labels[i] for i in pick]
            seqs = [seqs[i] for i in pick]
            print(f"Subsampled to N={len(seqs)} (seed={args.seed})")
        train_idxs, val_idxs = make_splits(len(seqs), args.test_fraction, args.seed)
        test_idxs = []
        print(f"Random split: train={len(train_idxs)} val={len(val_idxs)}")

    enc_len = args.seq_length or seq_len
    fingerprint = hashlib.sha1(
        (labels[0] + seqs[0][:64] + labels[-1] + seqs[-1][-64:] + str(len(seqs))).encode()
    ).hexdigest()[:12]

    cache_path = None
    embeddings = None
    esm_layer = args.esm_repr_layer
    encoder_id = args.awn_model if args.encoder == "awn" else args.esm_model

    if not args.no_embedding_cache:
        cache_path = embedding_cache_path(
            args.embedding_cache_dir,
            fasta=args.fasta,
            encoder=args.encoder,
            encoder_id=encoder_id,
            repr_layer=args.esm_repr_layer,
            n_seqs=len(seqs),
            seq_len=seq_len,
            limit=args.limit,
            seed=args.seed,
            content_fingerprint=fingerprint,
        )
        if cache_path.is_file():
            print(f"Loading embedding cache {cache_path}")
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            embeddings = payload["embeddings"].float()
            esm_layer = payload.get("esm_repr_layer")
            if esm_layer is not None:
                esm_layer = int(esm_layer)
                if esm_layer < 0:
                    esm_layer = None
            if embeddings.shape[0] != len(seqs):
                raise SystemExit(
                    f"Cache N={embeddings.shape[0]} != current N={len(seqs)}; "
                    "delete cache or change --limit/--seed."
                )

    if embeddings is None:
        if args.encoder == "awn":
            print(f"Embedding with AWN {args.awn_model!r}...")
            awn_model, metadata, target_mean = load_model(args.awn_model, device=device)
            if "max_sequence_length" in metadata:
                enc_len = min(enc_len, int(metadata["max_sequence_length"]))
            embeddings = embed_awn(
                awn_model,
                target_mean,
                seqs,
                device=device,
                token_budget=args.token_budget,
                seq_length=enc_len,
            )
            esm_layer = None
            print(f"  embeddings shape={tuple(embeddings.shape)} max_len={enc_len}")
            del awn_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"Embedding with {args.esm_model!r}...")
            embeddings, esm_name, esm_layer = embed_raw_esm(
                seqs,
                device=device,
                model_name=args.esm_model,
                repr_layer=args.esm_repr_layer,
                token_budget=args.token_budget,
                seq_length=enc_len,
                esm_dtype=args.esm_dtype,
            )
            encoder_id = esm_name
            print(f"  embeddings shape={tuple(embeddings.shape)} layer={esm_layer}")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "embeddings": embeddings,
                "labels": labels,
                "encoder": args.encoder,
                "seq_len": seq_len,
                "seed": args.seed,
                "limit": args.limit,
            }
            if args.encoder == "awn":
                payload["awn_model"] = args.awn_model
            else:
                payload["esm_model"] = encoder_id
                payload["esm_repr_layer"] = esm_layer
            torch.save(payload, cache_path)
            print(f"  cached to {cache_path}")

    if args.build_cache_only:
        where = cache_path if cache_path is not None else "(cache disabled)"
        print(
            f"build-cache-only: embeddings ready shape={tuple(embeddings.shape)} "
            f"at {where}. Exiting without training."
        )
        return

    tokens = tokenize_sequences(seqs)
    if tokens.shape[1] != seq_len:
        raise SystemExit("Token length mismatch.")

    train_emb = embeddings[train_idxs]
    val_emb = embeddings[val_idxs]
    train_tok = tokens[train_idxs]
    val_tok = tokens[val_idxs]
    latent_dim = int(embeddings.shape[1])

    default_prefix = (
        f"decoder_awn_{args.awn_model}" if args.encoder == "awn" else "decoder_esm2"
    )
    run_name = args.run_name or time.strftime(f"{default_prefix}_%Y%m%d_%H%M%S")
    run_dir = Path(args.ckpt_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_train": len(train_idxs),
        "n_val": len(val_idxs),
        "seq_len": seq_len,
        "latent_dim": latent_dim,
        "esm_repr_layer": esm_layer,
        "alphabet": DECODER_RESIDUES,
        "split": args.split,
        "n_test": len(test_idxs),
        "train_idxs": train_idxs,
        "val_idxs": val_idxs,
        "test_idxs": test_idxs,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = DecoderSplitDataset(train_emb, train_tok)
    val_ds = DecoderSplitDataset(val_emb, val_tok)
    train_bs = min(args.batch_size, len(train_ds))
    val_bs = min(args.batch_size, len(val_ds))
    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=DecoderSplitDataset.collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=DecoderSplitDataset.collate,
    )

    model = SequenceDecoder(
        latent_dim=latent_dim,
        seq_len=seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Decoder params: {n_params:,}  latent_dim={latent_dim} seq_len={seq_len}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min,
        )

    csv_path = run_dir / "metrics.csv"
    csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow([
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_nongap_acc", "lr",
    ])

    tb_writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        except Exception as exc:
            print(f"TensorBoard disabled ({exc})")

    best_val = float("inf") if args.best_metric == "val_loss" else float("-inf")
    best_epoch = 0
    stale_epochs = 0
    last_epoch = 0
    last_val_loss = float("nan")

    try:
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            train_loss, train_acc = run_epoch(
                model, train_loader, optimizer, device,
                max_grad_norm=args.max_grad_norm,
            )
            val_loss, val_acc, val_nongap = run_eval(model, val_loader, device)
            last_val_loss = val_loss
            lr = optimizer.param_groups[0]["lr"]

            if args.log_every > 0 and epoch % args.log_every == 0:
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | "
                    f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                    f"val loss {val_loss:.4f} acc {val_acc:.4f} "
                    f"nongap {val_nongap:.4f}"
                )

            csv_writer.writerow([
                epoch, train_loss, train_acc,
                val_loss, val_acc, val_nongap, lr,
            ])
            csv_fh.flush()

            if tb_writer is not None:
                tb_writer.add_scalar("loss/train", train_loss, epoch)
                tb_writer.add_scalar("loss/val", val_loss, epoch)
                tb_writer.add_scalar("acc/train", train_acc, epoch)
                tb_writer.add_scalar("acc/val", val_acc, epoch)
                tb_writer.add_scalar("acc/val_nongap", val_nongap, epoch)
                tb_writer.add_scalar("lr", lr, epoch)

            if args.best_metric == "val_loss":
                improved = val_loss < best_val
                metric_value = val_loss
            else:
                improved = val_acc > best_val
                metric_value = val_acc

            ckpt_extra = dict(
                encoder=args.encoder,
                split=args.split,
                train_idxs=train_idxs,
                val_idxs=val_idxs,
                test_idxs=test_idxs,
                best_metric=args.best_metric,
                best_metric_value=best_val if not improved else metric_value,
                val_acc=val_acc,
                val_nongap_acc=val_nongap,
            )
            if args.encoder == "awn":
                ckpt_extra["awn_model"] = args.awn_model
            else:
                ckpt_extra["esm_model"] = args.esm_model
                ckpt_extra["esm_repr_layer"] = esm_layer
            if improved:
                best_val = metric_value
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(
                    str(run_dir / "best.pt"),
                    model, optimizer, epoch, val_loss,
                    **ckpt_extra,
                )
            else:
                stale_epochs += 1

            if scheduler is not None:
                scheduler.step()

            if (
                args.early_stop_patience > 0
                and stale_epochs >= args.early_stop_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}: no improvement in "
                    f"{args.best_metric} for {args.early_stop_patience} epochs. "
                    f"Best epoch {best_epoch} ({args.best_metric}={best_val:.4f})."
                )
                break
    finally:
        csv_fh.close()
        if tb_writer is not None:
            tb_writer.close()

    final_extra = dict(
        encoder=args.encoder,
        split=args.split,
        train_idxs=train_idxs,
        val_idxs=val_idxs,
        test_idxs=test_idxs,
    )
    if args.encoder == "awn":
        final_extra["awn_model"] = args.awn_model
    else:
        final_extra["esm_model"] = args.esm_model
        final_extra["esm_repr_layer"] = esm_layer
    save_checkpoint(
        str(run_dir / "final.pt"),
        model, optimizer, last_epoch, last_val_loss,
        **final_extra,
    )
    enc_desc = (
        f"AWN {args.awn_model}"
        if args.encoder == "awn"
        else f"{args.esm_model} layer={esm_layer}"
    )
    print(
        f"Training complete.\n"
        f"  Encoder: {enc_desc}\n"
        f"  Best {args.best_metric}: {best_val:.4f} (epoch {best_epoch})\n"
        f"  Checkpoints: {run_dir / 'best.pt'}, {run_dir / 'final.pt'}\n"
        f"  Metrics: {csv_path}\n"
        f"  Evaluate: uv run python -m downstream_validation.evaluate_decoder "
        f"--fasta {args.fasta} "
        f"--decoder-checkpoint {run_dir / 'best.pt'} "
        + ("--use-checkpoint-test" if args.split == "aligned" else "--use-checkpoint-val")
    )


if __name__ == "__main__":
    main()
