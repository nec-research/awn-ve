#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/evaluate_pairwise.py
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

"""Correlate embedding pairwise distances with true pairwise BLOSUM distances.

Compares uncompressed fair-esm (default ``esm2_t33_650M_UR50D``) against
downstream_validation on the same sequence pairs. Hexbin panels mirror
``cont_dev/Repos/evaluate_pairwise.py`` (cosine and L2 vs BLOSUM62).

Example::

    uv run python -m downstream_validation.evaluate_pairwise \\
        --fasta data/filtered_msa.fasta \\
        --model config_3 \\
        --limit 2000 \\
        --output eval_pairwise.csv \\
        --plot eval_pairwise.png

When both encoders run, also writes ``eval_pairwise_geometry.png``: hexbins of
pairwise ``D_teacher`` vs ``D_student`` (cosine and L2).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from downstream_validation.loader import load_model
from downstream_validation.splits import aligned_splits

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}

_BLOSUM62_RAW = """\
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""

DEFAULT_SEED = 42


def _parse_blosum62() -> dict[tuple[str, str], int]:
    lines = _BLOSUM62_RAW.strip().splitlines()
    header = lines[0].split()
    matrix: dict[tuple[str, str], int] = {}
    for line in lines[1:]:
        parts = line.split()
        row_aa = parts[0]
        for col_aa, score in zip(header, parts[1:]):
            matrix[(row_aa, col_aa)] = int(score)
    return matrix


BLOSUM62 = _parse_blosum62()


def _build_b62_tensor() -> torch.Tensor:
    t = torch.zeros((len(AMINO_ACIDS), len(AMINO_ACIDS)), dtype=torch.float32)
    for a, ai in AA_TO_IDX.items():
        for b, bi in AA_TO_IDX.items():
            t[ai, bi] = float(BLOSUM62[(a, b)])
    return t


B62_TENSOR = _build_b62_tensor()


def _blosum_masked_tensors(
    seqs: list[str], device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build BLOSUM tensors that zero out non-canonical (gap) columns.

    Only columns where both residues are among the 20 canonical amino acids
    contribute. Returns
    ``(r_masked, onehot_masked, selfdiag, canon)`` where contributions from
    non-canonical positions are set to zero, so pairwise scores implicitly use
    only mutually-canonical columns.
    """
    seq_idx = torch.tensor(
        [[AA_TO_IDX.get(a, -1) for a in s] for s in seqs],
        dtype=torch.long,
        device=device,
    )
    canon = (seq_idx >= 0).to(torch.float32)  # (N, L) 1 where canonical
    safe = seq_idx.clamp(min=0)
    b62 = B62_TENSOR.to(device) if device is not None else B62_TENSOR
    r_masked = b62[safe] * canon.unsqueeze(-1)  # (N, L, 20)
    onehot_masked = F.one_hot(safe, len(AMINO_ACIDS)).to(torch.float32) * canon.unsqueeze(-1)
    selfdiag = b62[safe, safe] * canon  # (N, L) self-score, zeroed at gaps
    return r_masked, onehot_masked, selfdiag, canon


def pairwise_blosum_distance_matrix(seqs: list[str]) -> torch.Tensor:
    """``(N, N)`` normalized BLOSUM62 distance over mutually-canonical columns.

    ``D_ij = 1 - S_ij / max(S_ii, S_jj)`` where every sum is restricted to the
    columns where both sequences carry a canonical amino acid; gaps are excluded.
    """
    if not seqs:
        raise ValueError("No sequences provided.")
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(
            "Pairwise BLOSUM distance requires equal-length sequences, "
            f"found lengths {sorted(lengths)}."
        )
    if lengths.pop() == 0:
        raise ValueError("Sequences are empty; cannot compute BLOSUM distances.")

    r_masked, onehot_masked, selfdiag, canon = _blosum_masked_tensors(seqs)
    observed = torch.einsum("ilb,jlb->ij", r_masked, onehot_masked)
    # Self-scores restricted to the columns the *other* sequence also has canonical.
    self_i = selfdiag @ canon.T  # (i, j): sum_l [canon_i & canon_j] B62(i,i)
    self_j = canon @ selfdiag.T  # (i, j): sum_l [canon_i & canon_j] B62(j,j)
    max_st = torch.maximum(self_i, self_j)
    out = torch.where(max_st > 0, 1.0 - observed / max_st, torch.zeros_like(observed))
    out.fill_diagonal_(0.0)
    out = 0.5 * (out + out.T)
    return out.to(torch.float32)


def cosine_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Return the ``(N, N)`` symmetric cosine distance (``1 - cos``) matrix."""
    x = F.normalize(embeddings.detach().to(torch.float32), dim=1)
    d = 1.0 - (x @ x.T)
    d.clamp_(min=0.0, max=2.0)
    d.fill_diagonal_(0.0)
    return 0.5 * (d + d.T)


def l2_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Return the ``(N, N)`` Euclidean distance matrix with a zeroed diagonal."""
    x = embeddings.detach().to(torch.float32)
    d = torch.cdist(x, x)
    d.fill_diagonal_(0.0)
    return d


def upper_triangle_flat(matrix: torch.Tensor) -> torch.Tensor:
    """Flatten the strict upper triangle (``offset=1``) of a square matrix."""
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {tuple(matrix.shape)}.")
    iu, ju = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    return matrix[iu, ju]


def correlate(x: np.ndarray, y: np.ndarray) -> dict:
    """Return Pearson/Spearman coefficients and p-values for two flat arrays.

    Falls back to a Pearson-only estimate via ``numpy`` when SciPy is absent.
    """
    n_pairs = int(x.size)
    result = {
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "n_pairs": n_pairs,
    }
    try:
        from scipy import stats as _stats
    except ImportError:
        cm = np.corrcoef(x, y)
        result["pearson_r"] = float(cm[0, 1])
        return result

    pr = _stats.pearsonr(x, y)
    sr = _stats.spearmanr(x, y)
    result["pearson_r"] = float(getattr(pr, "statistic", pr[0]))
    result["pearson_p"] = float(getattr(pr, "pvalue", pr[1]))
    result["spearman_rho"] = float(getattr(sr, "statistic", sr[0]))
    result["spearman_p"] = float(getattr(sr, "pvalue", sr[1]))
    return result


def parse_fasta(path: str) -> tuple[list[str], list[str]]:
    """Read a FASTA file into ``(labels, sequences)``, normalizing gaps to ``-``."""
    labels: list[str] = []
    seqs: list[str] = []
    cur_label: str | None = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal cur_label, parts
        if cur_label is None:
            return
        labels.append(cur_label)
        # Match esm-surrogate read_fasta: uppercase and normalize '.' gaps to '-'
        # so the fair-esm tokenizer sees the same gap symbol the teacher used.
        seqs.append("".join(parts).upper().replace(".", "-"))
        cur_label = None
        parts = []

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur_label = line[1:].strip() or f"seq_{len(labels)}"
            else:
                parts.append(line)
        flush()
    if not seqs:
        raise ValueError(f"No sequences found in {path}")
    return labels, seqs


def _token_batches(seqs: list[str], token_budget: int, seq_length: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for i, seq in enumerate(seqs):
        cost = min(len(seq), seq_length) + 2
        if current and current_tokens + cost > token_budget:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(i)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches


def _mean_pool_residue_reps(
    tokens: torch.Tensor,
    reps: torch.Tensor,
    alphabet,
) -> torch.Tensor:
    mask = (
        tokens.ne(alphabet.padding_idx)
        & tokens.ne(alphabet.cls_idx)
        & tokens.ne(alphabet.eos_idx)
    )
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(reps.dtype)
    return (reps * mask.unsqueeze(-1)).sum(dim=1) / counts


@torch.inference_mode()
def embed_awn(
    model,
    target_mean: torch.Tensor | None,
    seqs: list[str],
    *,
    device: torch.device,
    token_budget: int,
    seq_length: int,
) -> torch.Tensor:
    """Mean-pool embeddings from an AWN model, restoring the teacher mean if given."""
    alphabet = model.backbone.alphabet
    converter = alphabet.get_batch_converter(seq_length)
    labels = [f"seq_{i}" for i in range(len(seqs))]
    batches = _token_batches(seqs, token_budget, seq_length)
    out = torch.empty((len(seqs), int(model.metadata["embed_dim"])), dtype=torch.float32)
    for batch_idxs in tqdm(batches, desc="Encoding AWN"):
        batch = [(labels[i], seqs[i]) for i in batch_idxs]
        _batch_labels, _batch_strs, tokens = converter(batch)
        emb = model(tokens.to(device))
        if target_mean is not None:
            emb = emb + target_mean
        out[batch_idxs] = emb.detach().cpu().float()
    return out


@torch.inference_mode()
def embed_raw_esm(
    seqs: list[str],
    *,
    device: torch.device,
    model_name: str,
    repr_layer: int | None,
    token_budget: int,
    seq_length: int,
    esm_dtype: str = "fp32",
) -> tuple[torch.Tensor, str, int]:
    """Mean-pool residue reps from a pretrained fair-esm model."""
    import esm

    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(
            f"Unknown fair-esm pretrained model {model_name!r}. "
            "Examples: esm2_t33_650M_UR50D, esm1_t34_670M_UR100."
        )
    model, alphabet = loader()
    model = model.to(device).eval()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[esm_dtype]
    if dtype != torch.float32:
        if device.type == "cpu" and dtype == torch.float16:
            raise ValueError("fp16 is not supported on CPU; use bf16 or fp32.")
        model = model.to(dtype)

    num_layers = int(model.num_layers)
    layer = num_layers if repr_layer is None else int(repr_layer)
    if layer < 1 or layer > num_layers:
        raise ValueError(f"--raw-esm-repr-layer must be in [1, {num_layers}], got {layer}.")

    converter = alphabet.get_batch_converter(seq_length)
    labels = [f"seq_{i}" for i in range(len(seqs))]
    batches = _token_batches(seqs, token_budget, seq_length)
    # Probe embed dim from first forward.
    first = batches[0]
    _l, _s, tok0 = converter([(labels[i], seqs[i]) for i in first])
    tok0 = tok0.to(device)
    out0 = model(tok0, repr_layers=[layer])
    dim = int(out0["representations"][layer].shape[-1])
    embeddings = torch.empty((len(seqs), dim), dtype=torch.float32)
    embeddings[first] = _mean_pool_residue_reps(
        tok0, out0["representations"][layer], alphabet
    ).float().cpu()

    for batch_idxs in tqdm(batches[1:], desc=f"Encoding {model_name}"):
        batch = [(labels[i], seqs[i]) for i in batch_idxs]
        _batch_labels, _batch_strs, tokens = converter(batch)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[layer])
        embeddings[batch_idxs] = _mean_pool_residue_reps(
            tokens, out["representations"][layer], alphabet
        ).float().cpu()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings, model_name, layer


def add_distance_panels(
    label_prefix: str,
    embeddings: torch.Tensor,
    d_true_ut: np.ndarray,
    results: dict[str, dict],
    panels: list[tuple[str, np.ndarray, dict]],
) -> None:
    """Correlate an embedding's cosine/L2 distances against BLOSUM and record them."""
    d_cos = cosine_distance_matrix(embeddings)
    d_cos_ut = upper_triangle_flat(d_cos).numpy()
    stats_cos = correlate(d_true_ut, d_cos_ut)
    key_cos = f"{label_prefix}_cosine"
    print(_format_row(key_cos, stats_cos))
    results[key_cos] = stats_cos
    panels.append((f"{label_prefix} cosine", d_cos_ut, stats_cos))

    d_l2 = l2_distance_matrix(embeddings)
    d_l2_ut = upper_triangle_flat(d_l2).numpy()
    stats_l2 = correlate(d_true_ut, d_l2_ut)
    key_l2 = f"{label_prefix}_l2"
    print(_format_row(key_l2, stats_l2))
    results[key_l2] = stats_l2
    panels.append((f"{label_prefix} L2", d_l2_ut, stats_l2))


def teacher_student_pairwise_geometry(
    teacher: torch.Tensor,
    student: torch.Tensor,
) -> dict[str, np.ndarray | dict]:
    """Correlate pairwise distance matrices of teacher vs student embeddings."""
    if teacher.shape != student.shape:
        raise ValueError(
            f"Teacher/student shape mismatch: {tuple(teacher.shape)} vs "
            f"{tuple(student.shape)}."
        )
    d_t_cos = upper_triangle_flat(cosine_distance_matrix(teacher)).numpy()
    d_s_cos = upper_triangle_flat(cosine_distance_matrix(student)).numpy()
    d_t_l2 = upper_triangle_flat(l2_distance_matrix(teacher)).numpy()
    d_s_l2 = upper_triangle_flat(l2_distance_matrix(student)).numpy()
    stats_cos = correlate(d_t_cos, d_s_cos)
    stats_l2 = correlate(d_t_l2, d_s_l2)
    return {
        "d_teacher_cosine": d_t_cos,
        "d_student_cosine": d_s_cos,
        "d_teacher_l2": d_t_l2,
        "d_student_l2": d_s_l2,
        "stats_cosine": stats_cos,
        "stats_l2": stats_l2,
    }


def plot_pairwise_geometry(
    plot_path: str,
    geometry: dict[str, np.ndarray | dict],
    n_seqs: int,
    max_points: int = 200_000,
    seed: int = DEFAULT_SEED,
) -> None:
    """Hexbin pairwise D_teacher vs D_student (cosine and L2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (
            "pairwise cosine",
            np.asarray(geometry["d_teacher_cosine"]),
            np.asarray(geometry["d_student_cosine"]),
            geometry["stats_cosine"],
        ),
        (
            "pairwise L2",
            np.asarray(geometry["d_teacher_l2"]),
            np.asarray(geometry["d_student_l2"]),
            geometry["stats_l2"],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    for ax, (label, x, y, stats) in zip(axes, panels):
        n_pairs = int(x.size)
        if n_pairs > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_pairs, size=max_points, replace=False)
            x_plot, y_plot = x[idx], y[idx]
        else:
            x_plot, y_plot = x, y
        hb = ax.hexbin(x_plot, y_plot, gridsize=60, mincnt=1, cmap="viridis")
        lim_lo = float(min(x_plot.min(), y_plot.min()))
        lim_hi = float(max(x_plot.max(), y_plot.max()))
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="white", linestyle="--", linewidth=1)
        ax.set_xlabel("teacher pairwise distance")
        ax.set_ylabel("student pairwise distance")
        ax.set_title(
            f"{label}  (N={n_seqs}, pairs={n_pairs})\n"
            f"r = {stats['pearson_r']:+.3f}   "
            f"rho = {stats['spearman_rho']:+.3f}"
        )
        fig.colorbar(hb, ax=ax, label="pair count")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved geometry plot to {plot_path}")


class _PearsonAccumulator:
    """Streaming exact Pearson r over pair streams (no full materialization)."""

    def __init__(self) -> None:
        self.n = 0.0
        self.sx = 0.0
        self.sy = 0.0
        self.sxy = 0.0
        self.sxx = 0.0
        self.syy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.double()
        y = y.double()
        self.n += float(x.numel())
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxy += float((x * y).sum())
        self.sxx += float((x * x).sum())
        self.syy += float((y * y).sum())

    def pearson(self) -> float:
        n = self.n
        if n < 2:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / n
        vx = self.sxx - self.sx * self.sx / n
        vy = self.syy - self.sy * self.sy / n
        denom = (vx * vy) ** 0.5
        return float(cov / denom) if denom > 0 else float("nan")

    def result(self) -> dict:
        return {
            "pearson_r": self.pearson(),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n_pairs": int(self.n),
        }


@torch.inference_mode()
def streamed_test_correlations(
    seqs: list[str],
    embeddings_by_name: dict[str, torch.Tensor],
    *,
    device: torch.device,
    block: int = 1024,
) -> dict[str, dict]:
    """Exact Pearson over *all* upper-triangle pairs via row/col blocking.

    Computes, for every named embedding set, Pearson r of cosine and L2
    pairwise distance against the true BLOSUM62 pairwise distance, streaming so
    the full ``N(N-1)/2`` pair arrays are never held in memory. Spearman is left
    as NaN here (use the subsampled path for rank statistics).
    """
    n = len(seqs)
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"Equal-length sequences required, got {sorted(lengths)}.")

    # Masked tensors so only mutually-canonical columns contribute (gaps excluded).
    r_all, onehot_all, selfdiag, canon = _blosum_masked_tensors(seqs, device=device)

    norm_by_name = {
        name: F.normalize(emb.to(device).float(), dim=1)
        for name, emb in embeddings_by_name.items()
    }
    raw_by_name = {name: emb.to(device).float() for name, emb in embeddings_by_name.items()}

    accs = {
        f"{name}_{metric}": _PearsonAccumulator()
        for name in embeddings_by_name
        for metric in ("cosine", "l2")
    }

    for r0 in tqdm(range(0, n, block), desc="Streaming BLOSUM pairs"):
        r1 = min(r0 + block, n)
        rows = slice(r0, r1)
        r_mat = r_all[rows]  # (b, L, 20), already gap-masked
        selfdiag_row = selfdiag[rows]
        canon_row = canon[rows]
        for c0 in range(r0, n, block):
            c1 = min(c0 + block, n)
            cols = slice(c0, c1)
            observed = torch.einsum("ilb,jlb->ij", r_mat, onehot_all[cols])  # (b, c)
            self_i = selfdiag_row @ canon[cols].T  # (b, c)
            self_j = canon_row @ selfdiag[cols].T  # (b, c)
            max_st = torch.maximum(self_i, self_j)
            d_true = torch.where(
                max_st > 0, 1.0 - observed / max_st, torch.zeros_like(observed)
            )
            # Keep only strictly-upper-triangle pairs (global i < j).
            gi = torch.arange(r0, r1, device=device).unsqueeze(1)
            gj = torch.arange(c0, c1, device=device).unsqueeze(0)
            keep = gi < gj
            if not keep.any():
                continue
            d_true_f = d_true[keep]
            for name in embeddings_by_name:
                xn = norm_by_name[name]
                d_cos = 1.0 - xn[rows] @ xn[cols].T
                d_cos = d_cos.clamp(min=0.0, max=2.0)[keep]
                accs[f"{name}_cosine"].update(d_true_f, d_cos)
                d_l2 = torch.cdist(raw_by_name[name][rows], raw_by_name[name][cols])[keep]
                accs[f"{name}_l2"].update(d_true_f, d_l2)

    return {key: acc.result() for key, acc in accs.items()}


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    if p < 1e-300:
        return "<1e-300"
    return f"{p:.2e}"


def _format_row(label: str, stats: dict) -> str:
    return (
        f"{label:<32s}: pearson r = {stats['pearson_r']:+.4f} "
        f"(p={_format_p(stats['pearson_p'])})   "
        f"spearman rho = {stats['spearman_rho']:+.4f} "
        f"(p={_format_p(stats['spearman_p'])})"
    )


def _write_csv(path: str, rows: list[tuple[str, dict]], n_seqs: int) -> None:
    fields = [
        "which", "n_seqs", "n_pairs",
        "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for which, stats in rows:
            writer.writerow({
                "which": which,
                "n_seqs": n_seqs,
                "n_pairs": stats["n_pairs"],
                "pearson_r": f"{stats['pearson_r']:.6f}",
                "pearson_p": f"{stats['pearson_p']:.6e}",
                "spearman_rho": f"{stats['spearman_rho']:.6f}",
                "spearman_p": f"{stats['spearman_p']:.6e}",
            })


def plot_hexbins(
    plot_path: str,
    d_true_ut: np.ndarray,
    panels: list[tuple[str, np.ndarray, dict]],
    n_seqs: int,
    max_points: int = 200_000,
    seed: int = DEFAULT_SEED,
    ncols: int | None = None,
) -> None:
    """Render hexbins of embedding vs true BLOSUM distance, one panel per encoder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pairs = d_true_ut.size
    if n_pairs > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_pairs, size=max_points, replace=False)
        x_plot = d_true_ut[idx]
    else:
        idx = None
        x_plot = d_true_ut

    n_panels = len(panels)
    if ncols is None:
        ncols = 2 if n_panels >= 4 else n_panels
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 5.5 * nrows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    for ax, (label, y, stats) in zip(flat_axes, panels):
        y_plot = y[idx] if idx is not None else y
        hb = ax.hexbin(x_plot, y_plot, gridsize=60, mincnt=1, cmap="viridis")
        ax.set_xlabel("true BLOSUM pairwise distance")
        ax.set_ylabel("predicted pairwise distance")
        ax.set_title(
            f"{label}  (N={n_seqs}, pairs={n_pairs})\n"
            f"r = {stats['pearson_r']:+.3f}   "
            f"rho = {stats['spearman_rho']:+.3f}"
        )
        fig.colorbar(hb, ax=ax, label="pair count")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the BLOSUM distance-preservation run."""
    p = argparse.ArgumentParser(
        description=(
            "Hexbin cosine/L2 vs true BLOSUM: uncompressed ESM vs downstream_validation."
        ),
    )
    p.add_argument("--fasta", required=True, help="Equal-length (aligned) FASTA.")
    p.add_argument(
        "--model",
        default="config_3",
        help="Bundled AWN model ID or export directory (default: config_3).",
    )
    p.add_argument(
        "--raw-esm-model",
        default="esm2_t33_650M_UR50D",
        help="fair-esm pretrained loader name for uncompressed ESM.",
    )
    p.add_argument(
        "--raw-esm-repr-layer",
        type=int,
        default=None,
        help="ESM representation layer (default: last layer).",
    )
    p.add_argument(
        "--raw-esm-dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Compute dtype for uncompressed ESM forward.",
    )
    p.add_argument(
        "--skip-raw-esm",
        action="store_true",
        help="Only evaluate downstream_validation (no uncompressed ESM panels).",
    )
    p.add_argument(
        "--skip-awn",
        action="store_true",
        help="Only evaluate uncompressed ESM (no AWN panels).",
    )
    p.add_argument("--limit", type=int, default=None, help="Subsample this many sequences.")
    p.add_argument(
        "--use-aligned-test",
        action="store_true",
        help=(
            "Evaluate on the aligned, leakage-free test fold (the AWN surrogate's "
            "held-out protein IDs). Uses the *entire* test fold by default."
        ),
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Optional: subsample this many test sequences for an exact in-memory "
            "correlation instead of streaming over the whole test fold."
        ),
    )
    p.add_argument(
        "--block",
        type=int,
        default=1024,
        help="Row/column block size for full-test streaming.",
    )
    p.add_argument(
        "--plot-sample",
        type=int,
        default=2000,
        help="Sequences subsampled for hexbins/Spearman when streaming the full test.",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index; -1 for CPU.")
    p.add_argument("--token-budget", type=int, default=4096)
    p.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Max AA length for encoders (default: AWN metadata max, or 1024 if --skip-awn).",
    )
    p.add_argument("--output", default=None, help="Metrics CSV path.")
    p.add_argument("--plot", default=None, help="BLOSUM hexbin PNG path.")
    p.add_argument(
        "--geometry-plot",
        default=None,
        help=(
            "Teacher↔student pairwise geometry PNG (D_teacher vs D_student). "
            "Default: <plot>_geometry.png when --plot is set and both encoders run."
        ),
    )
    return p.parse_args()


def _run_full_test(
    args: argparse.Namespace,
    seqs: list[str],
    awn_model,
    awn_mean,
    seq_length: int,
    device: torch.device,
) -> None:
    """Exact Pearson over all pairs of the full aligned test fold (streamed)."""
    embeddings_by_name: dict[str, torch.Tensor] = {}
    if not args.skip_raw_esm:
        print(f"Embedding full test fold with uncompressed {args.raw_esm_model!r}...")
        raw_emb, raw_name, raw_layer = embed_raw_esm(
            seqs,
            device=device,
            model_name=args.raw_esm_model,
            repr_layer=args.raw_esm_repr_layer,
            token_budget=args.token_budget,
            seq_length=seq_length,
            esm_dtype=args.raw_esm_dtype,
        )
        print(f"  raw ESM layer={raw_layer}, dim={raw_emb.shape[1]}")
        embeddings_by_name[raw_name] = raw_emb
    if not args.skip_awn:
        print(f"Embedding full test fold with downstream_validation model {args.model!r}...")
        assert awn_model is not None and awn_mean is not None
        awn_emb = embed_awn(
            awn_model,
            awn_mean,
            seqs,
            device=device,
            token_budget=args.token_budget,
            seq_length=seq_length,
        )
        embeddings_by_name[f"awn_{args.model}"] = awn_emb

    print(
        f"Streaming exact Pearson over all {len(seqs)}*({len(seqs)}-1)/2 "
        f"= {len(seqs) * (len(seqs) - 1) // 2} pairs (block={args.block})..."
    )
    results = streamed_test_correlations(
        seqs, embeddings_by_name, device=device, block=args.block
    )
    for key, stats in results.items():
        print(_format_row(key, stats))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(str(out), list(results.items()), n_seqs=len(seqs))
        print(f"Saved CSV to {out}")

    if args.plot:
        # Hexbins + Spearman from a manageable subsample of the same fold.
        n_plot = min(args.plot_sample, len(seqs))
        rng = np.random.default_rng(args.seed)
        sub = np.sort(rng.choice(len(seqs), size=n_plot, replace=False))
        sub_seqs = [seqs[i] for i in sub]
        d_true_ut = upper_triangle_flat(pairwise_blosum_distance_matrix(sub_seqs)).numpy()
        panels: list[tuple[str, np.ndarray, dict]] = []
        results_sub: dict[str, dict] = {}
        for name, emb in embeddings_by_name.items():
            add_distance_panels(name, emb[sub], d_true_ut, results_sub, panels)
        plot = Path(args.plot)
        plot.parent.mkdir(parents=True, exist_ok=True)
        plot_hexbins(str(plot), d_true_ut, panels, n_seqs=n_plot, seed=args.seed)


def main() -> None:
    """CLI entry point for the BLOSUM distance-preservation experiment."""
    args = parse_args()
    if not os.path.isfile(args.fasta):
        raise SystemExit(f"FASTA not found: {args.fasta}")
    if args.skip_raw_esm and args.skip_awn:
        raise SystemExit("Cannot pass both --skip-raw-esm and --skip-awn.")

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    _labels, seqs = parse_fasta(args.fasta)
    use_full_test = False
    if args.use_aligned_test:
        test_idxs = aligned_splits(_labels, seed=args.seed)["test_idxs"]
        if args.sample is not None and args.sample > 0:
            n_pick = min(args.sample, len(test_idxs))
            rng = random.Random(args.seed)
            pick = sorted(rng.sample(test_idxs, n_pick))
            print(
                f"Sampled N={len(pick)} from the aligned test fold "
                f"({len(test_idxs)} held-out IDs, seed={args.seed})"
            )
        else:
            pick = test_idxs
            use_full_test = True
            print(f"Using the full aligned test fold: N={len(pick)} (leakage-free)")
        seqs = [seqs[i] for i in pick]
    elif args.sample is not None and args.sample > 0 and args.sample < len(seqs):
        rng = random.Random(args.seed)
        pick = sorted(rng.sample(range(len(seqs)), args.sample))
        seqs = [seqs[i] for i in pick]
        print(f"Subsampled to N={len(seqs)} (seed={args.seed})")
    elif args.limit is not None and args.limit > 0 and args.limit < len(seqs):
        rng = random.Random(args.seed)
        pick = sorted(rng.sample(range(len(seqs)), args.limit))
        seqs = [seqs[i] for i in pick]
        print(f"Subsampled to N={len(seqs)} (seed={args.seed})")
    else:
        print(f"Loaded N={len(seqs)} sequences from {args.fasta}")

    if len(seqs) < 2:
        raise SystemExit("Need at least 2 sequences.")

    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise SystemExit(
            "FASTA sequences must be equal length (pre-aligned MSA), "
            f"found lengths {sorted(lengths)}."
        )

    seq_length = args.seq_length
    awn_model = awn_mean = None
    if not args.skip_awn:
        awn_model, metadata, awn_mean = load_model(args.model, device=device)
        if seq_length is None:
            seq_length = int(metadata["max_sequence_length"])
        if seq_length < 1 or seq_length > int(metadata["max_sequence_length"]):
            raise SystemExit(
                f"--seq-length must be in [1, {metadata['max_sequence_length']}]."
            )
    elif seq_length is None:
        seq_length = max(lengths)

    if use_full_test:
        _run_full_test(args, seqs, awn_model, awn_mean, seq_length, device)
        return

    print(f"Computing pairwise BLOSUM distance matrix for N={len(seqs)}...")
    d_true = pairwise_blosum_distance_matrix(seqs)
    d_true_ut = upper_triangle_flat(d_true).numpy()
    print(
        f"  D_true stats: mean={d_true_ut.mean():.4f}  "
        f"min={d_true_ut.min():.4f}  max={d_true_ut.max():.4f}  "
        f"n_pairs={d_true_ut.size}"
    )

    panels: list[tuple[str, np.ndarray, dict]] = []
    results: dict[str, dict] = {}
    raw_emb = awn_emb = None
    raw_name = None

    if not args.skip_raw_esm:
        print(f"Embedding with uncompressed {args.raw_esm_model!r}...")
        raw_emb, raw_name, raw_layer = embed_raw_esm(
            seqs,
            device=device,
            model_name=args.raw_esm_model,
            repr_layer=args.raw_esm_repr_layer,
            token_budget=args.token_budget,
            seq_length=seq_length,
            esm_dtype=args.raw_esm_dtype,
        )
        print(f"  raw ESM layer={raw_layer}, dim={raw_emb.shape[1]}")
        add_distance_panels(raw_name, raw_emb, d_true_ut, results, panels)

    if not args.skip_awn:
        print(f"Embedding with downstream_validation model {args.model!r}...")
        assert awn_model is not None and awn_mean is not None
        awn_emb = embed_awn(
            awn_model,
            awn_mean,
            seqs,
            device=device,
            token_budget=args.token_budget,
            seq_length=seq_length,
        )
        add_distance_panels(f"awn_{args.model}", awn_emb, d_true_ut, results, panels)

    # Prefer [raw_cos, awn_cos, raw_l2, awn_l2] for side-by-side comparison.
    if not args.skip_raw_esm and not args.skip_awn and len(panels) == 4:
        panels = [panels[0], panels[2], panels[1], panels[3]]

    geometry = None
    if raw_emb is not None and awn_emb is not None:
        if raw_emb.shape[1] != awn_emb.shape[1]:
            print(
                f"Skipping geometry plot: embed dim mismatch "
                f"{raw_emb.shape[1]} vs {awn_emb.shape[1]}."
            )
        else:
            assert raw_name is not None
            print(
                "Teacher↔student pairwise geometry "
                f"({raw_name} vs awn_{args.model})..."
            )
            geometry = teacher_student_pairwise_geometry(raw_emb, awn_emb)
            print(_format_row("geometry_cosine", geometry["stats_cosine"]))
            print(_format_row("geometry_l2", geometry["stats_l2"]))
            results["geometry_cosine"] = geometry["stats_cosine"]
            results["geometry_l2"] = geometry["stats_l2"]

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(str(out), list(results.items()), n_seqs=len(seqs))
        print(f"Saved CSV to {out}")

    if args.plot:
        plot = Path(args.plot)
        plot.parent.mkdir(parents=True, exist_ok=True)
        plot_hexbins(str(plot), d_true_ut, panels, n_seqs=len(seqs), seed=args.seed)

    geometry_path = args.geometry_plot
    if geometry_path is None and args.plot and geometry is not None:
        plot = Path(args.plot)
        geometry_path = str(plot.with_name(f"{plot.stem}_geometry{plot.suffix}"))
    if geometry is not None and geometry_path:
        Path(geometry_path).parent.mkdir(parents=True, exist_ok=True)
        plot_pairwise_geometry(
            geometry_path, geometry, n_seqs=len(seqs), seed=args.seed,
        )


if __name__ == "__main__":
    main()
