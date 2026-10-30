#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/splits.py
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

"""Leakage-free data splits aligned with the AWN surrogate, plus lineage labels.

The surrogate (``esm-surrogate``) splits proteins with ``ProteinIDSplitter``: a
deterministic hold-out that orders unique protein IDs by a seeded SHA-256 hash
and carves out a test fold and an inner-validation fold. ``aligned_splits``
delegates directly to ``esm_surrogate.data.ProteinIDSplitter`` (imported lazily),
so downstream experiments in this repository (decoder, BLOSUM, lineage) evaluate
on exactly the held-out proteins the surrogate never trained on. Because of this,
computing the split requires ``esm_surrogate``/MLWiz to be importable; the
lineage/header helpers below do not.

It also parses the pipe-delimited FASTA headers used here, e.g.::

    >idx:1|genome_id:2697049.4653096|lineage:AY.38|collection_date:2021-07-08|isolation_country:South_Africa

and collapses fine-grained Pango lineages into WHO variant groups for the
lineage-classification experiment.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

# Expected record counts for the bundled filtered_msa.fasta (N = 163,184),
# matching the surrogate's reported split. Used as a self-check.
EXPECTED_TOTAL = 163_184
EXPECTED_COUNTS = {"train": 132_179, "val": 14_687, "test": 16_318}


# --------------------------------------------------------------------------- #
# Protein-ID hold-out (delegated to esm_surrogate.data.ProteinIDSplitter)
# --------------------------------------------------------------------------- #
class _ProteinIDView:
    """Minimal adapter exposing ``protein_ids`` to ``ProteinIDSplitter.split``."""

    def __init__(self, protein_ids: list[str]) -> None:
        self.protein_ids = protein_ids


def aligned_splits(
    labels: Sequence[str],
    *,
    seed: int = 42,
    test_ratio: float = 0.1,
    inner_val_ratio: float = 0.1,
    outer_val_ratio: float = 0.1,
    verify: bool = True,
) -> dict[str, list[int]]:
    """Reproduce the surrogate's train/val/test split as indices into ``labels``.

    Delegates to ``esm_surrogate.data.ProteinIDSplitter`` (imported lazily, so
    this call requires ``esm_surrogate``/MLWiz to be installed). Returns the inner
    train/validation folds and the outer test fold as 0-based indices into the
    given ``labels`` order (i.e. FASTA order from ``parse_fasta``).
    """
    from esm_surrogate.data import ProteinIDSplitter, protein_id_from_label

    protein_ids = [protein_id_from_label(label) for label in labels]
    splitter = ProteinIDSplitter(
        n_outer_folds=1,
        n_inner_folds=1,
        seed=seed,
        inner_val_ratio=inner_val_ratio,
        outer_val_ratio=outer_val_ratio,
        test_ratio=test_ratio,
    )
    splitter.split(_ProteinIDView(protein_ids))
    inner_fold = splitter.inner_folds[0][0]
    outer_fold = splitter.outer_folds[0]
    splits = {
        "train_idxs": list(inner_fold.train_idxs),
        "val_idxs": list(inner_fold.val_idxs),
        "test_idxs": list(outer_fold.test_idxs),
    }

    if verify and len(set(protein_ids)) == EXPECTED_TOTAL:
        got = {
            "train": len(splits["train_idxs"]),
            "val": len(splits["val_idxs"]),
            "test": len(splits["test_idxs"]),
        }
        if got != EXPECTED_COUNTS:
            raise AssertionError(
                f"Aligned split counts {got} do not match the surrogate's "
                f"{EXPECTED_COUNTS}. Check seed/ratios or the FASTA."
            )
    return splits


# --------------------------------------------------------------------------- #
# FASTA headers and lineage grouping
# --------------------------------------------------------------------------- #
def parse_header_fields(label: str) -> dict[str, str]:
    """Parse a pipe-delimited header into a ``{key: value}`` dict.

    Unknown or malformed segments are ignored. Keys seen here include ``idx``,
    ``genome_id``, ``lineage``, ``collection_date``, and ``isolation_country``.
    """
    fields: dict[str, str] = {}
    for segment in label.split("|"):
        if ":" not in segment:
            continue
        key, value = segment.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def raw_lineage_from_label(label: str) -> str:
    """Return the raw Pango lineage from the header, or ``"Unassigned"``."""
    return parse_header_fields(label).get("lineage", "") or "Unassigned"


# WHO variant groups, matched by exact lineage or ``<prefix>.`` descendants.
# Order matters: the first matching group wins. This is a pragmatic SARS-CoV-2
# mapping for the lineages present in this dataset and is meant to be edited if
# the data changes.
WHO_VARIANT_PREFIXES: dict[str, tuple[str, ...]] = {
    "Alpha": ("B.1.1.7", "Q"),
    "Beta": ("B.1.351",),
    "Gamma": ("P.1",),
    "Delta": ("B.1.617.2", "AY"),
    "Omicron": (
        "B.1.1.529", "BA", "BB", "BC", "BD", "BE", "BF", "BG", "BH", "BJ",
        "BK", "BL", "BM", "BN", "BP", "BQ", "BR", "BS", "BU", "BV", "BW",
        "BY", "BZ", "CH", "CM", "DV", "EG", "FL", "FY", "GK", "GW", "HK",
        "HV", "JD", "JF", "JN", "KP", "KS", "LB", "LF", "LP",
        "XBB", "XBC", "XBF", "XBK", "XDR", "XDV",
    ),
}


def _matches_prefix(lineage: str, prefix: str) -> bool:
    return lineage == prefix or lineage.startswith(prefix + ".")


def collapse_lineage(pango: str) -> str:
    """Collapse a fine-grained Pango lineage to a WHO variant group.

    Returns one of the keys of ``WHO_VARIANT_PREFIXES`` or ``"Other"``.
    """
    lineage = (pango or "").strip().upper()
    if not lineage or lineage in {"NONE", "UNASSIGNED", "UNCLASSIFIED"}:
        return "Other"
    for group, prefixes in WHO_VARIANT_PREFIXES.items():
        if any(_matches_prefix(lineage, prefix) for prefix in prefixes):
            return group
    return "Other"


def group_from_label(label: str) -> str:
    """Convenience: collapsed WHO variant group for a FASTA label."""
    return collapse_lineage(raw_lineage_from_label(label))


def fasta_labels(path: str) -> list[str]:
    """Read only the record labels (header text after ``>``) from a FASTA file."""
    labels: list[str] = []
    with open(path, encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith(">"):
                labels.append(line[1:].strip() or f"seq_{len(labels)}")
    if not labels:
        raise ValueError(f"No FASTA records found in {path}")
    return labels


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the aligned split sizes and lineage-group distribution."
    )
    parser.add_argument("--fasta", default="data/filtered_msa.fasta")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from esm_surrogate.data import protein_id_from_label

    labels = fasta_labels(args.fasta)
    splits = aligned_splits(labels, seed=args.seed)
    unique_ids = len({protein_id_from_label(label) for label in labels})
    print(f"Total records: {len(labels)} (unique IDs: {unique_ids})")
    for name in ("train_idxs", "val_idxs", "test_idxs"):
        print(f"  {name:11s}: {len(splits[name])}")

    from collections import Counter

    test_groups = Counter(group_from_label(labels[i]) for i in splits["test_idxs"])
    print("Test-fold WHO variant groups:")
    for group, count in sorted(test_groups.items(), key=lambda kv: -kv[1]):
        print(f"  {group:10s}: {count}")


if __name__ == "__main__":
    _main()
