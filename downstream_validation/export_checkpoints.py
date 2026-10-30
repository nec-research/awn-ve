#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/export_checkpoints.py
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

"""One-time conversion of MLWiz terminal checkpoints to static AWN models.

Run from the repository root:
    .venv/bin/python -m downstream_validation.export_checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from esm_surrogate.data import ProteinEmbeddingDataset
from esm_surrogate.model import AdaptiveESM
from mlwiz.data.splitter import Splitter
from mlwiz.data.util import load_dataset

from .model import AdaptiveESMInference


DEFAULT_RESULTS_ROOT = Path("RESULTS")
EXPERIMENT = "esm_distillation_compression_mean_centered_targets_ProteinEmbeddingDataset"


def _results_root(value: Path | None) -> Path:
    root = (value if value is not None else DEFAULT_RESULTS_ROOT).expanduser()
    if not root.is_dir():
        raise FileNotFoundError("Set --results-root to the phase-2 RESULTS directory.")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_mean(manifest: dict[str, Any], repository_root: Path) -> torch.Tensor:
    config = manifest["config"]
    storage_folder = Path(config["storage_folder"])
    if not storage_folder.is_absolute():
        storage_folder = repository_root / storage_folder
    dataset = load_dataset(str(storage_folder), ProteinEmbeddingDataset)
    splits = Splitter.load(str(repository_root / "DATA_SPLITS/ProteinEmbeddingDataset/ProteinEmbeddingDataset_outer1_inner1.splits"))
    dataset.set_target_normalization(splits.inner_folds[0][0].train_idxs, mode="center")
    return dataset.target_mean.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _static_state(model: AdaptiveESM, metadata: dict[str, Any]) -> dict[str, torch.Tensor]:
    static = AdaptiveESMInference(metadata)
    backbone_state = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in model.state_dict().items()
        if key.startswith("backbone.")
    }
    incompatible = static.load_state_dict(backbone_state, strict=False)
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    expected_missing = {
        f"backbone.layers.{index}.{kind}.importance"
        for index in range(metadata["num_layers"])
        for kind in ("dynamic_mlp", "self_attn")
    }
    if unexpected or missing != expected_missing:
        raise RuntimeError(f"Static state conversion mismatch; missing={missing}, unexpected={unexpected}")
    mlp_probabilities = model._compute_width_probabilities(model.instantiated_mlp_widths())
    attention_probabilities = model._compute_attention_probabilities(model.instantiated_attention_head_widths())
    for layer, mlp_probability, attention_probability in zip(
        static.backbone.layers, mlp_probabilities, attention_probabilities
    ):
        layer.dynamic_mlp.importance.copy_(mlp_probability.detach().cpu())
        layer.self_attn.importance.copy_(attention_probability.detach().cpu())
    return static.state_dict()


def export_checkpoint(run_directory: Path, output_root: Path, repository_root: Path) -> dict[str, Any]:
    manifest = json.loads((run_directory / "model_manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = run_directory / "last_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = AdaptiveESM(manifest["dim_input_features"], manifest["dim_target"], manifest["config"])
    source.load_state_dict(checkpoint["model_state"], strict=True)
    mlp_widths = source.instantiated_mlp_widths()
    attention_head_widths = source.instantiated_attention_head_widths()
    config_name = next(parent.name for parent in run_directory.parents if parent.name.startswith("config_"))
    results_root = next(parent for parent in run_directory.parents if parent.name == EXPERIMENT).parent
    metadata = {
        "format_version": 1,
        "model_id": config_name,
        "architecture": source.architecture,
        "num_layers": source.num_layers,
        "embed_dim": source.embed_dim,
        "attention_heads": source.attention_heads,
        "token_dropout": source.backbone.token_dropout,
        "activation": "relu6",
        "max_sequence_length": int(manifest["dim_input_features"]),
        "mlp_widths": mlp_widths,
        "attention_head_widths": attention_head_widths,
        "target_normalization": "center",
        "source_checkpoint": str(Path("RESULTS") / checkpoint_path.relative_to(results_root)),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_epoch": int(checkpoint["epoch"]) + 1,
        "parameter_compression_target_ratio": float(manifest["config"]["parameter_compression_target_ratio"]),
        "validation_main_score": float(checkpoint["scores"]["validation_main_score"]),
        "materialized_parameters": sum(parameter.numel() for parameter in source.parameters()),
    }
    output_dir = output_root / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_static_state(source, metadata), output_dir / "model.pt")
    torch.save(_target_mean(manifest, repository_root), output_dir / "target_mean.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export terminal phase-2 selection checkpoints for standalone inference.")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "models")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    run_root = _results_root(args.results_root) / EXPERIMENT / "MODEL_ASSESSMENT/OUTER_FOLD_1/MODEL_SELECTION"
    runs = sorted(run_root.glob("config_*/INNER_FOLD_1/run_1"))
    if len(runs) != 3:
        raise RuntimeError(f"Expected exactly three model-selection runs, found {len(runs)} under {run_root}")
    exports = [export_checkpoint(run, args.output_root, repository_root) for run in runs]
    (args.output_root / "manifest.json").write_text(json.dumps({"exports": exports}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported {len(exports)} static models to {args.output_root}")


if __name__ == "__main__":
    main()
