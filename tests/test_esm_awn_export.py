#        Adaptive-width ESM embedding distillation
#
#   File:     tests/test_esm_awn_export.py
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

"""Regression tests for the standalone terminal-checkpoint exports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from downstream_validation.loader import load_model


RESULTS_ROOT = Path(os.environ.get("ESM_RESULTS_ROOT", "RESULTS")).expanduser()
RUN_ROOT = RESULTS_ROOT / "esm_distillation_compression_mean_centered_targets_ProteinEmbeddingDataset/MODEL_ASSESSMENT/OUTER_FOLD_1/MODEL_SELECTION"
MODELS_ROOT = Path("downstream_validation/models")


def _has_export(model_id: str) -> bool:
    """Whether all artifacts for an exported model are available locally."""
    return all((MODELS_ROOT / model_id / name).is_file() for name in ("metadata.json", "model.pt", "target_mean.pt"))


def _runs() -> list[Path]:
    return sorted(RUN_ROOT.glob("config_*/INNER_FOLD_1/run_1"))


@pytest.mark.skipif(not RUN_ROOT.is_dir(), reason="Phase-2 checkpoint tree is unavailable")
def test_static_exports_match_all_terminal_selection_checkpoints() -> None:
    """All three package models reproduce their source centered embeddings exactly."""
    from esm_surrogate.model import AdaptiveESM

    runs = _runs()
    assert len(runs) == 3
    for run in runs:
        model_id = run.parents[1].name
        manifest = json.loads((run / "model_manifest.json").read_text(encoding="utf-8"))
        source = AdaptiveESM(manifest["dim_input_features"], manifest["dim_target"], manifest["config"]).eval()
        checkpoint = torch.load(run / "last_checkpoint.pth", map_location="cpu", weights_only=False)
        source.load_state_dict(checkpoint["model_state"], strict=True)
        exported, metadata, target_mean = load_model(model_id)
        _, _, tokens = source.backbone.alphabet.get_batch_converter(20)([
            ("alpha", "ACDEFGHIK"),
            ("beta", "MSTNPKPQR"),
        ])
        with torch.inference_mode():
            source_centered = source(tokens)[0]
            exported_centered = exported(tokens)
        assert torch.equal(exported_centered, source_centered)
        assert torch.equal(exported_centered + target_mean, source_centered + target_mean)
        assert metadata["source_epoch"] == 25


@pytest.mark.skipif(not all(_has_export(f"config_{i}") for i in (1, 2, 3)), reason="Exported model artifacts are unavailable")
def test_export_state_contains_no_training_width_state() -> None:
    """Static packages load strictly and expose neither MLWiz nor distributions."""
    package_root = Path("downstream_validation")
    assert "mlwiz" not in (package_root / "model.py").read_text(encoding="utf-8")
    for model_dir in sorted((package_root / "models").glob("config_*")):
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        assert not any("variational_width" in key for key in state)
        model, _, _ = load_model(model_dir)
        assert isinstance(model, torch.nn.Module)


@pytest.mark.skipif(not _has_export("config_3"), reason="config_3 exported model artifacts are unavailable")
def test_fasta_cli_writes_raw_wrapper_and_skips_existing(tmp_path: Path) -> None:
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">first protein\nACDEFG\n>second\nMSTNPKPQR\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    command = [
        sys.executable,
        "-m",
        "downstream_validation.encode_proteins_from_fasta",
        "--input",
        str(fasta),
        "--output-dir",
        str(output_dir),
        "--model",
        "config_3",
        "--gpu",
        "-1",
        "--token-budget",
        "32",
    ]
    environment = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    subprocess.run(command, check=True, cwd=Path.cwd(), env=environment)
    first = torch.load(output_dir / "firstprotein.pt", map_location="cpu", weights_only=True)
    second = torch.load(output_dir / "second.pt", map_location="cpu", weights_only=True)
    assert first["label"] == "first protein"
    assert first["encodings"].shape == (1280,)
    assert second["encodings"].shape == (1280,)
    before = (output_dir / "firstprotein.pt").stat().st_mtime_ns
    subprocess.run([*command, "--skip-existing"], check=True, cwd=Path.cwd(), env=environment)
    assert (output_dir / "firstprotein.pt").stat().st_mtime_ns == before
