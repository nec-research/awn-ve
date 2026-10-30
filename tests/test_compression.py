#        Adaptive-width ESM embedding distillation
#
#   File:     tests/test_compression.py
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

"""Tests for Transformer-aware compression accounting and initialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from distribution import inv_softplus
from esm_surrogate.experiment import (
    CompressionAdaptiveWidthExperiment,
)
from esm_surrogate.metrics import (
    CompressionAdaptiveWidthMSELoss,
    FLOPsCompressionAdaptiveWidthMSELoss,
)
from esm_surrogate.model import AdaptiveESM


def _model_config(*, num_layers: int = 2, **overrides: object) -> dict:
    config = {
        "model": "esm_surrogate.model.AdaptiveESM",
        "device": "cpu",
        "architecture": "esm2",
        "num_layers": num_layers,
        "embed_dim": 16,
        "attention_heads": 4,
        "token_dropout": False,
        "quantile": 0.9,
        "n_observations": 10,
        "truncated_distribution": {
            "class_name": "distribution.TruncatedDistribution",
            "args": {
                "discretized_distribution": {
                    "class_name": "distribution.DiscretizedDistribution",
                    "args": {
                        "base_distribution": {
                            "class_name": "distribution.Exponential",
                            "args": {"starting_size": 4, "max_size": 12},
                        }
                    },
                }
            },
        },
    }
    config.update(overrides)
    return config


def test_soft_parameter_count_matches_materialized_transformer() -> None:
    """Soft widths should count adaptive Q/K and FFNs plus every fixed parameter."""
    model = AdaptiveESM(8, 16, _model_config())

    soft_count = model.soft_parameter_count()
    discrete_count = sum(parameter.numel() for parameter in model.parameters())

    assert soft_count.item() == pytest.approx(discrete_count)
    soft_count.backward()
    distributions = (
        *model.variational_widths,
        *model.attention_variational_widths,
    )
    assert all(
        distribution.discretized_distribution.base_distribution._rate.grad
        is not None
        for distribution in distributions
    )


def test_forward_exposes_configured_differentiable_budget() -> None:
    """Initial soft ratio and penalty should follow the configured budget."""
    model = AdaptiveESM(
        8,
        16,
        _model_config(
            parameter_compression_target_ratio=0.5,
            parameter_compression_lambda=2.0,
        ),
    )
    converter = model.backbone.alphabet.get_batch_converter(8)
    _, _, tokens = converter([("protein", "ACDE")])

    _, diagnostics, _, _ = model(tokens)

    assert diagnostics["parameter_compression_ratio"].item() == pytest.approx(1.0)
    assert diagnostics["parameter_compression_penalty"].item() == pytest.approx(
        0.5
    )
    assert diagnostics["soft_total_parameters"].requires_grad
    assert diagnostics["soft_flops"].requires_grad


def test_forward_exposes_configured_differentiable_flops_budget() -> None:
    """Initial soft FLOPs ratio and penalty should follow their own budget."""
    model = AdaptiveESM(
        8,
        16,
        _model_config(
            parameter_compression_target_ratio=0.5,
            parameter_compression_lambda=2.0,
        ),
    )
    converter = model.backbone.alphabet.get_batch_converter(8)
    _, _, tokens = converter([("protein", "ACDE")])

    _, diagnostics, _, _ = model(tokens)

    assert diagnostics["flops_compression_ratio"].item() == pytest.approx(1.0)
    assert diagnostics["flops_compression_penalty"].item() == pytest.approx(0.5)
    assert diagnostics["parameter_compression_penalty"].item() == pytest.approx(
        0.5
    )


def test_soft_flops_uses_transformer_attention_and_ffn_geometry() -> None:
    """The estimate should include adaptive QK/FFN and fixed attention work."""
    model = AdaptiveESM(8, 16, _model_config())
    sequence_length = 7
    embed_dim = 16
    heads = 4
    mlp_width = 4
    attention_head_width = 4
    expected_per_layer = (
        4 * sequence_length * embed_dim * mlp_width
        + 4
        * sequence_length
        * embed_dim
        * heads
        * attention_head_width
        + 2
        * sequence_length**2
        * heads
        * attention_head_width
        + 4 * sequence_length * embed_dim**2
        + 2 * sequence_length**2 * embed_dim
    )

    estimate = model.soft_flops(sequence_length)

    assert estimate.item() == pytest.approx(2 * expected_per_layer)
    estimate.backward()
    assert all(
        distribution.discretized_distribution.base_distribution._rate.grad
        is not None
        for distribution in (
            *model.variational_widths,
            *model.attention_variational_widths,
        )
    )


def test_compression_loss_adds_soft_parameter_budget_penalty() -> None:
    """The compression objective should preserve MSE ELBO and add its side loss."""
    predictions = torch.tensor([[1.0, 3.0]], requires_grad=True)
    targets = torch.zeros_like(predictions)
    penalty = torch.tensor([0.25], requires_grad=True)
    loss = CompressionAdaptiveWidthMSELoss(
        use_as_loss=True,
        reduction="mean",
        accumulate_over_epoch=False,
    )

    result = loss(
        targets,
        predictions,
        {
            "n_observations": 5,
            "parameter_compression_penalty": penalty,
        },
        torch.zeros(1),
        torch.zeros(1),
    )[loss.name]

    assert result.item() == pytest.approx(5.25)
    result.backward()
    assert penalty.grad.item() == pytest.approx(1.0)


def test_flops_compression_loss_adds_only_soft_flops_budget_penalty() -> None:
    """The alternative objective should optimize the FLOPs penalty."""
    predictions = torch.tensor([[1.0, 3.0]], requires_grad=True)
    targets = torch.zeros_like(predictions)
    flops_penalty = torch.tensor([0.75], requires_grad=True)
    parameter_penalty = torch.tensor([5.0], requires_grad=True)
    loss = FLOPsCompressionAdaptiveWidthMSELoss(
        use_as_loss=True,
        reduction="mean",
        accumulate_over_epoch=False,
    )

    result = loss(
        targets,
        predictions,
        {
            "n_observations": 5,
            "flops_compression_penalty": flops_penalty,
            "parameter_compression_penalty": parameter_penalty,
        },
        torch.zeros(1),
        torch.zeros(1),
    )[loss.name]

    assert result.item() == pytest.approx(5.75)
    result.backward()
    assert flops_penalty.grad.item() == pytest.approx(1.0)
    assert parameter_penalty.grad is None


def _write_source_run(
    previous_root: Path,
    *,
    config_id: int,
    model_config: dict,
    validation_error: float,
    restored_width: int | None = None,
) -> tuple[Path, AdaptiveESM]:
    """Write the source artifacts consumed by the compression experiment."""
    run = (
        previous_root
        / "MODEL_ASSESSMENT"
        / "OUTER_FOLD_1"
        / "MODEL_SELECTION"
        / f"config_{config_id}"
        / "INNER_FOLD_1"
        / "run_1"
    )
    run.mkdir(parents=True)
    model = AdaptiveESM(8, 16, model_config)
    if restored_width is not None:
        for distribution in (
            *model.variational_widths,
            *model.attention_variational_widths,
        ):
            base_distribution = (
                distribution.discretized_distribution.base_distribution
            )
            rate = base_distribution._find_rate(restored_width, model.quantile)
            with torch.no_grad():
                base_distribution._rate.copy_(
                    inv_softplus(rate.reshape_as(base_distribution._rate))
                )
        model.change_width()
    torch.save(
        {
            "scores": {"validation_main_score": torch.tensor(validation_error)},
            "model_state": model.state_dict(),
        },
        run / "best_checkpoint.pth",
    )
    (run / "model_manifest.json").write_text(
        json.dumps({"config": model_config}),
        encoding="utf-8",
    )
    return run, model


def test_compression_experiment_loads_lowest_validation_checkpoint(
    tmp_path: Path,
) -> None:
    """Selection should recover both the winning architecture and its weights."""
    previous_root = tmp_path / "source"
    _write_source_run(
        previous_root,
        config_id=1,
        model_config=_model_config(num_layers=2),
        validation_error=0.4,
    )
    winning_run, winning_model = _write_source_run(
        previous_root,
        config_id=2,
        model_config=_model_config(num_layers=1),
        validation_error=0.2,
        restored_width=7,
    )
    current_run = (
        tmp_path
        / "compression"
        / "MODEL_ASSESSMENT"
        / "OUTER_FOLD_1"
        / "MODEL_SELECTION"
        / "config_1"
        / "INNER_FOLD_1"
        / "run_1"
    )
    compression_config = {
        "previous_experiment": str(previous_root),
        "model": "esm_surrogate.model.AdaptiveESM",
        "device": "cpu",
        "parameter_compression_target_ratio": 0.5,
        "parameter_compression_lambda": 1.0,
    }

    experiment = CompressionAdaptiveWidthExperiment(
        compression_config, str(current_run), exp_seed=3
    )
    restored = experiment.create_model(8, 16, experiment.model_config)

    assert experiment.previous_checkpoint == winning_run / "best_checkpoint.pth"
    assert experiment.previous_validation_error == pytest.approx(0.2)
    assert experiment.model_config["num_layers"] == 1
    assert restored.num_layers == 1
    expected = winning_model.state_dict()["backbone.embed_tokens.weight"]
    assert torch.equal(restored.backbone.embed_tokens.weight, expected)

    converter = restored.backbone.alphabet.get_batch_converter(8)
    _, _, tokens = converter([("protein", "ACDE")])
    _, diagnostics, _, _ = restored(tokens)
    assert diagnostics["parameter_compression_ratio"].item() == pytest.approx(1.0)
    assert diagnostics["flops_compression_ratio"].item() == pytest.approx(1.0)


def test_compression_missing_source_config_does_not_fall_back(tmp_path: Path) -> None:
    """A missing requested source must not silently select another config."""
    run, _ = _write_source_run(
        tmp_path / "source",
        config_id=1,
        model_config=_model_config(),
        validation_error=0.1,
    )
    with pytest.raises(FileNotFoundError, match="config_2"):
        CompressionAdaptiveWidthExperiment._best_validation_checkpoint(
            tmp_path / "source", str(run), previous_config_id=2
        )


@pytest.mark.parametrize("config_id", [0, -1, True, "2", 2.5])
def test_compression_rejects_invalid_source_config_id(
    tmp_path: Path, config_id: object
) -> None:
    """Only positive integer source configuration identifiers are accepted."""
    with pytest.raises(ValueError, match="previous_config_id"):
        CompressionAdaptiveWidthExperiment(
            {"previous_experiment": str(tmp_path), "previous_config_id": config_id},
            str(tmp_path),
            exp_seed=3,
        )
