#        Adaptive-width ESM embedding distillation
#
#   File:     tests/test_model.py
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

"""Tests for adaptive ESM pooling, gradients, and dynamic optimizer migration."""

from __future__ import annotations

import pytest
import torch

from distribution import inv_softplus
from esm_surrogate.data import ESMBatchCollator
from esm_surrogate.model import AdaptiveESM, DynamicMLP


def _model_config(
    *,
    num_layers: int = 2,
    starting_size: int = 4,
    max_size: int = 12,
    **overrides: object,
) -> dict:
    config = {
        "architecture": "esm2",
        "num_layers": num_layers,
        "embed_dim": 16,
        "attention_heads": 4,
        "token_dropout": False,
        "quantile": 0.9,
        "n_observations": 100,
        "truncated_distribution": {
            "class_name": "distribution.TruncatedDistribution",
            "args": {
                "discretized_distribution": {
                    "class_name": "distribution.DiscretizedDistribution",
                    "args": {
                        "base_distribution": {
                            "class_name": "distribution.Exponential",
                            "args": {
                                "starting_size": starting_size,
                                "max_size": max_size,
                            },
                        }
                    },
                }
            },
        },
    }
    config.update(overrides)
    return config


def _tokens(model: AdaptiveESM) -> torch.Tensor:
    converter = model.backbone.alphabet.get_batch_converter(8)
    _, _, tokens = converter([("a", "ACDE"), ("b", "AAAAAA")])
    return tokens


def _set_distribution_width(model: AdaptiveESM, width: int) -> None:
    for distribution in model.variational_widths:
        _set_one_distribution_width(distribution, model.quantile, width)


def _set_one_distribution_width(
    distribution: torch.nn.Module, quantile: float, width: int
) -> None:
    base = distribution.discretized_distribution.base_distribution
    rate = base._find_rate(width, quantile)
    with torch.no_grad():
        base._rate.copy_(inv_softplus(rate.reshape_as(base._rate)))


def test_forward_uses_final_backbone_embedding_and_amino_acid_pooling() -> None:
    """Output should be the mean of final non-special token representations."""
    torch.manual_seed(2)
    model = AdaptiveESM(8, 16, _model_config())
    tokens = _tokens(model)
    predicted, diagnostics, _, _ = model(tokens)
    per_token = model.backbone(tokens, model.qW_probs, model.q_attention_probs)
    expected = model._pool_amino_acids(per_token, tokens)

    assert predicted.shape == (2, 16)
    assert torch.allclose(predicted, expected)
    assert diagnostics["widths"] == [4, 4]
    assert all(
        isinstance(layer.dynamic_mlp, DynamicMLP) for layer in model.backbone.layers
    )
    assert len({id(distribution) for distribution in model.variational_widths}) == 2
    assert all(layer.dynamic_mlp.width == 4 for layer in model.backbone.layers)
    assert not hasattr(model.backbone, "lm_head")


def test_dashboard_synthetic_input_can_export_operator_graph() -> None:
    """MLWiz's data-free token input should be traceable with torch.export."""
    model = AdaptiveESM(8, 16, _model_config(num_layers=1, token_dropout=True))
    synthetic_tokens = torch.zeros((2, 8), dtype=torch.int64)

    exported = torch.export.export(model.eval(), (synthetic_tokens,), strict=False)

    assert any(
        node.op == "call_function" and "aten." in str(node.target)
        for node in exported.graph.nodes
    )


def test_empty_protein_is_still_rejected_during_eager_execution() -> None:
    """Skipping data-dependent export guards must not weaken eager validation."""
    model = AdaptiveESM(8, 16, _model_config(num_layers=1))
    special_tokens_only = torch.full((1, 4), model.backbone.cls_idx, dtype=torch.int64)

    with pytest.raises(ValueError, match="zero amino-acid tokens"):
        model(special_tokens_only)


def test_dynamic_mlp_changes_both_hidden_projection_dimensions() -> None:
    """A block MLP should resize both sides and preserve overlapping weights."""
    model = AdaptiveESM(8, 16, _model_config(num_layers=1))
    dynamic_mlp = model.backbone.layers[0].dynamic_mlp
    old_input_weight = dynamic_mlp.input_layer.weight.detach().clone()
    old_output_weight = dynamic_mlp.output_layer.weight.detach().clone()

    changed = dynamic_mlp.change_width(7)

    assert changed
    assert dynamic_mlp.width == 7
    assert dynamic_mlp.input_layer.weight.shape == (7, 16)
    assert dynamic_mlp.output_layer.weight.shape == (16, 7)
    assert torch.equal(dynamic_mlp.input_layer.weight[:4], old_input_weight)
    assert torch.equal(dynamic_mlp.output_layer.weight[:, :4], old_output_weight)

    grown_input_weight = dynamic_mlp.input_layer.weight.detach().clone()
    grown_output_weight = dynamic_mlp.output_layer.weight.detach().clone()
    assert dynamic_mlp.change_width(3)
    assert dynamic_mlp.width == 3
    assert dynamic_mlp.input_layer.weight.shape == (3, 16)
    assert dynamic_mlp.output_layer.weight.shape == (16, 3)
    assert torch.equal(dynamic_mlp.input_layer.weight, grown_input_weight[:3])
    assert torch.equal(dynamic_mlp.output_layer.weight, grown_output_weight[:, :3])


def test_dynamic_mlp_uses_probability_rescaled_output_initialization() -> None:
    """The projection consuming weighted neurons should use AWN initialization."""
    torch.manual_seed(11)
    input_layer = torch.nn.Linear(32, 4)
    output_layer = torch.nn.Linear(4, 4096)
    input_weight = input_layer.weight.detach().clone()
    neuron_importance = torch.tensor([0.55, 0.25, 0.15, 0.05])

    dynamic_mlp = DynamicMLP(
        input_layer,
        output_layer,
        activation=torch.nn.functional.gelu,
        initial_importance=neuron_importance,
    )

    expected_standard_deviation = torch.sqrt(
        2.0 / neuron_importance.square().sum()
    ).item()
    assert dynamic_mlp.output_layer.weight.std().item() == pytest.approx(
        expected_standard_deviation, rel=0.02
    )
    assert torch.equal(dynamic_mlp.input_layer.weight, input_weight)


def test_forward_calls_change_width_before_dynamic_mlp_execution() -> None:
    """Each q(width) should materialize its block width on the next forward."""
    model = AdaptiveESM(8, 16, _model_config())
    _set_one_distribution_width(model.variational_widths[0], model.quantile, 6)
    _set_one_distribution_width(model.variational_widths[1], model.quantile, 8)

    _, diagnostics, _, _ = model(_tokens(model))

    assert diagnostics["widths"] == [6, 8]
    assert model.instantiated_widths() == [6, 8]
    assert [layer.dynamic_mlp.width for layer in model.backbone.layers] == [6, 8]


def test_width_distribution_receives_task_loss_gradients() -> None:
    """Importance scaling must connect every width rate to the L1 objective."""
    torch.manual_seed(3)
    model = AdaptiveESM(8, 16, _model_config())
    prediction, _, _, _ = model(_tokens(model))
    torch.nn.functional.l1_loss(prediction, torch.zeros_like(prediction)).backward()

    gradients = [
        distribution.discretized_distribution.base_distribution._rate.grad
        for distribution in model.variational_widths
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_width_probabilities_follow_parent_module_to_cuda() -> None:
    """Cached support values should follow a parent module's device move."""
    model = AdaptiveESM(8, 16, _model_config()).cuda()
    distribution = model.variational_widths[0]

    probabilities = distribution.compute_probability_vector_known_width(4)

    assert probabilities.device.type == "cuda"
    assert distribution.x.device == probabilities.device


def test_resize_migrates_weights_and_adam_state() -> None:
    """Growing FFNs should preserve overlapping weights and optimizer moments."""
    torch.manual_seed(5)
    model = AdaptiveESM(8, 16, _model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.set_optimizer(optimizer)

    prediction, _, _, _ = model(_tokens(model))
    prediction.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()

    dynamic_mlp = model.backbone.layers[0].dynamic_mlp
    old_input_parameter = dynamic_mlp.input_layer.weight
    old_output_parameter = dynamic_mlp.output_layer.weight
    old_input_weight = old_input_parameter.detach().clone()
    old_output_weight = old_output_parameter.detach().clone()
    old_input_moment = optimizer.state[old_input_parameter]["exp_avg"].detach().clone()
    old_output_moment = (
        optimizer.state[old_output_parameter]["exp_avg"].detach().clone()
    )
    _set_distribution_width(model, 6)
    widths = model.change_width()

    new_input_parameter = dynamic_mlp.input_layer.weight
    new_output_parameter = dynamic_mlp.output_layer.weight
    assert widths == [6, 6]
    assert model.instantiated_widths() == [6, 6]
    assert new_input_parameter is not old_input_parameter
    assert new_output_parameter is not old_output_parameter
    assert old_input_parameter not in optimizer.state
    assert old_output_parameter not in optimizer.state
    assert new_input_parameter in optimizer.state
    assert new_output_parameter in optimizer.state
    assert torch.equal(new_input_parameter[:4], old_input_weight)
    assert torch.equal(new_output_parameter[:, :4], old_output_weight)
    assert torch.equal(
        optimizer.state[new_input_parameter]["exp_avg"][:4], old_input_moment
    )
    assert torch.equal(
        optimizer.state[new_output_parameter]["exp_avg"][:, :4],
        old_output_moment,
    )
    assert (
        optimizer.state[new_input_parameter]["exp_avg"].shape
        == new_input_parameter.shape
    )
    assert (
        optimizer.state[new_output_parameter]["exp_avg"].shape
        == new_output_parameter.shape
    )


def test_checkpoint_load_resizes_before_restoring_weights() -> None:
    """A checkpoint with adapted widths should reload into a fresh model."""
    model = AdaptiveESM(8, 16, _model_config())
    _set_distribution_width(model, 7)
    model.change_width()
    state = model.state_dict()

    restored = AdaptiveESM(8, 16, _model_config())
    restored.load_state_dict(state)
    assert restored.instantiated_widths() == [7, 7]
    assert torch.equal(
        restored.backbone.layers[0].dynamic_mlp.input_layer.weight,
        model.backbone.layers[0].dynamic_mlp.input_layer.weight,
    )


def test_dynamic_mlp_activation_is_resolved_from_config() -> None:
    """Every block should use the activation selected by dotted path."""
    model = AdaptiveESM(
        8,
        16,
        _model_config(activation="torch.nn.functional.relu"),
    )

    assert all(
        layer.dynamic_mlp.activation is torch.nn.functional.relu
        for layer in model.backbone.layers
    )


def test_training_set_size_must_be_configured() -> None:
    """ELBO normalization must not silently assume a dataset size."""
    config = _model_config()
    del config["n_observations"]

    with pytest.raises(ValueError, match="n_observations must be configured"):
        AdaptiveESM(8, 16, config)


def test_forward_computes_global_width_and_theta_log_priors() -> None:
    """Prior outputs should match the reference Gaussian expressions."""
    model = AdaptiveESM(
        8,
        16,
        _model_config(
            theta_prior_scale=2.0,
            total_width_prior_mean=10,
            total_width_prior_scale=0.5,
            n_observations=17,
        ),
    )

    _, diagnostics, theta_log_prior, width_log_prior = model(_tokens(model))

    expected_theta = -sum(
        parameter.square().sum() for parameter in model.backbone.parameters()
    ) / (2.0 * 2.0**2)
    rates = torch.cat(
        [
            distribution.discretized_distribution.base_distribution.parameter
            for distribution in (
                *model.variational_widths,
                *model.attention_variational_widths,
            )
        ]
    )
    base_distribution = model.variational_widths[
        0
    ].discretized_distribution.base_distribution
    global_mean = base_distribution._find_rate(
        desired_size=10,
        quantile=model.quantile,
    )
    rate_at_one_width_scale = base_distribution._find_rate(
        desired_size=10.5,
        quantile=model.quantile,
    )
    global_scale = (global_mean - rate_at_one_width_scale).abs()
    expected_width = -torch.log(global_scale) - (rates.sum() - global_mean).square() / (
        2.0 * global_scale.square()
    )

    assert diagnostics["n_observations"] == 17
    assert model.total_width_prior_mean == 10
    assert torch.allclose(model.total_width_prior_rate_mean, global_mean.reshape(1))
    assert torch.allclose(model.total_width_prior_scale, global_scale.reshape(1))
    assert torch.allclose(theta_log_prior, expected_theta.reshape(1))
    assert torch.allclose(width_log_prior, expected_width.reshape(1))
    assert theta_log_prior.requires_grad
    assert width_log_prior.requires_grad


def test_collator_uses_fair_esm_batch_converter() -> None:
    """The training loader should produce native fair-esm BOS/EOS padding."""
    collator = ESMBatchCollator(max_sequence_length=3)
    tokens, targets = collator(
        [
            (("one", "ACDE"), torch.ones(16)),
            (("two", "M"), torch.zeros(16)),
        ]
    )
    assert tokens.shape == (2, 5)  # BOS + three residues + EOS
    assert targets.shape == (2, 16)
