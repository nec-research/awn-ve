#        Adaptive-width ESM embedding distillation
#
#   File:     tests/test_model_dynamic_attention.py
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

"""Tests for the ESM variant with adaptive query/key projections."""

from __future__ import annotations

import pytest
import torch

from distribution import inv_softplus
from esm_surrogate.model import (
    AdaptiveESM,
    DynamicAttention,
)


def _distribution(starting_size: int, max_size: int) -> dict:
    """Build one truncated exponential distribution configuration."""
    return {
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
    }


def _model_config(**overrides: object) -> dict:
    """Return a tiny valid adaptive-attention model configuration."""
    config = {
        "architecture": "esm2",
        "num_layers": 2,
        "embed_dim": 16,
        "attention_heads": 4,
        "token_dropout": False,
        "quantile": 0.9,
        "n_observations": 100,
        "truncated_distribution": _distribution(4, 12),
        "attention_truncated_distribution": _distribution(2, 4),
    }
    config.update(overrides)
    return config


def _tokens(model: AdaptiveESM) -> torch.Tensor:
    """Tokenize two short proteins with the model alphabet."""
    converter = model.backbone.alphabet.get_batch_converter(8)
    _, _, tokens = converter([("a", "ACDE"), ("b", "AAAAAA")])
    return tokens


def _set_width(distribution: torch.nn.Module, quantile: float, width: int) -> None:
    """Set a test distribution to materialize a requested quantile width."""
    base = distribution.discretized_distribution.base_distribution
    rate = base._find_rate(width, quantile)
    with torch.no_grad():
        base._rate.copy_(inv_softplus(rate.reshape_as(base._rate)))


def test_dynamic_attention_matches_native_geometry_at_full_width() -> None:
    """Head interleaving should preserve the attention computation."""
    from esm.modules import TransformerLayer

    torch.manual_seed(1)
    native = TransformerLayer(
        embed_dim=16,
        ffn_embed_dim=4,
        attention_heads=4,
        add_bias_kv=False,
        use_esm1b_layer_norm=True,
        use_rotary_embeddings=True,
    ).self_attn
    dynamic = DynamicAttention(
        native,
        initial_width=4,
        initial_importance=torch.ones(4),
    )
    with torch.no_grad():
        for native_projection, dynamic_projection in (
            (native.q_proj, dynamic.q_proj),
            (native.k_proj, dynamic.k_proj),
        ):
            native_projection.weight.copy_(
                dynamic_projection.weight.view(4, 4, 16)
                .transpose(0, 1)
                .reshape_as(native_projection.weight)
            )
            native_projection.bias.copy_(
                dynamic_projection.bias.view(4, 4)
                .transpose(0, 1)
                .reshape_as(native_projection.bias)
            )
    inputs = torch.randn(5, 2, 16)

    expected, _ = native(
        query=inputs,
        key=inputs,
        value=inputs,
        need_weights=True,
    )
    actual, _ = dynamic(
        query=inputs,
        key=inputs,
        value=inputs,
        need_weights=True,
        attention_importance=torch.ones(4),
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_query_and_value_use_probability_rescaled_initialization() -> None:
    """Q/V weight variance should follow AWN's importance-aware rule."""
    from esm.modules import TransformerLayer

    torch.manual_seed(7)
    native = TransformerLayer(
        embed_dim=128,
        ffn_embed_dim=16,
        attention_heads=4,
        add_bias_kv=False,
        use_esm1b_layer_norm=True,
        use_rotary_embeddings=False,
    ).self_attn
    native_key_weight = native.k_proj.weight.detach().clone()
    importance = torch.tensor([0.75, 0.25])

    dynamic = DynamicAttention(
        native,
        initial_width=2,
        initial_importance=importance,
    )

    expected_standard_deviation = torch.sqrt(2.0 / importance.square().sum()).item()
    assert dynamic.q_proj.weight.std().item() == pytest.approx(
        expected_standard_deviation, rel=0.1
    )
    assert dynamic.v_proj.weight.std().item() == pytest.approx(
        expected_standard_deviation, rel=0.03
    )
    expected_key_weight = (
        native_key_weight.view(4, 32, 128)
        .transpose(0, 1)
        .reshape_as(native_key_weight)[:8]
    )
    assert torch.equal(dynamic.k_proj.weight, expected_key_weight)


def test_forward_uses_dynamic_query_key_and_fixed_value_projections() -> None:
    """Query/key should share a width while value keeps the embedding size."""
    torch.manual_seed(2)
    model = AdaptiveESM(8, 16, _model_config())
    prediction, diagnostics, _, _ = model(_tokens(model))

    assert prediction.shape == (2, 16)
    assert diagnostics["mlp_widths"] == [4, 4]
    assert diagnostics["attention_widths"] == [8, 8]
    assert diagnostics["attention_head_widths"] == [2, 2]
    assert model.instantiated_mlp_widths() == [4, 4]
    assert model.instantiated_attention_widths() == [8, 8]
    assert model.instantiated_attention_head_widths() == [2, 2]
    assert len({id(item) for item in model.attention_variational_widths}) == 2
    for layer in model.backbone.layers:
        assert isinstance(layer.self_attn, DynamicAttention)
        assert layer.self_attn.q_proj.out_features == 8
        assert layer.self_attn.k_proj.out_features == 8
        assert layer.self_attn.v_proj.out_features == 16
        assert layer.self_attn.out_proj.in_features == 16


def test_dashboard_synthetic_input_can_export_dynamic_attention_graph() -> None:
    """The dynamic-attention variant should support MLWiz operator tracing."""
    model = AdaptiveESM(8, 16, _model_config(num_layers=1, token_dropout=True))
    synthetic_tokens = torch.zeros((2, 8), dtype=torch.int64)

    exported = torch.export.export(model.eval(), (synthetic_tokens,), strict=False)

    assert any(
        node.op == "call_function" and "aten." in str(node.target)
        for node in exported.graph.nodes
    )


def test_attention_width_receives_task_gradient_and_resizes_both_projections() -> None:
    """One attention distribution should train and resize both q and k."""
    torch.manual_seed(3)
    model = AdaptiveESM(8, 16, _model_config())
    attention = model.backbone.layers[0].self_attn
    old_query_weight = attention.q_proj.weight.detach().clone()
    old_key_weight = attention.k_proj.weight.detach().clone()
    fixed_value_parameter = attention.v_proj.weight

    _set_width(model.attention_variational_widths[0], model.quantile, 3)
    prediction, diagnostics, _, _ = model(_tokens(model))
    prediction.square().mean().backward()

    assert diagnostics["attention_widths"] == [12, 8]
    assert diagnostics["attention_head_widths"] == [3, 2]
    assert attention.q_proj.weight.shape == (12, 16)
    assert attention.k_proj.weight.shape == (12, 16)
    assert torch.equal(attention.q_proj.weight[:8], old_query_weight)
    assert torch.equal(attention.k_proj.weight[:8], old_key_weight)
    assert attention.v_proj.weight is fixed_value_parameter
    gradient = model.attention_variational_widths[
        0
    ].discretized_distribution.base_distribution._rate.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_attention_resize_migrates_adam_state() -> None:
    """Replacing query/key parameters should retain overlapping Adam moments."""
    torch.manual_seed(5)
    model = AdaptiveESM(8, 16, _model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.set_optimizer(optimizer)

    prediction, _, _, _ = model(_tokens(model))
    prediction.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()

    attention = model.backbone.layers[0].self_attn
    old_query = attention.q_proj.weight
    old_key = attention.k_proj.weight
    old_query_moment = optimizer.state[old_query]["exp_avg"].detach().clone()
    old_key_moment = optimizer.state[old_key]["exp_avg"].detach().clone()
    _set_width(model.attention_variational_widths[0], model.quantile, 3)
    model.change_width()

    new_query = attention.q_proj.weight
    new_key = attention.k_proj.weight
    assert old_query not in optimizer.state
    assert old_key not in optimizer.state
    assert torch.equal(optimizer.state[new_query]["exp_avg"][:8], old_query_moment)
    assert torch.equal(optimizer.state[new_key]["exp_avg"][:8], old_key_moment)


def test_checkpoint_load_restores_both_width_families() -> None:
    """Checkpoint loading should resize attention before restoring its tensors."""
    model = AdaptiveESM(8, 16, _model_config())
    _set_width(model.variational_widths[0], model.quantile, 7)
    _set_width(model.attention_variational_widths[0], model.quantile, 3)
    model.change_width()
    state = model.state_dict()

    restored = AdaptiveESM(8, 16, _model_config())
    restored.load_state_dict(state)

    assert restored.instantiated_mlp_widths() == [7, 4]
    assert restored.instantiated_attention_widths() == [12, 8]
    assert restored.instantiated_attention_head_widths() == [3, 2]
    assert torch.equal(
        restored.backbone.layers[0].self_attn.q_proj.weight,
        model.backbone.layers[0].self_attn.q_proj.weight,
    )
