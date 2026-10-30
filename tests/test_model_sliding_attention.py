#        Adaptive-width ESM embedding distillation
#
#   File:     tests/test_model_sliding_attention.py
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

"""Tests for checkpoint-compatible adaptive sliding-window attention."""

from __future__ import annotations

import pytest
import torch

from distribution import inv_softplus
from esm_surrogate.model import AdaptiveESM as DenseAdaptiveESM
from esm_surrogate.model_sliding_attention import (
    AdaptiveESM as SlidingWindowAdaptiveESM,
)
from esm_surrogate.model_sliding_attention import (
    SlidingWindowDynamicAttention,
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
        "attention_window_radius": 2,
        "attention_block_size": 16,
    }
    config.update(overrides)
    return config


def _tokens(model: DenseAdaptiveESM) -> torch.Tensor:
    """Tokenize two proteins, including padding in the shorter sequence."""
    converter = model.backbone.alphabet.get_batch_converter(8)
    _, _, tokens = converter([("a", "ACDE"), ("b", "AAAAAA")])
    return tokens


def _set_width(distribution: torch.nn.Module, quantile: float, width: int) -> None:
    """Set a test distribution to materialize a requested quantile width."""
    base = distribution.discretized_distribution.base_distribution
    rate = base._find_rate(width, quantile)
    with torch.no_grad():
        base._rate.copy_(inv_softplus(rate.reshape_as(base._rate)))


def test_dense_checkpoint_loads_strictly_and_full_window_matches() -> None:
    """A dense checkpoint should retain its keys and outputs under a full window."""
    torch.manual_seed(1)
    dense = DenseAdaptiveESM(8, 16, _model_config()).eval()
    sliding = SlidingWindowAdaptiveESM(
        8,
        16,
        _model_config(attention_window_radius=32),
    ).eval()
    dense_state = dense.state_dict()

    sliding.load_state_dict(dense_state, strict=True)
    tokens = _tokens(dense)
    dense_prediction = dense(tokens)[0]
    sliding_prediction = sliding(tokens)[0]

    assert dense_state.keys() == sliding.state_dict().keys()
    assert torch.allclose(sliding_prediction, dense_prediction, atol=1e-6)


def test_local_cpu_attention_preserves_padding_and_returns_requested_weights() -> None:
    """The correctness fallback should combine local and padding masks."""
    torch.manual_seed(2)
    model = SlidingWindowAdaptiveESM(8, 16, _model_config()).eval()
    attention = model.backbone.layers[0].self_attn
    hidden = torch.randn(8, 2, 16)
    padding_mask = torch.zeros(2, 8, dtype=torch.bool)
    padding_mask[0, 6:] = True
    importance = torch.ones(attention.head_width)

    attended, weights = attention(
        hidden,
        hidden,
        hidden,
        key_padding_mask=padding_mask,
        need_weights=True,
        attention_importance=importance,
    )

    assert attended.shape == hidden.shape
    assert weights is not None
    assert weights.shape == (2, 8, 8)
    query_positions = torch.arange(8)
    key_positions = torch.arange(8)
    outside_window = (
        query_positions[:, None] - key_positions[None, :]
    ).abs() > attention.window_radius
    assert torch.count_nonzero(weights.masked_select(outside_window[None])) == 0
    assert torch.count_nonzero(weights[0, :, 6:]) == 0


def test_long_padding_tail_does_not_create_nonfinite_pooled_outputs() -> None:
    """Fully masked padded query rows must remain harmless to final pooling."""
    torch.manual_seed(5)
    model = SlidingWindowAdaptiveESM(
        24,
        16,
        _model_config(attention_window_radius=1),
    ).eval()
    converter = model.backbone.alphabet.get_batch_converter(24)
    _, _, single_tokens = converter([("short", "A")])
    _, _, batched_tokens = converter(
        [
            ("short", "A"),
            ("long", "ACDEFGHIKLMNPQRSTVWY"),
        ]
    )

    single_prediction = model(single_tokens)[0][0]
    batched_prediction = model(batched_tokens)[0][0]

    assert torch.isfinite(batched_prediction).all()
    assert torch.allclose(batched_prediction, single_prediction, atol=1e-6)


def test_adaptive_query_key_width_still_resizes_and_receives_gradient() -> None:
    """Sliding attention should retain dynamic projection and rate gradients."""
    torch.manual_seed(3)
    model = SlidingWindowAdaptiveESM(8, 16, _model_config())
    attention = model.backbone.layers[0].self_attn
    old_query_weight = attention.q_proj.weight.detach().clone()

    _set_width(model.attention_variational_widths[0], model.quantile, 3)
    prediction, diagnostics, _, _ = model(_tokens(model))
    prediction.square().mean().backward()

    assert diagnostics["attention_head_widths"] == [3, 2]
    assert attention.q_proj.weight.shape == (12, 16)
    assert torch.equal(attention.q_proj.weight[:8], old_query_weight)
    gradient = model.attention_variational_widths[
        0
    ].discretized_distribution.base_distribution._rate.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_sliding_model_remains_exportable_on_cpu() -> None:
    """Dashboard operator tracing should retain its CPU correctness path."""
    model = SlidingWindowAdaptiveESM(
        8,
        16,
        _model_config(num_layers=1, token_dropout=True),
    ).eval()
    synthetic_tokens = torch.zeros((2, 8), dtype=torch.int64)

    exported = torch.export.export(model, (synthetic_tokens,), strict=False)

    assert any(
        node.op == "call_function" and "aten." in str(node.target)
        for node in exported.graph.nodes
    )


def test_invalid_window_configuration_is_rejected() -> None:
    """Window radii and block sizes should fail early with clear errors."""
    with pytest.raises(ValueError, match="non-negative"):
        SlidingWindowAdaptiveESM(
            8,
            16,
            _model_config(attention_window_radius=-1),
        )
    with pytest.raises(ValueError, match="power of two"):
        SlidingWindowAdaptiveESM(
            8,
            16,
            _model_config(attention_block_size=24),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flex_cuda_matches_dense_local_attention_with_tiny_adaptive_heads() -> None:
    """Compiled FlexAttention should match the exact local reference path."""
    torch.manual_seed(4)
    model = (
        SlidingWindowAdaptiveESM(
            8,
            16,
            _model_config(num_layers=1, attention_block_size=64),
        )
        .eval()
        .cuda()
    )
    attention = model.backbone.layers[0].self_attn
    assert isinstance(attention, SlidingWindowDynamicAttention)
    hidden = torch.randn(8, 2, 16, device="cuda")
    padding_mask = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
    padding_mask[0, 3:] = True
    importance = torch.ones(attention.head_width, device="cuda")

    with torch.no_grad():
        expected, _ = attention._dense_window_forward(
            hidden,
            hidden,
            hidden,
            key_padding_mask=padding_mask,
            need_weights=False,
            need_head_weights=False,
            attn_mask=None,
            attention_importance=importance,
        )
        actual, returned_weights = attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=padding_mask,
            need_weights=False,
            attention_importance=importance,
        )

    assert returned_weights is None
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)

    differentiable_importance = torch.ones(
        attention.head_width,
        device="cuda",
        requires_grad=True,
    )
    differentiable_output, _ = attention(
        hidden,
        hidden,
        hidden,
        key_padding_mask=padding_mask,
        need_weights=False,
        attention_importance=differentiable_importance,
    )
    differentiable_output.square().mean().backward()

    assert differentiable_importance.grad is not None
    assert torch.isfinite(differentiable_importance.grad).all()
    assert attention.q_proj.weight.grad is not None
    assert torch.isfinite(attention.q_proj.weight.grad).all()

    # Exercise the 128-wide adaptive bucket that previously let Inductor pick
    # a kernel tile larger than a 64-token sparse-mask block.
    attention.change_width(128)
    wide_importance = torch.ones(128, device="cuda")
    with torch.no_grad():
        wide_expected, _ = attention._dense_window_forward(
            hidden,
            hidden,
            hidden,
            key_padding_mask=padding_mask,
            need_weights=False,
            need_head_weights=False,
            attn_mask=None,
            attention_importance=wide_importance,
        )
        wide_actual, _ = attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=padding_mask,
            need_weights=False,
            attention_importance=wide_importance,
        )

    assert torch.allclose(wide_actual, wide_expected, atol=2e-5, rtol=2e-5)
