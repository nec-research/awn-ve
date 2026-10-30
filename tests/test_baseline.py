#        Adaptive-width ESM embedding distillation
#
#   File:     tests/test_baseline.py
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

"""Tests for the fixed-depth adaptive MLP baseline."""

from __future__ import annotations

import pytest
import torch

from distribution import inv_softplus
from esm_surrogate.baseline import DynamicMLPBaseline


def _baseline_config(
    *,
    num_hidden_layers: int = 3,
    starting_size: int = 4,
    token_embedding_dim: int = 3,
) -> dict[str, object]:
    return {
        "num_hidden_layers": num_hidden_layers,
        "token_embedding_dim": token_embedding_dim,
        "activation": "torch.nn.functional.relu",
        "quantile": 0.9,
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
                                "max_size": 12,
                            },
                        }
                    },
                }
            },
        },
    }


def _set_width(distribution: torch.nn.Module, quantile: float, width: int) -> None:
    base = distribution.discretized_distribution.base_distribution
    rate = base._find_rate(width, quantile)
    with torch.no_grad():
        base._rate.copy_(inv_softplus(rate.reshape_as(base._rate)))


def test_forward_embeds_flattens_and_pads_tokens_before_prediction() -> None:
    """Variable batch lengths should map to one fixed-size embedded MLP input."""
    model = DynamicMLPBaseline(5, 6, _baseline_config())
    tokens = torch.tensor([[0, 5, 6, 2], [0, 7, 2, 1]])

    flattened = model.flatten_tokens(tokens)
    prediction, final_hidden = model(tokens)

    assert model.max_token_length == 7  # five amino acids plus BOS and EOS
    assert model.token_embedding_dim == 3
    assert flattened.shape == (2, 21)
    embedded_tokens = flattened.reshape(2, 7, 3)
    assert torch.equal(embedded_tokens[:, :4], model.token_embedding(tokens))
    assert torch.count_nonzero(embedded_tokens[:, 4:]) == 0
    assert prediction.shape == (2, 6)
    assert final_hidden.shape == (2, 4)
    assert len(model.layers) == 4
    assert model.instantiated_widths() == [4, 4, 4]


def test_widths_adapt_without_changing_the_configured_depth() -> None:
    """Every hidden width should resize both adjacent projections."""
    model = DynamicMLPBaseline(5, 6, _baseline_config())
    requested_widths = [5, 6, 7]
    for distribution, width in zip(model.variational_widths, requested_widths):
        _set_width(distribution, model.quantile, width)

    widths = model.change_width()

    assert widths == requested_widths
    assert model.instantiated_widths() == requested_widths
    assert len(model.layers) == model.num_hidden_layers + 1
    assert [(layer.in_features, layer.out_features) for layer in model.layers] == [
        (21, 5),
        (5, 6),
        (6, 7),
        (7, 6),
    ]


def test_task_loss_trains_each_hidden_width_distribution() -> None:
    """Direct embedding error should backpropagate into every learned width."""
    torch.manual_seed(7)
    model = DynamicMLPBaseline(5, 6, _baseline_config())
    tokens = torch.tensor([[0, 5, 6, 2], [0, 7, 8, 2]])

    prediction, _ = model(tokens)
    torch.nn.functional.l1_loss(prediction, torch.zeros_like(prediction)).backward()

    gradients = [
        distribution.discretized_distribution.base_distribution._rate.grad
        for distribution in model.variational_widths
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()
    assert torch.count_nonzero(model.token_embedding.weight.grad) > 0


def test_resize_migrates_adam_state_across_fixed_hidden_layers() -> None:
    """Changing widths should retain overlapping weights and optimizer moments."""
    torch.manual_seed(13)
    model = DynamicMLPBaseline(5, 6, _baseline_config(num_hidden_layers=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.set_optimizer(optimizer)
    tokens = torch.tensor([[0, 5, 6, 2], [0, 7, 8, 2]])
    prediction, _ = model(tokens)
    prediction.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()

    old_parameter = model.layers[1].weight
    old_weight = old_parameter.detach().clone()
    old_moment = optimizer.state[old_parameter]["exp_avg"].detach().clone()
    for distribution, width in zip(model.variational_widths, (5, 6)):
        _set_width(distribution, model.quantile, width)

    model.change_width()

    new_parameter = model.layers[1].weight
    assert new_parameter is not old_parameter
    assert old_parameter not in optimizer.state
    assert new_parameter in optimizer.state
    assert new_parameter.shape == (6, 5)
    assert torch.equal(new_parameter[:4, :4], old_weight)
    assert torch.equal(optimizer.state[new_parameter]["exp_avg"][:4, :4], old_moment)


def test_sequences_longer_than_the_configured_dataset_limit_are_rejected() -> None:
    """The fixed first projection should fail clearly on oversized inputs."""
    model = DynamicMLPBaseline(3, 2, _baseline_config(num_hidden_layers=1))

    with pytest.raises(ValueError, match="exceeds the configured maximum"):
        model(torch.ones((2, 6), dtype=torch.long))


def test_token_embedding_dimension_must_be_positive() -> None:
    """Invalid learned token-vector dimensions should fail during construction."""
    with pytest.raises(ValueError, match="token_embedding_dim must be positive"):
        DynamicMLPBaseline(3, 2, _baseline_config(token_embedding_dim=0))
