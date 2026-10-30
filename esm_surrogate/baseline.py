#        Adaptive-width ESM embedding distillation
#
#   File:     esm_surrogate/baseline.py
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

"""Adaptive-width MLP baseline for direct protein-embedding regression."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from mlwiz.model.interface import ModelInterface
from mlwiz.util import return_class_and_args
from torch import nn

from distribution import TruncatedDistribution
from dynamic_optimizers import update_optimizer
from esm_surrogate._model_base import (
    _copy_linear_overlap,
    _rescaled_weight_init,
    _resolve_activation,
)


class DynamicMLPBaseline(ModelInterface):
    """Embed and flatten token sequences for fixed-depth MLP regression.

    ``num_hidden_layers`` fixes the depth for the lifetime of the model. Each
    hidden layer owns an independent adaptive-width distribution, so only the
    number of neurons changes during training. Token IDs emitted by
    :class:`esm_surrogate.data.ESMBatchCollator` pass through a learned embedding;
    shorter batches are padded to the dataset's configured maximum sequence
    length before the token embeddings are flattened.
    """

    def __init__(
        self, dim_input_features: int, dim_target: int, config: Mapping[str, Any]
    ) -> None:
        config = dict(config)
        super().__init__(dim_input_features, dim_target, config)
        if not isinstance(dim_input_features, int):
            raise TypeError("DynamicMLPBaseline requires an integer sequence length.")
        if dim_input_features < 1:
            raise ValueError("dim_input_features must be positive.")

        self.num_hidden_layers = int(config.get("num_hidden_layers", 2))
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be positive.")
        self.token_embedding_dim = int(config.get("token_embedding_dim", 16))
        if self.token_embedding_dim < 1:
            raise ValueError("token_embedding_dim must be positive.")
        self.quantile = float(config.get("quantile", 0.9))
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0, 1).")

        try:
            import esm
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "fair-esm is required to construct DynamicMLPBaseline."
            ) from error
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.padding_idx = int(alphabet.padding_idx)
        self.max_token_length = (
            int(dim_input_features)
            + int(alphabet.prepend_bos)
            + int(alphabet.append_eos)
        )
        self.token_embedding = nn.Embedding(
            len(alphabet),
            self.token_embedding_dim,
            padding_idx=self.padding_idx,
        )

        distribution_class, distribution_args = return_class_and_args(
            config, "truncated_distribution"
        )
        if distribution_class is not TruncatedDistribution:
            raise TypeError(
                "DynamicMLPBaseline expects distribution.TruncatedDistribution."
            )
        self.variational_widths = nn.ModuleList(
            [
                distribution_class(
                    truncation_quantile=self.quantile, **distribution_args
                )
                for _ in range(self.num_hidden_layers)
            ]
        )
        initial_widths = self.current_widths()
        initial_probabilities = self._compute_width_probabilities(initial_widths)

        self.activation = _resolve_activation(
            config.get("activation", "torch.nn.functional.relu6")
        )
        has_bias = bool(config.get("has_bias", True))
        flattened_embedding_dim = self.max_token_length * self.token_embedding_dim
        layer_dimensions = [flattened_embedding_dim, *initial_widths, int(dim_target)]
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_width, output_width, bias=has_bias)
                for input_width, output_width in zip(
                    layer_dimensions[:-1], layer_dimensions[1:]
                )
            ]
        )
        # Every projection after the first consumes probability-weighted
        # neurons and therefore uses the AWN probability-rescaled initializer.
        for layer, probabilities in zip(self.layers[1:], initial_probabilities):
            _rescaled_weight_init(layer.weight, probabilities)

        self.torch_optimizer: torch.optim.Optimizer | None = None
        self.old_parameters = dict(self.named_parameters())

    def set_optimizer(self, optimizer: object) -> None:
        """Attach MLWiz's optimizer so resized parameters retain their state."""
        self.torch_optimizer = getattr(optimizer, "optimizer", optimizer)

    def current_widths(self) -> list[int]:
        """Return hidden widths implied by the learned distributions."""
        return [
            max(int(distribution.compute_truncation_number()), 1)
            for distribution in self.variational_widths
        ]

    def instantiated_widths(self) -> list[int]:
        """Return the materialized width of each fixed hidden layer."""
        widths = [int(layer.out_features) for layer in self.layers[:-1]]
        for layer_id, width in enumerate(widths):
            if self.layers[layer_id + 1].in_features != width:
                raise RuntimeError(
                    f"Hidden layer {layer_id} has inconsistent adjacent shapes."
                )
        return widths

    def _replace_linear(
        self,
        layer_id: int,
        *,
        in_features: int | None = None,
        out_features: int | None = None,
    ) -> bool:
        """Resize one linear projection while retaining overlapping parameters."""
        layer = self.layers[layer_id]
        new_in_features = layer.in_features if in_features is None else in_features
        new_out_features = layer.out_features if out_features is None else out_features
        if (
            new_in_features == layer.in_features
            and new_out_features == layer.out_features
        ):
            return False

        replacement = nn.Linear(
            new_in_features,
            new_out_features,
            bias=layer.bias is not None,
        ).to(device=layer.weight.device, dtype=layer.weight.dtype)
        replacement.weight.requires_grad_(layer.weight.requires_grad)
        if replacement.bias is not None and layer.bias is not None:
            replacement.bias.requires_grad_(layer.bias.requires_grad)
        _copy_linear_overlap(layer, replacement)
        self.layers[layer_id] = replacement
        return True

    def change_width(self) -> list[int]:
        """Materialize every adaptive hidden width and migrate optimizer state."""
        widths = self.current_widths()
        changed = False
        for layer_id, width in enumerate(widths):
            changed = self._replace_linear(layer_id, out_features=width) or changed
            changed = self._replace_linear(layer_id + 1, in_features=width) or changed

        if changed:
            new_parameters = dict(self.named_parameters())
            if self.torch_optimizer is not None:
                update_optimizer(
                    self.torch_optimizer,
                    new_parameters,
                    optimized_params=self.old_parameters,
                    reset_state=False,
                    remove_params=True,
                )
            self.old_parameters = new_parameters
        return widths

    def _compute_width_probabilities(self, widths: Sequence[int]) -> list[torch.Tensor]:
        """Compute differentiable neuron importances at materialized widths."""
        probabilities = [
            distribution.compute_probability_vector_known_width(width)
            for distribution, width in zip(self.variational_widths, widths)
        ]
        self.qW_probs = probabilities
        return probabilities

    def flatten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Right-pad, embed, and flatten tokens to the fixed MLP input size."""
        if tokens.ndim < 2:
            raise ValueError(
                "Expected tokens with a batch dimension and a sequence dimension."
            )
        flattened_tokens = tokens.flatten(start_dim=1)
        if flattened_tokens.size(1) > self.max_token_length:
            raise ValueError(
                f"Flattened token sequences have length {flattened_tokens.size(1)}, "
                f"which exceeds the configured maximum {self.max_token_length}."
            )
        padded_tokens = F.pad(
            flattened_tokens,
            (0, self.max_token_length - flattened_tokens.size(1)),
            value=self.padding_idx,
        )
        return self.token_embedding(padded_tokens).flatten(start_dim=1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one teacher embedding directly from each flattened sequence."""
        widths = self.change_width()
        probabilities = self._compute_width_probabilities(widths)
        hidden = self.flatten_tokens(tokens)
        for layer, neuron_importance in zip(self.layers[:-1], probabilities):
            hidden = self.activation(layer(hidden))
            hidden = hidden * neuron_importance.to(dtype=hidden.dtype)
        return self.layers[-1](hidden), hidden

    def state_dict(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Return a checkpoint whose linear shapes match the current widths."""
        widths = self.change_width()
        self._compute_width_probabilities(widths)
        return super().state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True
    ) -> Any:
        """Resize from checkpointed distributions before restoring all tensors."""
        distribution_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("variational_widths.")
        }
        super().load_state_dict(distribution_state, strict=False)
        self.change_width()
        result = super().load_state_dict(state_dict, strict=strict)
        self.old_parameters = dict(self.named_parameters())
        self._compute_width_probabilities(self.instantiated_widths())
        return result


__all__ = ["DynamicMLPBaseline"]
