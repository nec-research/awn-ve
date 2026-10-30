#        Adaptive-width ESM embedding distillation
#
#   File:     esm_surrogate/_model_base.py
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

"""Internal adaptive-FFN components used by the dynamic-attention model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import torch
from mlwiz.model.interface import ModelInterface
from mlwiz.util import return_class_and_args, s2c
from torch import nn

from distribution import RateParameterizedDistribution, TruncatedDistribution
from dynamic_optimizers import update_optimizer


def _copy_linear_overlap(source: nn.Linear, destination: nn.Linear) -> None:
    """Copy the overlapping part of one linear layer into another."""
    output_size = min(source.out_features, destination.out_features)
    input_size = min(source.in_features, destination.in_features)
    with torch.no_grad():
        destination.weight[:output_size, :input_size].copy_(
            source.weight[:output_size, :input_size]
        )
        if source.bias is not None and destination.bias is not None:
            destination.bias[:output_size].copy_(source.bias[:output_size])


def _rescaled_weight_init(
    weight: torch.Tensor, neuron_importance: torch.Tensor
) -> None:
    """Apply AWN's Gaussian initialization for probability-scaled inputs."""
    squared_importance = neuron_importance.square().sum().detach()
    if not torch.isfinite(squared_importance) or squared_importance <= 0:
        raise ValueError("Neuron importances must have a positive finite norm.")
    standard_deviation = torch.sqrt(2.0 / squared_importance).to(
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        weight.normal_(mean=0.0, std=standard_deviation)


def _resolve_activation(value: object) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve a configured activation function or parameter-free module."""
    activation = s2c(value) if isinstance(value, str) else value
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        activation = activation()
    if not callable(activation):
        raise TypeError(
            "activation must be a callable or a dotted path resolving to one."
        )
    return activation


class DynamicMLP(nn.Module):
    """One-hidden-layer MLP whose hidden dimension can grow or shrink.

    This is the ESM feed-forward specialization of AWN's ``DynamicMLP``. The
    input/output dimensions stay equal to the transformer embedding dimension;
    ``change_width`` changes the first projection's output and the following
    projection's input while preserving their overlapping parameters.
    """

    def __init__(
        self,
        input_layer: nn.Linear,
        output_layer: nn.Linear,
        activation: Callable[[torch.Tensor], torch.Tensor],
        initial_importance: torch.Tensor,
    ) -> None:
        super().__init__()
        if input_layer.out_features != output_layer.in_features:
            raise ValueError("DynamicMLP linear layers must share a hidden width.")
        if (
            initial_importance.ndim != 1
            or initial_importance.numel() != input_layer.out_features
        ):
            raise ValueError(
                f"Expected {input_layer.out_features} initial neuron importances, "
                f"found shape {tuple(initial_importance.shape)}."
            )
        self.layers = nn.ModuleList([input_layer, output_layer])
        self.activation = activation
        _rescaled_weight_init(self.output_layer.weight, initial_importance)

    @property
    def input_layer(self) -> nn.Linear:
        """Return the embedding-to-hidden projection."""
        return self.get_layer(0)

    @property
    def output_layer(self) -> nn.Linear:
        """Return the hidden-to-embedding projection."""
        return self.get_layer(1)

    @property
    def width(self) -> int:
        """Return the currently materialized hidden dimension."""
        input_width = self.input_layer.out_features
        output_width = self.output_layer.in_features
        if input_width != output_width:
            raise RuntimeError(
                "DynamicMLP projections have inconsistent hidden dimensions."
            )
        return int(input_width)

    def get_layer(self, layer_id: int) -> nn.Linear:
        """Return linear projection ``0`` (input) or ``1`` (output)."""
        if layer_id not in (0, 1):
            raise IndexError("DynamicMLP layer_id must be 0 or 1.")
        return self.layers[layer_id]

    def set_layer(self, layer_id: int, layer: nn.Linear) -> None:
        """Replace one of the two MLP projections."""
        if layer_id not in (0, 1):
            raise IndexError("DynamicMLP layer_id must be 0 or 1.")
        if not isinstance(layer, nn.Linear):
            raise TypeError("DynamicMLP layers must be torch.nn.Linear modules.")
        self.layers[layer_id] = layer

    def change_shape(
        self, layer_id: int, num_neurons: int, change_output: bool
    ) -> bool:
        """Resize one dimension of a projection and preserve overlapping weights."""
        num_neurons = int(num_neurons)
        if num_neurons < 1:
            raise ValueError("DynamicMLP dimensions must be positive.")

        layer = self.get_layer(layer_id)
        new_in_features = layer.in_features if change_output else num_neurons
        new_out_features = num_neurons if change_output else layer.out_features
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
        self.set_layer(layer_id, replacement)
        return True

    def change_width(self, num_neurons: int) -> bool:
        """Resize the hidden dimension on both sides of the activation."""
        num_neurons = int(num_neurons)
        if num_neurons < 1:
            raise ValueError("DynamicMLP width must be positive.")
        if num_neurons == self.width:
            return False

        changed_input_projection = self.change_shape(
            layer_id=0,
            num_neurons=num_neurons,
            change_output=True,
        )
        changed_output_projection = self.change_shape(
            layer_id=1,
            num_neurons=num_neurons,
            change_output=False,
        )
        if self.width != num_neurons:
            raise RuntimeError("DynamicMLP failed to materialize the requested width.")
        return changed_input_projection or changed_output_projection

    def forward(
        self, inputs: torch.Tensor, neuron_importance: torch.Tensor
    ) -> torch.Tensor:
        """Apply the dynamic MLP and weight hidden neurons by q(width)."""
        if neuron_importance.ndim != 1 or neuron_importance.numel() != self.width:
            raise ValueError(
                f"Expected {self.width} neuron-importance values, found "
                f"shape {tuple(neuron_importance.shape)}."
            )
        hidden = self.activation(self.input_layer(inputs))
        hidden = hidden * neuron_importance
        return self.output_layer(hidden)


class DynamicTransformerLayer(nn.Module):
    """Native ESM attention/norm block with its FFN replaced by ``DynamicMLP``."""

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        use_rotary_embeddings: bool,
        activation: Callable[[torch.Tensor], torch.Tensor],
        initial_neuron_importance: torch.Tensor,
    ) -> None:
        super().__init__()
        from esm.modules import TransformerLayer

        native_layer = TransformerLayer(
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            add_bias_kv=False,
            use_esm1b_layer_norm=True,
            use_rotary_embeddings=use_rotary_embeddings,
        )
        self.embed_dim = int(embed_dim)
        self.attention_heads = int(attention_heads)
        self.use_rotary_embeddings = bool(use_rotary_embeddings)
        self.self_attn = native_layer.self_attn
        self.self_attn_layer_norm = native_layer.self_attn_layer_norm
        self.final_layer_norm = native_layer.final_layer_norm
        self.dynamic_mlp = DynamicMLP(
            native_layer.fc1,
            native_layer.fc2,
            activation=activation,
            initial_importance=initial_neuron_importance,
        )

    @property
    def ffn_embed_dim(self) -> int:
        """Return the dynamic MLP hidden dimension."""
        return self.dynamic_mlp.width

    @property
    def fc1(self) -> nn.Linear:
        """Expose the current input projection for ESM-style introspection."""
        return self.dynamic_mlp.input_layer

    @property
    def fc2(self) -> nn.Linear:
        """Expose the current output projection for ESM-style introspection."""
        return self.dynamic_mlp.output_layer

    def change_width(self, num_neurons: int) -> bool:
        """Delegate FFN hidden-width changes to the block-owned DynamicMLP."""
        return self.dynamic_mlp.change_width(num_neurons)

    def forward(
        self,
        hidden: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
        need_head_weights: bool = False,
        neuron_importance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run native ESM attention followed by the adaptive-width MLP."""
        if neuron_importance is None:
            raise ValueError("DynamicTransformerLayer requires neuron_importance.")

        residual = hidden
        hidden = self.self_attn_layer_norm(hidden)
        hidden, attention = self.self_attn(
            query=hidden,
            key=hidden,
            value=hidden,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.final_layer_norm(hidden)
        hidden = self.dynamic_mlp(hidden, neuron_importance)
        hidden = residual + hidden
        return hidden, attention


class ESMBackbone(nn.Module):
    """ESM-1b/ESM-2 backbone without language-model or contact heads.

    Attention, normalization, and embeddings come from fair-esm. Every native
    two-projection FFN is replaced by a block-owned :class:`DynamicMLP`.
    """

    def __init__(
        self,
        architecture: str,
        num_layers: int,
        embed_dim: int,
        attention_heads: int,
        initial_ffn_width: int,
        token_dropout: bool,
        max_positions: int,
        emb_layer_norm_before: bool,
        activation: Callable[[torch.Tensor], torch.Tensor],
        initial_ffn_importances: Sequence[torch.Tensor],
    ) -> None:
        super().__init__()
        try:
            import esm
            from esm.modules import (
                ESM1bLayerNorm,
                LearnedPositionalEmbedding,
            )
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "fair-esm is required to construct AdaptiveESM; install the "
                "project dependencies first."
            ) from error

        self.architecture = architecture
        self.num_layers = int(num_layers)
        self.embed_dim = int(embed_dim)
        self.attention_heads = int(attention_heads)
        self.token_dropout = bool(token_dropout)
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.padding_idx = self.alphabet.padding_idx
        self.mask_idx = self.alphabet.mask_idx
        self.cls_idx = self.alphabet.cls_idx
        self.eos_idx = self.alphabet.eos_idx
        self.prepend_bos = self.alphabet.prepend_bos
        self.append_eos = self.alphabet.append_eos
        self.embed_scale = 1.0

        self.embed_tokens = nn.Embedding(
            len(self.alphabet), self.embed_dim, padding_idx=self.padding_idx
        )
        if len(initial_ffn_importances) != self.num_layers:
            raise ValueError("Expected one initial FFN importance vector per block.")
        use_rotary_embeddings = architecture == "esm2"
        self.layers = nn.ModuleList(
            [
                DynamicTransformerLayer(
                    self.embed_dim,
                    initial_ffn_width,
                    self.attention_heads,
                    use_rotary_embeddings=use_rotary_embeddings,
                    activation=activation,
                    initial_neuron_importance=initial_importance,
                )
                for initial_importance in initial_ffn_importances
            ]
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        if architecture == "esm1b":
            self.embed_positions: nn.Module | None = LearnedPositionalEmbedding(
                max_positions, self.embed_dim, self.padding_idx
            )
            self.emb_layer_norm_before: nn.Module | None = (
                ESM1bLayerNorm(self.embed_dim) if emb_layer_norm_before else None
            )
        else:
            self.embed_positions = None
            self.emb_layer_norm_before = None

    def _embed(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Apply the architecture-specific native ESM input embedding path."""
        hidden = self.embed_scale * self.embed_tokens(tokens)
        if self.token_dropout:
            hidden.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            expected_mask_ratio = 0.15 * 0.8
            source_lengths = (~padding_mask).sum(-1).clamp_min(1)
            observed_mask_ratio = (tokens == self.mask_idx).sum(-1).to(
                hidden.dtype
            ) / source_lengths
            hidden = (
                hidden
                * (1.0 - expected_mask_ratio)
                / (1.0 - observed_mask_ratio)[:, None, None]
            )

        if self.embed_positions is not None:
            hidden = hidden + self.embed_positions(tokens)
        if self.emb_layer_norm_before is not None:
            hidden = self.emb_layer_norm_before(hidden)
        return hidden * (1 - padding_mask.unsqueeze(-1).type_as(hidden))

    def forward(
        self,
        tokens: torch.Tensor,
        neuron_importances: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return final per-token representations using each block's DynamicMLP."""
        if tokens.ndim != 2:
            raise ValueError(
                f"Expected token tensor shaped [batch, length], found "
                f"{tuple(tokens.shape)}."
            )
        if len(neuron_importances) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} neuron-importance vectors, found "
                f"{len(neuron_importances)}."
            )
        padding_mask = tokens.eq(self.padding_idx)
        hidden = self._embed(tokens, padding_mask).transpose(0, 1)

        for layer, neuron_importance in zip(self.layers, neuron_importances):
            # An all-false mask is equivalent to None and remains exportable.
            hidden, _ = layer(
                hidden,
                self_attn_padding_mask=padding_mask,
                need_head_weights=False,
                neuron_importance=neuron_importance,
            )
        return self.emb_layer_norm_after(hidden).transpose(0, 1)


class _BaseAdaptiveESM(ModelInterface):
    """Internal adaptive-FFN ESM used to construct the public model."""

    _DEFAULTS = {
        "esm1b": SimpleNamespace(
            num_layers=34,
            embed_dim=1280,
            attention_heads=20,
            max_positions=1024,
            emb_layer_norm_before=True,
        ),
        "esm2": SimpleNamespace(
            num_layers=33,
            embed_dim=1280,
            attention_heads=20,
            max_positions=1024,
            emb_layer_norm_before=False,
        ),
    }

    def __init__(
        self, dim_input_features: int, dim_target: int, config: Mapping[str, Any]
    ) -> None:
        super().__init__(dim_input_features, dim_target, config)
        config = dict(config)
        architecture = str(config.get("architecture", "esm2")).lower().replace("-", "")
        if architecture not in self._DEFAULTS:
            raise ValueError("architecture must be 'esm1b' or 'esm2'.")
        defaults = self._DEFAULTS[architecture]

        self.architecture = architecture
        self.num_layers = int(config.get("num_layers", defaults.num_layers))
        self.embed_dim = int(config.get("embed_dim", defaults.embed_dim))
        self.attention_heads = int(
            config.get("attention_heads", defaults.attention_heads)
        )
        if self.embed_dim != int(dim_target):
            raise ValueError(
                f"Student embed_dim ({self.embed_dim}) must match the teacher "
                f"embedding dimension ({dim_target})."
            )
        if self.embed_dim % self.attention_heads != 0:
            raise ValueError("embed_dim must be divisible by attention_heads.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")

        self.quantile = float(config.get("quantile", 0.9))
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0, 1).")
        distribution_class, distribution_args = return_class_and_args(
            config, "truncated_distribution"
        )
        if distribution_class is None:
            raise ValueError("truncated_distribution must be configured.")
        if distribution_class is not TruncatedDistribution:
            raise TypeError(
                "AdaptiveESM currently expects distribution.TruncatedDistribution."
            )

        self.variational_widths = nn.ModuleList(
            [
                distribution_class(
                    truncation_quantile=self.quantile, **distribution_args
                )
                for _ in range(self.num_layers)
            ]
        )
        for distribution in self.variational_widths:
            distribution.to("cpu")

        for layer_id, distribution in enumerate(self.variational_widths):
            base_distribution = distribution.discretized_distribution.base_distribution
            if not isinstance(base_distribution, RateParameterizedDistribution):
                raise TypeError(
                    "The global width prior requires a rate-parameterized "
                    f"distribution, found {base_distribution.__class__.__name__} "
                    f"at transformer block {layer_id}."
                )

        theta_prior_scale = config.get("theta_prior_scale")
        if theta_prior_scale is not None and float(theta_prior_scale) <= 0.0:
            raise ValueError("theta_prior_scale must be positive when configured.")
        self.register_buffer(
            "theta_prior_scale",
            None
            if theta_prior_scale is None
            else torch.tensor([float(theta_prior_scale)]),
        )

        total_width_prior_mean = config.get("total_width_prior_mean")
        total_width_prior_scale = config.get("total_width_prior_scale")
        if (total_width_prior_mean is None) != (total_width_prior_scale is None):
            raise ValueError(
                "total_width_prior_mean and total_width_prior_scale must be "
                "configured together."
            )
        if total_width_prior_mean is not None and float(total_width_prior_mean) <= 0.0:
            raise ValueError("total_width_prior_mean must be positive when configured.")
        if (
            total_width_prior_scale is not None
            and float(total_width_prior_scale) <= 0.0
        ):
            raise ValueError(
                "total_width_prior_scale must be positive when configured."
            )
        if total_width_prior_mean is None:
            global_rate_prior_mean = None
            global_rate_prior_scale = None
        else:
            base_distribution = self.variational_widths[
                0
            ].discretized_distribution.base_distribution
            global_rate_prior_mean = torch.as_tensor(
                base_distribution._find_rate(
                    desired_size=float(total_width_prior_mean),
                    quantile=self.quantile,
                )
            ).reshape(1)
            rate_at_one_width_scale = torch.as_tensor(
                base_distribution._find_rate(
                    desired_size=(
                        float(total_width_prior_mean) + float(total_width_prior_scale)
                    ),
                    quantile=self.quantile,
                )
            ).reshape(1)
            global_rate_prior_scale = (
                global_rate_prior_mean - rate_at_one_width_scale
            ).abs()
            if torch.any(global_rate_prior_scale == 0):
                raise ValueError(
                    "total_width_prior_scale must produce a nonzero rate scale."
                )
        self.total_width_prior_mean = (
            None if total_width_prior_mean is None else float(total_width_prior_mean)
        )
        self.register_buffer(
            "total_width_prior_rate_mean",
            global_rate_prior_mean,
        )
        self.register_buffer(
            "total_width_prior_scale",
            global_rate_prior_scale,
        )
        if "n_observations" not in config:
            raise ValueError(
                "n_observations must be configured for ELBO normalization."
            )
        self.n_observations = int(config["n_observations"])
        if self.n_observations < 1:
            raise ValueError("n_observations must be positive.")

        initial_widths = self.current_widths()
        if len(set(initial_widths)) != 1:
            raise ValueError(
                "All FFN distributions must start at the same width during "
                "backbone construction."
            )
        initial_ffn_width = initial_widths[0]
        initial_ffn_importances = self._compute_width_probabilities(initial_widths)
        activation = _resolve_activation(
            config.get("activation", "torch.nn.functional.gelu")
        )
        self.backbone = ESMBackbone(
            architecture=architecture,
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            attention_heads=self.attention_heads,
            initial_ffn_width=initial_ffn_width,
            token_dropout=bool(config.get("token_dropout", True)),
            max_positions=int(config.get("max_positions", defaults.max_positions)),
            emb_layer_norm_before=bool(
                config.get("emb_layer_norm_before", defaults.emb_layer_norm_before)
            ),
            activation=activation,
            initial_ffn_importances=initial_ffn_importances,
        )

        self.torch_optimizer: torch.optim.Optimizer | None = None
        self.old_parameters = dict(self.named_parameters())

    @property
    def padding_idx(self) -> int:
        """Return the fair-esm padding token index."""
        return self.backbone.padding_idx

    def set_optimizer(self, optimizer: object) -> None:
        """Attach MLWiz's optimizer so resized parameters retain optimizer state."""
        self.torch_optimizer = getattr(optimizer, "optimizer", optimizer)

    def current_widths(self) -> list[int]:
        """Return integer FFN widths implied by the current distributions."""
        return [
            max(int(distribution.compute_truncation_number()), 1)
            for distribution in self.variational_widths
        ]

    def instantiated_widths(self) -> list[int]:
        """Return FFN widths currently materialized in transformer blocks."""
        return [layer.dynamic_mlp.width for layer in self.backbone.layers]

    def change_width(self) -> list[int]:
        """Resize every block DynamicMLP and migrate optimizer state.

        Each transformer block changes the output width of its MLP input
        projection and the matching input width of its output projection. This
        mirrors AWN's two-sided ``change_shape`` update for a hidden layer.
        """
        widths = self.current_widths()
        changed = False
        for layer, width in zip(self.backbone.layers, widths):
            changed = layer.change_width(width) or changed
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
        """Compute differentiable q(width) neuron weights at fixed widths."""
        probabilities = [
            distribution.compute_probability_vector_known_width(width)
            for distribution, width in zip(self.variational_widths, widths)
        ]
        self.qW_probs = probabilities
        return probabilities

    def _pool_amino_acids(
        self, token_representations: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool amino-acid tokens exactly as the teacher encoder does."""
        amino_acid_mask = tokens.ne(self.backbone.padding_idx)
        if self.backbone.prepend_bos:
            amino_acid_mask &= tokens.ne(self.backbone.cls_idx)
        if self.backbone.append_eos:
            amino_acid_mask &= tokens.ne(self.backbone.eos_idx)
        counts = amino_acid_mask.sum(dim=1, keepdim=True)
        # Preserve eager validation without forcing export to guard on values.
        if not torch.compiler.is_compiling() and torch.any(counts == 0):
            raise ValueError("Cannot pool a protein with zero amino-acid tokens.")
        return (token_representations * amino_acid_mask.unsqueeze(-1)).sum(
            dim=1
        ) / counts.to(token_representations.dtype)

    def _theta_log_prior(self) -> torch.Tensor:
        """Return the unnormalized isotropic Gaussian log prior over the backbone."""
        if self.theta_prior_scale is None:
            parameter = next(self.backbone.parameters())
            return parameter.new_zeros(1)
        squared_norm = sum(
            parameter.square().sum() for parameter in self.backbone.parameters()
        )
        return (-squared_norm / (2.0 * self.theta_prior_scale.square())).reshape(1)

    def _global_width_log_prior(self) -> torch.Tensor:
        """Return a Gaussian log prior over the sum of all width rates."""
        if (
            self.total_width_prior_rate_mean is None
            or self.total_width_prior_scale is None
        ):
            parameter = next(self.variational_widths.parameters())
            return parameter.new_zeros(1)
        rates = torch.cat(
            [
                distribution.discretized_distribution.base_distribution.parameter.reshape(
                    -1
                )
                for distribution in self.variational_widths
            ]
        )
        global_rate = rates.sum().reshape(1)
        return -torch.log(self.total_width_prior_scale) - (
            global_rate - self.total_width_prior_rate_mean
        ).square() / (2.0 * self.total_width_prior_scale.square())

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, object], torch.Tensor, torch.Tensor]:
        """Produce embeddings, diagnostics, and reference-style log priors."""
        widths = self.change_width()
        probabilities = self._compute_width_probabilities(widths)
        token_representations = self.backbone(tokens, probabilities)
        protein_embeddings = self._pool_amino_acids(token_representations, tokens)
        theta_log_prior = self._theta_log_prior()
        global_width_log_prior = self._global_width_log_prior()
        diagnostics = {
            "widths": list(widths),
            "total_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "width_probabilities": [
                probability.detach() for probability in probabilities
            ],
            "n_observations": self.n_observations,
        }
        return (
            protein_embeddings,
            diagnostics,
            theta_log_prior,
            global_width_log_prior,
        )

    def state_dict(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Return a checkpoint whose materialized FFN shapes match q(width)."""
        widths = self.change_width()
        self._compute_width_probabilities(widths)
        return super().state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True
    ) -> Any:
        """Resize from checkpointed distributions before loading FFN tensors."""
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
