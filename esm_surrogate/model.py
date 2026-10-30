#        Adaptive-width ESM embedding distillation
#
#   File:     esm_surrogate/model.py
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

"""Adaptive ESM with dynamic query/key and feed-forward widths."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from mlwiz.util import return_class_and_args
from torch import nn

from distribution import RateParameterizedDistribution, TruncatedDistribution
from dynamic_optimizers import update_optimizer
from esm_surrogate._model_base import (
    DynamicMLP,
    DynamicTransformerLayer,
    ESMBackbone,
    _BaseAdaptiveESM,
    _copy_linear_overlap,
    _rescaled_weight_init,
)


class DynamicAttention(nn.Module):
    """ESM self-attention with a shared adaptive query/key head width.

    Query and key form a shared latent space, so they are resized together and
    weighted symmetrically by one neuron-importance vector. Value and output
    projections retain the backbone embedding dimension. The adaptive width is
    per head, ensuring every positive distribution width defines valid
    multi-head geometry. Query and value weights use AWN's probability-rescaled
    Gaussian initialization; key retains fair-esm's initialization.
    """

    def __init__(
        self,
        native_attention: nn.Module,
        initial_width: int,
        initial_importance: torch.Tensor,
    ) -> None:
        super().__init__()
        if native_attention.bias_k is not None or native_attention.bias_v is not None:
            raise ValueError("DynamicAttention does not support bias key/value tokens.")
        if native_attention.add_zero_attn:
            raise ValueError("DynamicAttention does not support zero attention tokens.")

        self.embed_dim = int(native_attention.embed_dim)
        self.num_heads = int(native_attention.num_heads)
        self.dropout = float(native_attention.dropout)
        self.use_rotary_embeddings = native_attention.rot_emb is not None
        # Store query/key channels by latent dimension first, then by head.
        # Prefix-preserving resize and optimizer migration can consequently
        # retain the same neurons in every head.
        self.q_proj = self._interleave_heads(native_attention.q_proj, self.num_heads)
        self.k_proj = self._interleave_heads(native_attention.k_proj, self.num_heads)
        self.v_proj = native_attention.v_proj
        self.out_proj = native_attention.out_proj
        self.head_dim = 0
        self.scaling = 1.0
        self.rot_emb: nn.Module | None = None
        self.change_width(initial_width)
        if initial_importance.ndim != 1 or initial_importance.numel() != initial_width:
            raise ValueError(
                f"Expected {initial_width} initial attention importances, found "
                f"shape {tuple(initial_importance.shape)}."
            )
        _rescaled_weight_init(self.q_proj.weight, initial_importance)
        _rescaled_weight_init(self.v_proj.weight, initial_importance)

    @property
    def width(self) -> int:
        """Return the materialized output size of each query/key projection."""
        if self.q_proj.out_features != self.k_proj.out_features:
            raise RuntimeError("Query and key projections have inconsistent widths.")
        return int(self.q_proj.out_features)

    @property
    def head_width(self) -> int:
        """Return the adaptive query/key dimension within each head."""
        if self.width % self.num_heads != 0:
            raise RuntimeError("Query/key width is not divisible into attention heads.")
        return self.width // self.num_heads

    @staticmethod
    def _interleave_heads(projection: nn.Linear, num_heads: int) -> nn.Linear:
        """Convert native head-major rows to resize-friendly neuron-major rows."""
        if projection.out_features % num_heads != 0:
            raise ValueError("Projection width must be divisible by num_heads.")
        head_width = projection.out_features // num_heads
        replacement = nn.Linear(
            projection.in_features,
            projection.out_features,
            bias=projection.bias is not None,
        ).to(device=projection.weight.device, dtype=projection.weight.dtype)
        replacement.weight.requires_grad_(projection.weight.requires_grad)
        with torch.no_grad():
            replacement.weight.copy_(
                projection.weight.view(num_heads, head_width, projection.in_features)
                .transpose(0, 1)
                .reshape_as(replacement.weight)
            )
            if replacement.bias is not None and projection.bias is not None:
                replacement.bias.requires_grad_(projection.bias.requires_grad)
                replacement.bias.copy_(
                    projection.bias.view(num_heads, head_width)
                    .transpose(0, 1)
                    .reshape_as(replacement.bias)
                )
        return replacement

    @staticmethod
    def _resize_projection(projection: nn.Linear, width: int) -> nn.Linear:
        """Resize a projection's output while retaining overlapping state."""
        replacement = nn.Linear(
            projection.in_features,
            width,
            bias=projection.bias is not None,
        ).to(device=projection.weight.device, dtype=projection.weight.dtype)
        replacement.weight.requires_grad_(projection.weight.requires_grad)
        if replacement.bias is not None and projection.bias is not None:
            replacement.bias.requires_grad_(projection.bias.requires_grad)
        _copy_linear_overlap(projection, replacement)
        return replacement

    def _refresh_geometry(self) -> None:
        """Update per-head scaling and rotary state after a resize."""
        self.head_dim = self.head_width
        self.scaling = self.head_dim**-0.5
        rotary_dim = self.head_dim - self.head_dim % 2
        if not self.use_rotary_embeddings or rotary_dim == 0:
            self.rot_emb = None
            return

        from esm.rotary_embedding import RotaryEmbedding

        self.rot_emb = RotaryEmbedding(rotary_dim).to(
            device=self.q_proj.weight.device,
            dtype=self.q_proj.weight.dtype,
        )

    def change_width(self, head_width: int) -> bool:
        """Resize every query/key head while preserving overlapping weights."""
        head_width = int(head_width)
        if head_width < 1:
            raise ValueError("DynamicAttention head width must be positive.")
        projection_width = head_width * self.num_heads
        if projection_width == self.width:
            if self.head_dim == 0:
                self._refresh_geometry()
            return False

        self.q_proj = self._resize_projection(self.q_proj, projection_width)
        self.k_proj = self._resize_projection(self.k_proj, projection_width)
        self._refresh_geometry()
        return True

    def _apply_rotary(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to the largest even prefix of each per-head vector."""
        if self.rot_emb is None:
            return query, key
        rotary_dim = int(self.rot_emb.inv_freq.numel() * 2)
        rotated_query, rotated_key = self.rot_emb(
            query[..., :rotary_dim], key[..., :rotary_dim]
        )
        if rotary_dim == self.head_dim:
            return rotated_query, rotated_key
        return (
            torch.cat((rotated_query, query[..., rotary_dim:]), dim=-1),
            torch.cat((rotated_key, key[..., rotary_dim:]), dim=-1),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        attention_importance: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply dynamic multi-head self-attention."""
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported DynamicAttention arguments: {unsupported}.")
        if attention_importance is None:
            raise ValueError("DynamicAttention requires attention_importance.")
        if (
            attention_importance.ndim != 1
            or attention_importance.numel() != self.head_width
        ):
            raise ValueError(
                f"Expected {self.head_width} attention-importance values, found "
                f"shape {tuple(attention_importance.shape)}."
            )

        target_length, batch_size, embed_dim = query.shape
        source_length = key.size(0)
        if embed_dim != self.embed_dim or value.size(-1) != self.embed_dim:
            raise ValueError(
                f"DynamicAttention expects embedding dimension {self.embed_dim}."
            )

        # Splitting the importance equally between q and k applies it once to
        # their dot product while treating the two projections symmetrically.
        importance_scale = attention_importance.sqrt().to(dtype=query.dtype)
        projected_query = self.q_proj(query).view(
            target_length, batch_size, self.head_dim, self.num_heads
        )
        projected_key = self.k_proj(key).view(
            source_length, batch_size, self.head_dim, self.num_heads
        )
        projected_query = projected_query.permute(0, 1, 3, 2)
        projected_key = projected_key.permute(0, 1, 3, 2)
        projected_query = projected_query * importance_scale
        projected_key = projected_key * importance_scale
        projected_value = self.v_proj(value)

        projected_query = (
            projected_query.contiguous()
            .view(target_length, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        projected_key = (
            projected_key.contiguous()
            .view(source_length, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        value_head_dim = self.embed_dim // self.num_heads
        projected_value = (
            projected_value.contiguous()
            .view(source_length, batch_size * self.num_heads, value_head_dim)
            .transpose(0, 1)
        )
        projected_query, projected_key = self._apply_rotary(
            projected_query, projected_key
        )

        attention_weights = torch.bmm(
            projected_query * self.scaling, projected_key.transpose(1, 2)
        )
        if attn_mask is not None:
            attention_weights = attention_weights + attn_mask.unsqueeze(0)
        if key_padding_mask is not None:
            if tuple(key_padding_mask.shape) != (batch_size, source_length):
                raise ValueError(
                    "key_padding_mask must have shape [batch, source_length]."
                )
            attention_weights = attention_weights.view(
                batch_size, self.num_heads, target_length, source_length
            )
            attention_weights = attention_weights.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool), float("-inf")
            )
            attention_weights = attention_weights.view(
                batch_size * self.num_heads, target_length, source_length
            )

        normalized_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32)
        attention_probabilities = F.dropout(
            normalized_weights.to(attention_weights.dtype),
            p=self.dropout,
            training=self.training,
        )
        attended = torch.bmm(attention_probabilities, projected_value)
        attended = (
            attended.transpose(0, 1)
            .contiguous()
            .view(target_length, batch_size, self.embed_dim)
        )
        attended = self.out_proj(attended)

        if not need_weights and not need_head_weights:
            return attended, None
        returned_weights = normalized_weights.view(
            batch_size, self.num_heads, target_length, source_length
        ).to(attended.dtype)
        returned_weights = returned_weights.transpose(1, 0)
        if not need_head_weights:
            returned_weights = returned_weights.mean(dim=0)
        return attended, returned_weights


class DynamicAttentionTransformerLayer(DynamicTransformerLayer):
    """Transformer block with adaptive query/key and feed-forward widths."""

    def __init__(
        self,
        source_layer: DynamicTransformerLayer,
        attention_width: int,
        attention_importance: torch.Tensor,
    ) -> None:
        nn.Module.__init__(self)
        self.embed_dim = source_layer.embed_dim
        self.attention_heads = source_layer.attention_heads
        self.use_rotary_embeddings = source_layer.use_rotary_embeddings
        self.self_attn = DynamicAttention(
            source_layer.self_attn,
            attention_width,
            attention_importance,
        )
        self.self_attn_layer_norm = source_layer.self_attn_layer_norm
        self.final_layer_norm = source_layer.final_layer_norm
        self.dynamic_mlp = source_layer.dynamic_mlp

    @property
    def attention_width(self) -> int:
        """Return the materialized size of each query/key projection."""
        return self.self_attn.width

    @property
    def attention_head_width(self) -> int:
        """Return the materialized adaptive dimension within each head."""
        return self.self_attn.head_width

    def change_attention_width(self, width: int) -> bool:
        """Resize this block's query and key projections."""
        return self.self_attn.change_width(width)

    def forward(
        self,
        hidden: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
        need_head_weights: bool = False,
        neuron_importance: torch.Tensor | None = None,
        attention_importance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run adaptive attention followed by the adaptive MLP."""
        if neuron_importance is None:
            raise ValueError(
                "DynamicAttentionTransformerLayer requires neuron_importance."
            )

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
            attention_importance=attention_importance,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.final_layer_norm(hidden)
        hidden = self.dynamic_mlp(hidden, neuron_importance)
        hidden = residual + hidden
        return hidden, attention


class DynamicAttentionESMBackbone(ESMBackbone):
    """Existing ESM backbone extended with dynamic-attention blocks."""

    def __init__(
        self,
        source: ESMBackbone,
        initial_attention_widths: Sequence[int],
        initial_attention_importances: Sequence[torch.Tensor],
    ) -> None:
        nn.Module.__init__(self)
        if len(initial_attention_widths) != len(source.layers):
            raise ValueError("Expected one initial attention width per block.")
        if len(initial_attention_importances) != len(source.layers):
            raise ValueError("Expected one initial attention importance per block.")

        for name in (
            "architecture",
            "num_layers",
            "embed_dim",
            "attention_heads",
            "token_dropout",
            "alphabet",
            "padding_idx",
            "mask_idx",
            "cls_idx",
            "eos_idx",
            "prepend_bos",
            "append_eos",
            "embed_scale",
        ):
            setattr(self, name, getattr(source, name))
        self.embed_tokens = source.embed_tokens
        self.embed_positions = source.embed_positions
        self.emb_layer_norm_before = source.emb_layer_norm_before
        self.emb_layer_norm_after = source.emb_layer_norm_after
        self.layers = nn.ModuleList(
            [
                DynamicAttentionTransformerLayer(layer, width, importance)
                for layer, width, importance in zip(
                    source.layers,
                    initial_attention_widths,
                    initial_attention_importances,
                )
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        neuron_importances: Sequence[torch.Tensor],
        attention_importances: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return representations using both adaptive components per block."""
        if tokens.ndim != 2:
            raise ValueError(
                f"Expected token tensor shaped [batch, length], found "
                f"{tuple(tokens.shape)}."
            )
        if len(neuron_importances) != len(self.layers):
            raise ValueError("Expected one MLP importance vector per block.")
        if len(attention_importances) != len(self.layers):
            raise ValueError("Expected one attention importance vector per block.")

        padding_mask = tokens.eq(self.padding_idx)
        hidden = self._embed(tokens, padding_mask).transpose(0, 1)
        for layer, neuron_importance, attention_importance in zip(
            self.layers, neuron_importances, attention_importances
        ):
            hidden, _ = layer(
                hidden,
                self_attn_padding_mask=padding_mask,
                need_head_weights=False,
                neuron_importance=neuron_importance,
                attention_importance=attention_importance,
            )
        return self.emb_layer_norm_after(hidden).transpose(0, 1)


class AdaptiveESM(_BaseAdaptiveESM):
    """Adaptive ESM whose MLP and query/key widths vary independently.

    ``attention_truncated_distribution`` configures the query/key width per
    head and falls back to ``truncated_distribution`` when omitted. Each
    materialized query/key projection therefore has size
    ``attention_heads * attention_head_width``.
    """

    def __init__(
        self, dim_input_features: int, dim_target: int, config: Mapping[str, Any]
    ) -> None:
        config = dict(config)
        super().__init__(dim_input_features, dim_target, config)

        distribution_key = (
            "attention_truncated_distribution"
            if "attention_truncated_distribution" in config
            else "truncated_distribution"
        )
        distribution_class, distribution_args = return_class_and_args(
            config, distribution_key
        )
        if distribution_class is not TruncatedDistribution:
            raise TypeError(
                "AdaptiveESM expects a TruncatedDistribution for attention widths."
            )
        self.attention_variational_widths = nn.ModuleList(
            [
                distribution_class(
                    truncation_quantile=self.quantile, **distribution_args
                )
                for _ in range(self.num_layers)
            ]
        )
        for layer_id, distribution in enumerate(self.attention_variational_widths):
            distribution.to("cpu")
            base_distribution = distribution.discretized_distribution.base_distribution
            if not isinstance(base_distribution, RateParameterizedDistribution):
                raise TypeError(
                    "Adaptive attention requires a rate-parameterized "
                    f"distribution, found {base_distribution.__class__.__name__} "
                    f"at transformer block {layer_id}."
                )

        initial_attention_widths = self.current_attention_widths()
        initial_attention_importances = self._compute_attention_probabilities(
            initial_attention_widths
        )
        self.backbone = DynamicAttentionESMBackbone(
            self.backbone,
            initial_attention_widths,
            initial_attention_importances,
        )
        compression_ratio = config.get("parameter_compression_target_ratio")
        compression_scale = config.get("parameter_compression_lambda")
        if (compression_ratio is None) != (compression_scale is None):
            raise ValueError(
                "parameter_compression_target_ratio and "
                "parameter_compression_lambda must be configured together."
            )
        if compression_ratio is not None and not 0.0 < float(compression_ratio) <= 1.0:
            raise ValueError(
                "parameter_compression_target_ratio must be in (0, 1]."
            )
        if compression_scale is not None and float(compression_scale) < 0.0:
            raise ValueError("parameter_compression_lambda must be non-negative.")
        self.parameter_compression_target_ratio = (
            None if compression_ratio is None else float(compression_ratio)
        )
        self.parameter_compression_lambda = (
            None if compression_scale is None else float(compression_scale)
        )

        materialized_widths = torch.tensor(
            self.instantiated_mlp_widths(), dtype=torch.get_default_dtype()
        )
        materialized_attention_widths = torch.tensor(
            self.instantiated_attention_head_widths(),
            dtype=torch.get_default_dtype(),
        )
        adaptive_parameters = self._adaptive_parameter_count(
            materialized_widths, materialized_attention_widths
        )
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        self.register_buffer(
            "_constant_parameter_count",
            adaptive_parameters.new_tensor(total_parameters) - adaptive_parameters,
            persistent=False,
        )
        self.register_buffer(
            "_initial_mlp_widths",
            self.soft_mlp_widths().detach(),
            persistent=False,
        )
        self.register_buffer(
            "_initial_attention_head_widths",
            self.soft_attention_head_widths().detach(),
            persistent=False,
        )
        self.register_buffer(
            "_initial_parameter_count",
            self.soft_parameter_count().detach(),
            persistent=False,
        )
        self.old_parameters = dict(self.named_parameters())

    @torch.no_grad()
    def set_compression_reference(self) -> None:
        """Use the current soft widths as the compression-ratio baseline.

        Compression starts from a restored checkpoint rather than necessarily
        from the architecture's configured widths. Capturing the checkpoint's
        soft widths here makes the parameter and FLOPs ratios equal one at the
        start of the compression run.
        """
        self._initial_mlp_widths.copy_(self.soft_mlp_widths())
        self._initial_attention_head_widths.copy_(
            self.soft_attention_head_widths()
        )
        self._initial_parameter_count.copy_(self.soft_parameter_count())

    def _soft_widths(
        self, distributions: Sequence[TruncatedDistribution]
    ) -> torch.Tensor:
        """Return continuous quantile widths with gradients to every rate."""
        return torch.cat(
            [
                distribution.discretized_distribution.base_distribution.quantile(
                    self.quantile
                )[1].reshape(1)
                for distribution in distributions
            ]
        )

    def soft_mlp_widths(self) -> torch.Tensor:
        """Return differentiable continuous FFN widths for all blocks."""
        return self._soft_widths(self.variational_widths)

    def soft_attention_head_widths(self) -> torch.Tensor:
        """Return differentiable continuous query/key widths within each head."""
        return self._soft_widths(self.attention_variational_widths)

    def _adaptive_parameter_count(
        self,
        mlp_widths: torch.Tensor,
        attention_head_widths: torch.Tensor,
    ) -> torch.Tensor:
        """Count parameters whose shapes depend on the supplied soft widths."""
        mlp_widths = mlp_widths.reshape(-1)
        attention_head_widths = attention_head_widths.reshape(-1)
        if mlp_widths.numel() != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} MLP widths.")
        if attention_head_widths.numel() != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} attention widths.")

        total = mlp_widths.new_zeros(())
        for layer, mlp_width, attention_head_width in zip(
            self.backbone.layers, mlp_widths, attention_head_widths
        ):
            mlp = layer.dynamic_mlp
            mlp_coefficient = (
                mlp.input_layer.in_features
                + int(mlp.input_layer.bias is not None)
                + mlp.output_layer.out_features
            )
            attention = layer.self_attn
            attention_coefficient = self.attention_heads * (
                attention.q_proj.in_features
                + int(attention.q_proj.bias is not None)
                + attention.k_proj.in_features
                + int(attention.k_proj.bias is not None)
            )
            total = (
                total
                + mlp_width * mlp_coefficient
                + attention_head_width * attention_coefficient
            )
        return total

    def soft_parameter_count(self) -> torch.Tensor:
        """Return the differentiable total parameter count of the network."""
        adaptive_count = self._adaptive_parameter_count(
            self.soft_mlp_widths(), self.soft_attention_head_widths()
        )
        return adaptive_count + self._constant_parameter_count

    def _flops_for_widths(
        self,
        sequence_length: int,
        mlp_widths: torch.Tensor,
        attention_head_widths: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate Transformer matmul FLOPs for supplied continuous widths."""
        sequence_length = int(sequence_length)
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")
        mlp_widths = mlp_widths.reshape(-1)
        attention_head_widths = attention_head_widths.reshape(-1)
        if mlp_widths.numel() != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} MLP widths.")
        if attention_head_widths.numel() != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} attention widths.")
        total = mlp_widths.new_zeros(())
        for layer, mlp_width, attention_head_width in zip(
            self.backbone.layers, mlp_widths, attention_head_widths
        ):
            mlp = layer.dynamic_mlp
            attention = layer.self_attn
            # Dense projections: 2 FLOPs per input/output pair.
            total = total + 2 * sequence_length * mlp_width * (
                mlp.input_layer.in_features + mlp.output_layer.out_features
            )
            total = total + 2 * sequence_length * (
                attention.v_proj.in_features * attention.v_proj.out_features
                + attention.out_proj.in_features * attention.out_proj.out_features
            )
            total = total + 2 * sequence_length * self.attention_heads * (
                attention.q_proj.in_features + attention.k_proj.in_features
            ) * attention_head_width
            # QK^T uses the soft head width; attention-times-value retains the
            # fixed value projection width.
            total = total + 2 * sequence_length**2 * (
                self.attention_heads * attention_head_width
                + attention.v_proj.out_features
            )
        return total

    def soft_flops(self, sequence_length: int) -> torch.Tensor:
        """Estimate per-sequence Transformer matmul FLOPs differentiably.

        One multiply-add is counted as two FLOPs. The estimate covers all
        dense projections and both attention matrix multiplications. Embedding
        lookups, normalization, activations, masking, and softmax are omitted.
        """
        return self._flops_for_widths(
            sequence_length,
            self.soft_mlp_widths(),
            self.soft_attention_head_widths(),
        )

    def _compression_diagnostics(
        self, sequence_length: int
    ) -> dict[str, torch.Tensor]:
        """Return differentiable size, FLOPs, ratio, and budget penalty."""
        soft_parameters = self.soft_parameter_count()
        parameter_ratio = soft_parameters / self._initial_parameter_count.clamp_min(
            1.0
        )
        if self.parameter_compression_target_ratio is None:
            penalty = torch.zeros_like(parameter_ratio)
        else:
            penalty = self.parameter_compression_lambda * (
                parameter_ratio - self.parameter_compression_target_ratio
            ).square()
        soft_flops = self.soft_flops(sequence_length)
        initial_flops = self._flops_for_widths(
            sequence_length,
            self._initial_mlp_widths,
            self._initial_attention_head_widths,
        )
        flops_ratio = soft_flops / initial_flops.clamp_min(1.0)
        if self.parameter_compression_target_ratio is None:
            flops_penalty = torch.zeros_like(flops_ratio)
        else:
            flops_penalty = self.parameter_compression_lambda * (
                flops_ratio - self.parameter_compression_target_ratio
            ).square()
        return {
            "soft_total_parameters": soft_parameters.reshape(1),
            "parameter_compression_ratio": parameter_ratio.reshape(1),
            "parameter_compression_penalty": penalty.reshape(1),
            "soft_flops": soft_flops.reshape(1),
            "flops_compression_ratio": flops_ratio.reshape(1),
            "flops_compression_penalty": flops_penalty.reshape(1),
        }

    def current_attention_widths(self) -> list[int]:
        """Return per-head widths implied by the attention distributions."""
        return [
            max(int(distribution.compute_truncation_number()), 1)
            for distribution in self.attention_variational_widths
        ]

    def instantiated_mlp_widths(self) -> list[int]:
        """Return the materialized MLP hidden width of every block."""
        return self.instantiated_widths()

    def instantiated_attention_widths(self) -> list[int]:
        """Return the total query/key projection size of every block."""
        return [layer.attention_width for layer in self.backbone.layers]

    def instantiated_attention_head_widths(self) -> list[int]:
        """Return the materialized adaptive query/key width within each head."""
        return [layer.attention_head_width for layer in self.backbone.layers]

    def change_width(self) -> list[int]:
        """Resize all adaptive projections and migrate optimizer state once."""
        mlp_widths = self.current_widths()
        attention_head_widths = self.current_attention_widths()

        changed = False
        for layer, mlp_width, attention_head_width in zip(
            self.backbone.layers, mlp_widths, attention_head_widths
        ):
            changed = layer.change_width(mlp_width) or changed
            changed = layer.change_attention_width(attention_head_width) or changed
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
        return mlp_widths

    def _compute_attention_probabilities(
        self, widths: Sequence[int]
    ) -> list[torch.Tensor]:
        """Compute shared differentiable query/key importance vectors."""
        probabilities = [
            distribution.compute_probability_vector_known_width(width)
            for distribution, width in zip(self.attention_variational_widths, widths)
        ]
        self.q_attention_probs = probabilities
        return probabilities

    def _global_width_log_prior(self) -> torch.Tensor:
        """Return the global prior over all MLP and attention width rates."""
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
                for distribution in (
                    *self.variational_widths,
                    *self.attention_variational_widths,
                )
            ]
        )
        global_rate = rates.sum().reshape(1)
        return -torch.log(self.total_width_prior_scale) - (
            global_rate - self.total_width_prior_rate_mean
        ).square() / (2.0 * self.total_width_prior_scale.square())

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, object], torch.Tensor, torch.Tensor]:
        """Produce embeddings and diagnostics for both adaptive components."""
        mlp_widths = self.change_width()
        attention_widths = self.instantiated_attention_widths()
        attention_head_widths = self.instantiated_attention_head_widths()
        mlp_probabilities = self._compute_width_probabilities(mlp_widths)
        attention_probabilities = self._compute_attention_probabilities(
            attention_head_widths
        )
        token_representations = self.backbone(
            tokens, mlp_probabilities, attention_probabilities
        )
        protein_embeddings = self._pool_amino_acids(token_representations, tokens)
        diagnostics = {
            "widths": list(mlp_widths),
            "mlp_widths": list(mlp_widths),
            "attention_widths": list(attention_widths),
            "attention_head_widths": list(attention_head_widths),
            "total_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "width_probabilities": [
                probability.detach() for probability in mlp_probabilities
            ],
            "attention_width_probabilities": [
                probability.detach() for probability in attention_probabilities
            ],
            "n_observations": self.n_observations,
            **self._compression_diagnostics(tokens.size(1)),
        }
        return (
            protein_embeddings,
            diagnostics,
            self._theta_log_prior(),
            self._global_width_log_prior(),
        )

    def state_dict(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Return a checkpoint with both adaptive widths materialized."""
        mlp_widths = self.change_width()
        self._compute_width_probabilities(mlp_widths)
        self._compute_attention_probabilities(self.instantiated_attention_head_widths())
        return nn.Module.state_dict(self, *args, **kwargs)

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True
    ) -> Any:
        """Resize both adaptive components before restoring their tensors."""
        distribution_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("variational_widths.", "attention_variational_widths."))
        }
        nn.Module.load_state_dict(self, distribution_state, strict=False)
        mlp_widths = self.change_width()
        result = nn.Module.load_state_dict(self, state_dict, strict=strict)
        self.old_parameters = dict(self.named_parameters())
        self._compute_width_probabilities(mlp_widths)
        self._compute_attention_probabilities(self.instantiated_attention_head_widths())
        return result


__all__ = [
    "AdaptiveESM",
    "DynamicAttention",
    "DynamicAttentionESMBackbone",
    "DynamicAttentionTransformerLayer",
    "DynamicMLP",
    "DynamicTransformerLayer",
]
