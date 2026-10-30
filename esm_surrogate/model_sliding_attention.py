#        Adaptive-width ESM embedding distillation
#
#   File:     esm_surrogate/model_sliding_attention.py
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

"""Checkpoint-compatible adaptive ESM with efficient sliding-window attention.

The dense implementation in :mod:`esm_surrogate.model` remains unchanged.  This
module reuses its projections, adaptive-width distributions, optimizer
migration, checkpoint layout, and pooling while replacing only the attention
score computation.  CUDA inference and training use a compiled FlexAttention
kernel with a cached block-sparse mask.  Correct dense local attention is kept
as a fallback for CPU execution, attention-weight inspection, and nonzero
attention dropout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from esm_surrogate.model import (
    AdaptiveESM as DenseAdaptiveESM,
)
from esm_surrogate.model import (
    DynamicAttention,
    DynamicAttentionESMBackbone,
    DynamicAttentionTransformerLayer,
    DynamicMLP,
    DynamicTransformerLayer,
)

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )
except ImportError as error:  # pragma: no cover - depends on installed PyTorch
    raise ImportError(
        "Sliding-window attention requires torch.nn.attention.flex_attention."
    ) from error


# PyTorch's eager FlexAttention fallback materializes dense scores. Dynamic
# block-mask metadata lets one compiled callable serve every local window; the
# adaptive head dimension still specializes into a small number of power-of-two
# buckets. Tiny block sizes need the static compiler path because Triton's
# dynamically selected tile may be larger than the BlockMask tile.
_compiled_flex_attention_dynamic = torch.compile(flex_attention, dynamic=True)
_compiled_flex_attention_static = torch.compile(flex_attention, dynamic=False)
_BLOCK_MASK_CACHE: dict[tuple[str, int | None, int, int, int, int], BlockMask] = {}


def clear_sliding_window_mask_cache() -> None:
    """Release cached FlexAttention block masks."""
    _BLOCK_MASK_CACHE.clear()


def _validate_window_radius(window_radius: int) -> int:
    """Return a validated symmetric sliding-window radius."""
    window_radius = int(window_radius)
    if window_radius < 0:
        raise ValueError("attention_window_radius must be non-negative.")
    return window_radius


def _validate_block_size(block_size: int) -> int:
    """Return a validated FlexAttention block size."""
    block_size = int(block_size)
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError(
            "attention_block_size must be a power of two greater than or equal to 16."
        )
    return block_size


def _sliding_block_mask(
    *,
    query_length: int,
    key_length: int,
    window_radius: int,
    block_size: int,
    device: torch.device,
) -> BlockMask:
    """Return a cached head- and batch-independent symmetric local mask."""
    cache_key = (
        device.type,
        device.index,
        int(query_length),
        int(key_length),
        int(window_radius),
        int(block_size),
    )
    cached = _BLOCK_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Capturing a tensor rather than a Python integer prevents torch.compile
    # from specializing on every tested radius. BlockMask data may change
    # without recompiling the FlexAttention kernel.
    radius_tensor = torch.tensor(window_radius, dtype=torch.int32, device=device)

    def sliding_window_mask(
        batch_index: torch.Tensor,
        head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        del batch_index, head_index
        return (query_index - key_index).abs() <= radius_tensor

    block_mask = create_block_mask(
        sliding_window_mask,
        B=None,
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=device,
        BLOCK_SIZE=block_size,
        # Block-mask construction is otherwise noticeably expensive. The mask
        # is also cached and shared by every transformer layer.
        _compile=device.type == "cuda",
    )
    _BLOCK_MASK_CACHE[cache_key] = block_mask
    return block_mask


class SlidingWindowDynamicAttention(DynamicAttention):
    """Adaptive Q/K attention whose score computation is block-sparse and local.

    Query/key projection widths remain jointly adaptive per head. Value/output
    projections, rotary embeddings, importance weighting, padding semantics,
    optimizer migration, and checkpoint tensor names are inherited unchanged
    from :class:`esm_surrogate.model.DynamicAttention`.
    """

    def __init__(
        self,
        source_attention: DynamicAttention,
        window_radius: int,
        block_size: int = 128,
    ) -> None:
        """Wrap an initialized dense attention module without changing weights."""
        nn.Module.__init__(self)
        self.embed_dim = source_attention.embed_dim
        self.num_heads = source_attention.num_heads
        self.dropout = source_attention.dropout
        self.use_rotary_embeddings = source_attention.use_rotary_embeddings
        self.q_proj = source_attention.q_proj
        self.k_proj = source_attention.k_proj
        self.v_proj = source_attention.v_proj
        self.out_proj = source_attention.out_proj
        self.head_dim = source_attention.head_dim
        self.scaling = source_attention.scaling
        self.rot_emb = source_attention.rot_emb
        self.window_radius = _validate_window_radius(window_radius)
        self.block_size = _validate_block_size(block_size)

    @property
    def window_size(self) -> int:
        """Return the nominal number of visible tokens away from boundaries."""
        return 2 * self.window_radius + 1

    def set_window_radius(self, window_radius: int) -> None:
        """Change the local radius without modifying parameters or checkpoints."""
        self.window_radius = _validate_window_radius(window_radius)

    def _combined_dense_mask(
        self,
        *,
        target_length: int,
        source_length: int,
        reference: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Combine an optional additive mask with the symmetric local mask."""
        if self.window_radius >= max(target_length, source_length) - 1:
            return attn_mask
        query_positions = torch.arange(target_length, device=reference.device)
        key_positions = torch.arange(source_length, device=reference.device)
        is_local = (
            query_positions[:, None] - key_positions[None, :]
        ).abs() <= self.window_radius
        local_bias = torch.zeros(
            (target_length, source_length),
            dtype=reference.dtype,
            device=reference.device,
        ).masked_fill(~is_local, float("-inf"))
        if attn_mask is None:
            return local_bias
        if tuple(attn_mask.shape) != (target_length, source_length):
            raise ValueError(
                "attn_mask must have shape [target_length, source_length]."
            )
        return local_bias + attn_mask.to(dtype=reference.dtype)

    def _dense_window_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
        need_weights: bool,
        need_head_weights: bool,
        attn_mask: torch.Tensor | None,
        attention_importance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the reference dense computation with an exact local mask."""
        combined_mask = self._combined_dense_mask(
            target_length=query.size(0),
            source_length=key.size(0),
            reference=query,
            attn_mask=attn_mask,
        )
        attended, weights = super().forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_head_weights=need_head_weights,
            attn_mask=combined_mask,
            attention_importance=attention_importance,
        )
        if key_padding_mask is not None and query.size(0) == key.size(0):
            # A padded query far beyond a short sequence can have no local,
            # non-padding key. Zero that unused self-attention row so a dense
            # all-masked softmax cannot leak NaNs into final masked pooling.
            attended = attended.masked_fill(
                key_padding_mask.transpose(0, 1).unsqueeze(-1),
                0.0,
            )
        return attended, weights

    @staticmethod
    def _pad_flex_head_dimension(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad head widths into stable power-of-two FlexAttention buckets."""
        original_width = tensor.size(-1)
        kernel_width = max(16, 1 << (original_width - 1).bit_length())
        if original_width == kernel_width:
            return tensor, original_width
        return (
            torch.nn.functional.pad(tensor, (0, kernel_width - original_width)),
            original_width,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        attention_importance: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply adaptive sliding-window multi-head self-attention."""
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unsupported SlidingWindowDynamicAttention arguments: {unsupported}."
            )
        if attention_importance is None:
            raise ValueError(
                "SlidingWindowDynamicAttention requires attention_importance."
            )
        if (
            attention_importance.ndim != 1
            or attention_importance.numel() != self.head_width
        ):
            raise ValueError(
                f"Expected {self.head_width} attention-importance values, found "
                f"shape {tuple(attention_importance.shape)}."
            )

        # FlexAttention does not return probabilities and intentionally omits
        # dropout. Keep a semantically complete dense local path for those less
        # common modes and for CPU/dashboard execution.
        if (
            query.device.type != "cuda"
            or need_weights
            or need_head_weights
            or (self.training and self.dropout > 0.0)
        ):
            return self._dense_window_forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_head_weights=need_head_weights,
                attn_mask=attn_mask,
                attention_importance=attention_importance,
            )

        target_length, batch_size, embed_dim = query.shape
        source_length = key.size(0)
        if embed_dim != self.embed_dim or value.size(-1) != self.embed_dim:
            raise ValueError(
                f"SlidingWindowDynamicAttention expects embedding dimension "
                f"{self.embed_dim}."
            )
        if key_padding_mask is not None and tuple(key_padding_mask.shape) != (
            batch_size,
            source_length,
        ):
            raise ValueError("key_padding_mask must have shape [batch, source_length].")
        if attn_mask is not None and tuple(attn_mask.shape) != (
            target_length,
            source_length,
        ):
            raise ValueError(
                "attn_mask must have shape [target_length, source_length]."
            )

        # Splitting importance equally across q and k applies it exactly once to
        # their dot product, matching the dense adaptive implementation.
        importance_scale = attention_importance.sqrt().to(dtype=query.dtype)
        projected_query = self.q_proj(query).view(
            target_length, batch_size, self.head_dim, self.num_heads
        )
        projected_key = self.k_proj(key).view(
            source_length, batch_size, self.head_dim, self.num_heads
        )
        projected_query = projected_query.permute(1, 3, 0, 2)
        projected_key = projected_key.permute(1, 3, 0, 2)
        projected_query = projected_query * importance_scale
        projected_key = projected_key * importance_scale

        value_head_dim = self.embed_dim // self.num_heads
        projected_value = (
            self.v_proj(value)
            .view(source_length, batch_size, self.num_heads, value_head_dim)
            .permute(1, 2, 0, 3)
        )
        projected_query, projected_key = self._apply_rotary(
            projected_query, projected_key
        )
        # Keep FlexAttention's explicit scale constant across independently
        # adaptive layers. Passing each Python float scale into torch.compile
        # would specialize and eventually exhaust Dynamo's recompile cache.
        projected_query = projected_query * self.scaling

        # Zero-padding into power-of-two buckets retains exact scores/outputs,
        # satisfies the CUDA kernel's E/Ev >= 16 requirement, and prevents every
        # one-neuron adaptive-width change from compiling a distinct kernel.
        flex_query, _ = self._pad_flex_head_dimension(projected_query.contiguous())
        flex_key, _ = self._pad_flex_head_dimension(projected_key.contiguous())
        flex_value, original_value_width = self._pad_flex_head_dimension(
            projected_value.contiguous()
        )
        block_mask = _sliding_block_mask(
            query_length=target_length,
            key_length=source_length,
            window_radius=self.window_radius,
            block_size=self.block_size,
            device=query.device,
        )

        score_mod = None
        if key_padding_mask is not None or attn_mask is not None:
            padding_mask = key_padding_mask
            additive_mask = attn_mask

            def apply_input_masks(
                score: torch.Tensor,
                batch_index: torch.Tensor,
                head_index: torch.Tensor,
                query_index: torch.Tensor,
                key_index: torch.Tensor,
            ) -> torch.Tensor:
                del head_index
                if additive_mask is not None:
                    score = score + additive_mask[query_index, key_index]
                if padding_mask is not None:
                    score = torch.where(
                        padding_mask[batch_index, key_index],
                        -float("inf"),
                        score,
                    )
                return score

            score_mod = apply_input_masks

        compiled_flex_attention = (
            _compiled_flex_attention_dynamic
            if self.block_size >= 64
            else _compiled_flex_attention_static
        )
        # Inductor may otherwise select a 128-token kernel tile for narrower
        # adaptive Q/K buckets even when the sparse BlockMask uses 64-token
        # blocks. FlexAttention requires each sparse block to be divisible by
        # its kernel tile.
        kernel_tile_size = min(self.block_size, 64)
        attended = compiled_flex_attention(
            flex_query,
            flex_key,
            flex_value,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=1.0,
            kernel_options={
                "BLOCK_M": kernel_tile_size,
                "BLOCK_N": kernel_tile_size,
            },
        )[..., :original_value_width]
        attended = (
            attended.permute(2, 0, 1, 3)
            .contiguous()
            .view(target_length, batch_size, self.embed_dim)
        )
        attended = self.out_proj(attended)
        if key_padding_mask is not None and target_length == source_length:
            attended = attended.masked_fill(
                key_padding_mask.transpose(0, 1).unsqueeze(-1),
                0.0,
            )
        return attended, None


class SlidingWindowDynamicAttentionTransformerLayer(DynamicAttentionTransformerLayer):
    """Transformer block that avoids requesting discarded attention weights."""

    def __init__(
        self,
        source_layer: DynamicAttentionTransformerLayer,
        window_radius: int,
        block_size: int,
    ) -> None:
        """Wrap an adaptive transformer layer without changing its parameters."""
        nn.Module.__init__(self)
        self.embed_dim = source_layer.embed_dim
        self.attention_heads = source_layer.attention_heads
        self.use_rotary_embeddings = source_layer.use_rotary_embeddings
        self.self_attn = SlidingWindowDynamicAttention(
            source_layer.self_attn,
            window_radius=window_radius,
            block_size=block_size,
        )
        self.self_attn_layer_norm = source_layer.self_attn_layer_norm
        self.final_layer_norm = source_layer.final_layer_norm
        self.dynamic_mlp = source_layer.dynamic_mlp

    def forward(
        self,
        hidden: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
        need_head_weights: bool = False,
        neuron_importance: torch.Tensor | None = None,
        attention_importance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run local adaptive attention followed by the adaptive MLP."""
        if neuron_importance is None:
            raise ValueError(
                "SlidingWindowDynamicAttentionTransformerLayer requires "
                "neuron_importance."
            )

        residual = hidden
        hidden = self.self_attn_layer_norm(hidden)
        hidden, attention = self.self_attn(
            query=hidden,
            key=hidden,
            value=hidden,
            key_padding_mask=self_attn_padding_mask,
            # The backbone discards weights. Avoiding them is what permits the
            # fused sparse path; explicit head-weight inspection remains valid.
            need_weights=need_head_weights,
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


class AdaptiveESM(DenseAdaptiveESM):
    """Adaptive ESM using efficient symmetric sliding-window self-attention.

    The class deliberately has the same public name and checkpoint state layout
    as ``esm_surrogate.model.AdaptiveESM`` so an existing MLWiz checkpoint can be
    loaded strictly. Configure ``attention_window_radius`` (default ``128``) and
    optionally ``attention_block_size`` (default ``128``).
    """

    def __init__(
        self, dim_input_features: int, dim_target: int, config: Mapping[str, Any]
    ) -> None:
        config = dict(config)
        window_radius = _validate_window_radius(
            int(config.get("attention_window_radius", 128))
        )
        block_size = _validate_block_size(int(config.get("attention_block_size", 128)))
        super().__init__(dim_input_features, dim_target, config)
        self.backbone.layers = nn.ModuleList(
            [
                SlidingWindowDynamicAttentionTransformerLayer(
                    layer,
                    window_radius=window_radius,
                    block_size=block_size,
                )
                for layer in self.backbone.layers
            ]
        )
        self.attention_window_radius = window_radius
        self.attention_block_size = block_size
        # Wrapping retains the exact Parameter objects and names, but refresh the
        # mapping explicitly for future adaptive-width optimizer migration.
        self.old_parameters = dict(self.named_parameters())

    @property
    def attention_window_size(self) -> int:
        """Return the nominal number of visible tokens in every local layer."""
        return 2 * self.attention_window_radius + 1

    def set_attention_window_radius(self, window_radius: int) -> None:
        """Set one symmetric local radius on every transformer layer."""
        window_radius = _validate_window_radius(window_radius)
        for layer in self.backbone.layers:
            layer.self_attn.set_window_radius(window_radius)
        self.attention_window_radius = window_radius


SlidingWindowAdaptiveESM = AdaptiveESM


__all__ = [
    "AdaptiveESM",
    "DenseAdaptiveESM",
    "DynamicAttentionESMBackbone",
    "DynamicAttentionTransformerLayer",
    "DynamicMLP",
    "DynamicTransformerLayer",
    "SlidingWindowAdaptiveESM",
    "SlidingWindowDynamicAttention",
    "SlidingWindowDynamicAttentionTransformerLayer",
    "clear_sliding_window_mask_cache",
]
