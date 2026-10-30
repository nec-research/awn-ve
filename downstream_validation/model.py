#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/model.py
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

"""Fixed-width, inference-only implementation of the exported AWN ESM model."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class StaticAttention(nn.Module):
    """The checkpoint's fixed adaptive query/key attention operation."""

    def __init__(self, embed_dim: int, num_heads: int, head_width: int) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_width = int(head_width)
        self.q_proj = nn.Linear(embed_dim, head_width * num_heads)
        self.k_proj = nn.Linear(embed_dim, head_width * num_heads)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scaling = head_width**-0.5
        rotary_dim = head_width - head_width % 2
        self.rot_emb: nn.Module | None
        if rotary_dim:
            from esm.rotary_embedding import RotaryEmbedding

            self.rot_emb = RotaryEmbedding(rotary_dim)
        else:
            self.rot_emb = None
        self.register_buffer("importance", torch.ones(head_width))

    def _apply_rotary(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rot_emb is None:
            return query, key
        rotary_dim = int(self.rot_emb.inv_freq.numel() * 2)
        rotated_query, rotated_key = self.rot_emb(
            query[..., :rotary_dim], key[..., :rotary_dim]
        )
        if rotary_dim == self.head_width:
            return rotated_query, rotated_key
        return (
            torch.cat((rotated_query, query[..., rotary_dim:]), dim=-1),
            torch.cat((rotated_key, key[..., rotary_dim:]), dim=-1),
        )

    def forward(
        self, hidden: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        length, batch_size, _ = hidden.shape
        importance_scale = self.importance.sqrt().to(dtype=hidden.dtype)
        query = self.q_proj(hidden).view(
            length, batch_size, self.head_width, self.num_heads
        ).permute(0, 1, 3, 2)
        key = self.k_proj(hidden).view(
            length, batch_size, self.head_width, self.num_heads
        ).permute(0, 1, 3, 2)
        query = query * importance_scale
        key = key * importance_scale
        value = self.v_proj(hidden)
        query = query.contiguous().view(
            length, batch_size * self.num_heads, self.head_width
        ).transpose(0, 1)
        key = key.contiguous().view(
            length, batch_size * self.num_heads, self.head_width
        ).transpose(0, 1)
        value = value.contiguous().view(
            length, batch_size * self.num_heads, self.embed_dim // self.num_heads
        ).transpose(0, 1)
        query, key = self._apply_rotary(query, key)
        weights = torch.bmm(query * self.scaling, key.transpose(1, 2))
        weights = weights.view(batch_size, self.num_heads, length, length)
        weights = weights.masked_fill(
            padding_mask[:, None, None, :].to(torch.bool), float("-inf")
        ).view(batch_size * self.num_heads, length, length)
        probabilities = F.softmax(weights, dim=-1, dtype=torch.float32).to(weights.dtype)
        attended = torch.bmm(probabilities, value)
        attended = attended.transpose(0, 1).contiguous().view(
            length, batch_size, self.embed_dim
        )
        return self.out_proj(attended)


class StaticMLP(nn.Module):
    """Fixed-width AWN MLP retaining its learned channel probabilities."""

    def __init__(self, embed_dim: int, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(embed_dim, width), nn.Linear(width, embed_dim)])
        self.register_buffer("importance", torch.ones(width))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers[1](F.relu6(self.layers[0](hidden)) * self.importance)


class StaticTransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, attention_heads: int, mlp_width: int, attention_head_width: int) -> None:
        super().__init__()
        from esm.modules import ESM1bLayerNorm

        self.self_attn = StaticAttention(embed_dim, attention_heads, attention_head_width)
        self.self_attn_layer_norm = ESM1bLayerNorm(embed_dim)
        self.final_layer_norm = ESM1bLayerNorm(embed_dim)
        self.dynamic_mlp = StaticMLP(embed_dim, mlp_width)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.self_attn_layer_norm(hidden)
        hidden = residual + self.self_attn(hidden, padding_mask)
        residual = hidden
        hidden = self.final_layer_norm(hidden)
        return residual + self.dynamic_mlp(hidden)


class StaticBackbone(nn.Module):
    def __init__(self, metadata: dict) -> None:
        super().__init__()
        import esm
        from esm.modules import ESM1bLayerNorm

        self.embed_dim = int(metadata["embed_dim"])
        self.token_dropout = bool(metadata["token_dropout"])
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.padding_idx = self.alphabet.padding_idx
        self.mask_idx = self.alphabet.mask_idx
        self.cls_idx = self.alphabet.cls_idx
        self.eos_idx = self.alphabet.eos_idx
        self.embed_tokens = nn.Embedding(len(self.alphabet), self.embed_dim, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([
            StaticTransformerLayer(self.embed_dim, int(metadata["attention_heads"]), int(mlp_width), int(attention_width))
            for mlp_width, attention_width in zip(metadata["mlp_widths"], metadata["attention_head_widths"])
        ])
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        padding_mask = tokens.eq(self.padding_idx)
        hidden = self.embed_tokens(tokens)
        if self.token_dropout:
            hidden = hidden.masked_fill((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            source_lengths = (~padding_mask).sum(-1).clamp_min(1)
            observed_mask_ratio = (tokens == self.mask_idx).sum(-1).to(hidden.dtype) / source_lengths
            hidden = hidden * (1.0 - 0.15 * 0.8) / (1.0 - observed_mask_ratio)[:, None, None]
        hidden = hidden * (1 - padding_mask.unsqueeze(-1).type_as(hidden))
        hidden = hidden.transpose(0, 1)
        for layer in self.layers:
            hidden = layer(hidden, padding_mask)
        return self.emb_layer_norm_after(hidden).transpose(0, 1)


class AdaptiveESMInference(nn.Module):
    """A static exported AWN model returning centered protein embeddings only."""

    def __init__(self, metadata: dict) -> None:
        super().__init__()
        self.metadata = dict(metadata)
        self.backbone = StaticBackbone(self.metadata)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        representations = self.backbone(tokens)
        mask = tokens.ne(self.backbone.padding_idx)
        mask &= tokens.ne(self.backbone.cls_idx)
        mask &= tokens.ne(self.backbone.eos_idx)
        counts = mask.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("Cannot pool a protein with zero amino-acid tokens.")
        return (representations * mask.unsqueeze(-1)).sum(dim=1) / counts.to(representations.dtype)
