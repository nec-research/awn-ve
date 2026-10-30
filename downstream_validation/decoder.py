#        Adaptive-width ESM embedding distillation
#
#   File:     downstream_validation/decoder.py
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

"""Autoregressive MSA sequence decoder conditioned on a latent embedding.

Maps a mean-pooled ESM-2 (or compatible) embedding to a fixed-length MSA
sequence over the 20 canonical amino acids plus gap ``-``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DECODER_RESIDUES = AMINO_ACIDS + "-"  # 21 symbols
BOS_ID = len(DECODER_RESIDUES)  # 21
VOCAB_SIZE = BOS_ID + 1  # 22

_CHAR_TO_ID = {c: i for i, c in enumerate(DECODER_RESIDUES)}
_ID_TO_CHAR = {i: c for c, i in _CHAR_TO_ID.items()}


def tokenize_sequences(seqs: list[str], max_len: int | None = None) -> torch.Tensor:
    """Encode sequences as ``(N, L)`` int64 token ids."""
    if not seqs:
        raise ValueError("No sequences to tokenize.")
    gap_id = _CHAR_TO_ID["-"]
    encoded: list[list[int]] = []
    for s in seqs:
        chars = s[:max_len] if max_len is not None and max_len > 0 else s
        encoded.append([_CHAR_TO_ID.get(ch, gap_id) for ch in chars])
    lengths = {len(row) for row in encoded}
    if len(lengths) != 1:
        raise ValueError(
            f"Decoder expects fixed-length sequences, got lengths {sorted(lengths)}."
        )
    return torch.tensor(encoded, dtype=torch.long)


def detokenize(tokens: torch.Tensor) -> list[str]:
    """Decode ``(N, L)`` or ``(L,)`` token ids to amino-acid / gap strings."""
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    out: list[str] = []
    for row in tokens.detach().cpu().tolist():
        chars = []
        for tid in row:
            if tid == BOS_ID:
                continue
            chars.append(_ID_TO_CHAR.get(int(tid), "-"))
        out.append("".join(chars))
    return out


class SequenceDecoder(nn.Module):
    """Causal Transformer that predicts sequence tokens conditioned on z.

    Latent ``z`` replaces the BOS token embedding (prefix conditioning).
    Teacher-forced training predicts tokens ``0..L-1`` from
    ``[z, t0, ..., t_{L-2}]``.
    """

    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        vocab_size: int = VOCAB_SIZE,
        bos_id: int = BOS_ID,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        self.latent_dim = int(latent_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ff_dim = int(ff_dim)
        self.dropout_p = float(dropout)
        self.vocab_size = int(vocab_size)
        self.bos_id = int(bos_id)

        self.z_proj = nn.Linear(self.latent_dim, self.d_model)
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.seq_len, self.d_model)
        self.emb_dropout = nn.Dropout(self.dropout_p)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.z_proj.weight)
        nn.init.zeros_(self.z_proj.bias)

    def _build_inputs(self, z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be (B, L), got {tuple(tokens.shape)}")
        bsz, length = tokens.shape
        if length != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got tokens length {length}."
            )
        if z.shape[0] != bsz:
            raise ValueError(f"Batch mismatch: z has {z.shape[0]}, tokens have {bsz}.")
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: model={self.latent_dim}, z={z.shape[-1]}."
            )

        z_tok = self.z_proj(z).unsqueeze(1)
        if length == 1:
            x = z_tok
        else:
            x = torch.cat([z_tok, self.token_emb(tokens[:, :-1])], dim=1)

        positions = torch.arange(length, device=tokens.device)
        x = x + self.pos_emb(positions).unsqueeze(0)
        return self.emb_dropout(x)

    def forward(self, z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced pass; returns ``(B, L, vocab)`` next-token logits."""
        x = self._build_inputs(z, tokens)
        length = tokens.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length)
        causal_mask = causal_mask.to(device=tokens.device, dtype=x.dtype)
        h = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.lm_head(self.ln_f(h))

    def _prefix_logits(self, z: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        device = z.device
        z_tok = self.z_proj(z).unsqueeze(1)
        if prefix.numel() == 0:
            x = z_tok
            cur_len = 1
        else:
            x = torch.cat([z_tok, self.token_emb(prefix)], dim=1)
            cur_len = prefix.shape[1] + 1
        positions = torch.arange(cur_len, device=device)
        x = self.emb_dropout(x + self.pos_emb(positions).unsqueeze(0))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(cur_len)
        causal_mask = causal_mask.to(device=device, dtype=x.dtype)
        h = self.transformer(x, mask=causal_mask, is_causal=True)
        logits = self.lm_head(self.ln_f(h))[:, -1, :].clone()
        logits[:, self.bos_id] = -float("inf")
        return logits

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        seq_len: int | None = None,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """Autoregressive decode. Returns ``(B, L)`` token ids."""
        self.eval()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        length = int(seq_len) if seq_len is not None else self.seq_len
        if length > self.seq_len:
            raise ValueError(
                f"Requested seq_len {length} exceeds model seq_len {self.seq_len}."
            )
        bsz = z.shape[0]
        device = z.device
        generated = torch.empty((bsz, 0), dtype=torch.long, device=device)
        for _ in range(length):
            logits = self._prefix_logits(z, generated)
            if greedy or temperature <= 0:
                next_id = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
        return generated
