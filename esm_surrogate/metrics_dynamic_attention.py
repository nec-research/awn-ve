#        Adaptive-width ESM embedding distillation
#
#   File:     esm_surrogate/metrics_dynamic_attention.py
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

"""Metrics persistence for models with dynamic attention and MLP widths."""

from __future__ import annotations

from typing import Callable

import torch
from mlwiz.training.callback.plotter import Plotter
from mlwiz.training.event.state import State

from esm_surrogate.metrics import AdaptiveWidthMetricsPlotter


class DynamicAttentionMetricsPlotter(AdaptiveWidthMetricsPlotter):
    """Persist query/key and MLP widths as distinct dashboard groups.

    Epoch histories are stored in ``attention_widths`` and ``mlp_widths``.
    With step sampling enabled, the same names are used beneath ``step`` so
    attention and feed-forward curves remain independently selectable.
    """

    _WIDTH_READERS = {
        "attention_widths": "instantiated_attention_widths",
        "mlp_widths": "instantiated_mlp_widths",
    }

    @classmethod
    def _width_groups(cls, model: torch.nn.Module) -> dict[str, list[int]]:
        """Return independently labeled materialized widths from a model."""
        model = getattr(model, "module", model)
        groups = {}
        for group_name, reader_name in cls._WIDTH_READERS.items():
            reader: Callable[[], object] | None = getattr(model, reader_name, None)
            if not callable(reader):
                raise TypeError(
                    "DynamicAttentionMetricsPlotter requires a model with "
                    f"{reader_name}()."
                )
            widths = reader()
            groups[group_name] = [int(width) for width in widths]
        return groups

    @staticmethod
    def _append_sampled_group(
        step_metrics: dict[str, object],
        group_name: str,
        widths: list[int],
        sample_count: int,
    ) -> None:
        """Append one aligned per-layer width sample to a step group."""
        histories = step_metrics.setdefault(group_name, {})
        if not isinstance(histories, dict):
            raise TypeError(f"Expected `step/{group_name}` to be a dictionary.")
        for history in histories.values():
            if not isinstance(history, list):
                raise TypeError("Expected every sampled width history to be a list.")
            history.extend([None] * (sample_count - len(history)))

        for index, width in enumerate(widths, start=1):
            name = f"layer_{index}"
            history = histories.setdefault(name, [None] * sample_count)
            if not isinstance(history, list):
                raise TypeError(
                    f"Expected sampled width history `{group_name}/{name}` "
                    "to be a list."
                )
            history.extend([None] * (sample_count - len(history)))
            history[sample_count - 1] = width

    def _store_sampled_widths(self, state: State) -> None:
        """Store separate attention and MLP histories at a sampled step."""
        if (
            not self.main_process
            or not self.store_on_disk
            or self.store_every_N_steps is None
            or self._training_step % self.store_every_N_steps != 0
        ):
            return

        step_metrics = self.stored_metrics.get("step")
        if not isinstance(step_metrics, dict):
            raise TypeError("Expected the stored `step` metrics to be a dictionary.")
        steps = step_metrics.get("steps")
        if not isinstance(steps, list) or not steps:
            raise RuntimeError("Cannot store sampled widths without a step value.")

        for group_name, widths in self._width_groups(state.model).items():
            self._append_sampled_group(
                step_metrics, group_name, widths, sample_count=len(steps)
            )
        self._store_metrics()

    def on_epoch_end(self, state: State) -> None:
        """Store ordinary metrics, timing, and two labeled width families."""
        Plotter.on_epoch_end(self, state)
        if not self.main_process or not self.store_on_disk:
            return
        if not self._epoch_forward_times_ms:
            raise RuntimeError("Cannot summarize forward time without any batches.")

        mean_forward_time_ms = sum(self._epoch_forward_times_ms) / len(
            self._epoch_forward_times_ms
        )
        self.stored_metrics.setdefault("model_forward_time_ms", []).append(
            mean_forward_time_ms
        )
        for group_name, widths in self._width_groups(state.model).items():
            self.stored_metrics.setdefault(group_name, []).append(widths)

        if self.store_every_N_epochs is not None and (
            (state.epoch + 1) % self.store_every_N_epochs == 0
        ):
            self._store_metrics()
