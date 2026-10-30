#        Adaptive-width ESM embedding distillation
#
#   File:     esm_surrogate/metrics.py
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

"""MLWiz callbacks for adaptive-width and model-forward measurements."""

from __future__ import annotations

import math
from time import perf_counter

import torch
from mlwiz.training.callback.engine_callback import EngineCallback
from mlwiz.training.callback.metric import Metric, Regression
from mlwiz.training.callback.plotter import Plotter
from mlwiz.training.event.state import State


class TimedEngineCallback(EngineCallback):
    """Run the model forward pass and expose its synchronized wall-clock time."""

    @staticmethod
    def _synchronize(device: str | torch.device) -> None:
        """Wait for outstanding CUDA work when timing a CUDA model."""
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def on_forward(self, state: State) -> None:
        """Store model outputs and the completed forward duration in milliseconds."""
        self._synchronize(state.device)
        start = perf_counter()
        super().on_forward(state)
        self._synchronize(state.device)
        state.update(model_forward_time_ms=(perf_counter() - start) * 1_000.0)


class SmoothL1Loss(Regression):
    """MLWiz-compatible wrapper around :class:`torch.nn.SmoothL1Loss`."""

    def __init__(
        self,
        use_as_loss: bool = False,
        reduction: str = "mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
        beta: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
            **kwargs,
        )
        self.metric = torch.nn.SmoothL1Loss(beta=beta, reduction=reduction)


def _root_mean_squared_error(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> torch.Tensor:
    """Return exact RMSE with a finite zero-residual gradient."""
    residual = predictions - targets
    if residual.numel() == 0:
        raise ValueError("RMSE requires at least one prediction.")
    return torch.linalg.vector_norm(residual) / math.sqrt(residual.numel())


class RootMeanSquaredError(Regression):
    """MLWiz-compatible root mean squared error metric."""

    def __init__(self, reduction: str = "mean", **kwargs: object) -> None:
        if reduction != "mean":
            raise ValueError("RootMeanSquaredError only supports reduction='mean'.")
        super().__init__(reduction=reduction, **kwargs)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Return RMSE across all samples and target coordinates."""
        return _root_mean_squared_error(targets, predictions)


class AdaptiveWidthL1Loss(Metric):
    """L1 ELBO loss regularized by global theta and width log priors."""

    def __init__(
        self,
        use_as_loss: bool = False,
        reduction: str = "mean",
        accumulate_over_epoch: bool = False,
        **kwargs: object,
    ) -> None:
        if not use_as_loss:
            raise ValueError(f"{type(self).__name__} can only be used as a loss.")
        if accumulate_over_epoch:
            raise ValueError(
                f"{type(self).__name__} requires accumulate_over_epoch=False "
                "because its priors are evaluated once per batch."
            )
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"{type(self).__name__} reduction must be 'mean' or 'sum'."
            )
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=False,
            **kwargs,
        )

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return student and teacher embeddings for the data-fit term."""
        return outputs[0], targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Return the unregularized embedding L1 term."""
        return torch.nn.functional.l1_loss(
            predictions, targets, reduction=self.reduction
        )

    def forward(
        self, targets: torch.Tensor, *outputs: object
    ) -> dict[str, torch.Tensor]:
        """Compute the dataset-scaled data-fit ELBO and return its negative."""
        if len(outputs) < 4:
            raise ValueError(
                f"{type(self).__name__} expects model outputs containing diagnostics, "
                "theta log prior, and global width log prior."
            )
        predictions = outputs[0]
        diagnostics = outputs[1]
        if not isinstance(diagnostics, dict):
            raise TypeError("AdaptiveESM diagnostics must be a dictionary.")
        n_observations = int(diagnostics.get("n_observations", 0))
        if n_observations < 1:
            raise ValueError("Model diagnostics must contain positive n_observations.")
        theta_log_prior = outputs[2]
        global_width_log_prior = outputs[3]
        if not torch.is_tensor(theta_log_prior) or not torch.is_tensor(
            global_width_log_prior
        ):
            raise TypeError("AdaptiveESM log priors must be tensors.")

        # As in the reference ELBO, treat every minibatch mean as an estimate
        # over the full training set before adding the two global log priors.
        log_p_y = -self.compute_metric(targets, predictions) * n_observations
        elbo = log_p_y + theta_log_prior.sum() + global_width_log_prior.sum()
        elbo = elbo / n_observations
        return {self.name: -elbo}


class AdaptiveWidthMSELoss(AdaptiveWidthL1Loss):
    """MSE data loss regularized by global theta and width log priors."""

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Return the unregularized embedding MSE term."""
        return torch.nn.functional.mse_loss(
            predictions,
            targets,
            reduction=self.reduction,
        )


class CompressionAdaptiveWidthMSELoss(AdaptiveWidthMSELoss):
    """Adaptive MSE objective with a differentiable parameter-budget penalty."""

    def forward(
        self, targets: torch.Tensor, *outputs: object
    ) -> dict[str, torch.Tensor]:
        """Add the configured soft parameter-ratio penalty to the MSE ELBO."""
        result = super().forward(targets, *outputs)
        diagnostics = outputs[1]
        if not isinstance(diagnostics, dict):
            raise TypeError("AdaptiveESM diagnostics must be a dictionary.")
        penalty = diagnostics.get("parameter_compression_penalty")
        if not torch.is_tensor(penalty):
            raise ValueError(
                "CompressionAdaptiveWidthMSELoss requires a differentiable "
                "parameter_compression_penalty diagnostic."
            )
        result[self.name] = result[self.name] + penalty.mean()
        return result


class FLOPsCompressionAdaptiveWidthMSELoss(AdaptiveWidthMSELoss):
    """Adaptive MSE objective with a differentiable FLOPs-budget penalty."""

    def forward(
        self, targets: torch.Tensor, *outputs: object
    ) -> dict[str, torch.Tensor]:
        """Add the configured soft FLOPs-ratio penalty to the MSE ELBO."""
        result = super().forward(targets, *outputs)
        diagnostics = outputs[1]
        if not isinstance(diagnostics, dict):
            raise TypeError("AdaptiveESM diagnostics must be a dictionary.")
        penalty = diagnostics.get("flops_compression_penalty")
        if not torch.is_tensor(penalty):
            raise ValueError(
                "FLOPsCompressionAdaptiveWidthMSELoss requires a differentiable "
                "flops_compression_penalty diagnostic."
            )
        result[self.name] = result[self.name] + penalty.mean()
        return result


class AdaptiveWidthHuberLoss(AdaptiveWidthL1Loss):
    """Huber data loss regularized by global theta and width log priors."""

    def __init__(self, delta: float = 1.0, **kwargs: object) -> None:
        if delta <= 0:
            raise ValueError("AdaptiveWidthHuberLoss delta must be positive.")
        self.delta = float(delta)
        super().__init__(**kwargs)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Return the unregularized embedding Huber term."""
        return torch.nn.functional.huber_loss(
            predictions,
            targets,
            reduction=self.reduction,
            delta=self.delta,
        )


class AdaptiveWidthRMSELoss(AdaptiveWidthL1Loss):
    """RMSE data loss regularized by global theta and width log priors."""

    def __init__(self, reduction: str = "mean", **kwargs: object) -> None:
        if reduction != "mean":
            raise ValueError("AdaptiveWidthRMSELoss only supports reduction='mean'.")
        super().__init__(reduction=reduction, **kwargs)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Return RMSE across the minibatch and all embedding coordinates."""
        return _root_mean_squared_error(targets, predictions)


class ThetaLogPrior(Metric):
    """Track the raw Gaussian log prior over all backbone parameters."""

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the theta log-prior model output."""
        if len(outputs) < 3 or not torch.is_tensor(outputs[2]):
            raise ValueError("ThetaLogPrior requires AdaptiveESM prior outputs.")
        return outputs[2], targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Average the theta log prior across metric observations."""
        del targets
        return predictions.mean()


class GlobalWidthLogPrior(Metric):
    """Track the raw Gaussian log prior over the sum of all width rates."""

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the global width log-prior model output."""
        if len(outputs) < 4 or not torch.is_tensor(outputs[3]):
            raise ValueError("GlobalWidthLogPrior requires AdaptiveESM prior outputs.")
        return outputs[3], targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Average the global width log prior across metric observations."""
        del targets
        return predictions.mean()


class TotalWidth(Metric):
    """Track the sum of the currently materialized transformer FFN widths."""

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and sum the per-block widths from model diagnostics."""
        if len(outputs) < 2 or not isinstance(outputs[1], dict):
            raise ValueError("TotalWidth requires AdaptiveESM diagnostics.")
        widths = outputs[1].get("widths")
        if not isinstance(widths, (list, tuple)) or not widths:
            raise ValueError(
                "AdaptiveESM diagnostics must contain a non-empty widths sequence."
            )
        return (
            torch.tensor(
                [sum(int(width) for width in widths)],
                device=targets.device,
                dtype=torch.float64,
            ),
            targets,
        )

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Average total width across metric observations."""
        del targets
        return predictions.mean()


class TotalParameters(Metric):
    """Track the total number of parameters in the current adaptive model."""

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the current parameter count from model diagnostics."""
        if len(outputs) < 2 or not isinstance(outputs[1], dict):
            raise ValueError("TotalParameters requires AdaptiveESM diagnostics.")
        total_parameters = outputs[1].get("total_parameters")
        if not isinstance(total_parameters, int) or total_parameters < 0:
            raise ValueError(
                "AdaptiveESM diagnostics must contain a non-negative integer "
                "total_parameters value."
            )
        return (
            torch.tensor(
                [total_parameters], device=targets.device, dtype=torch.float64
            ),
            targets,
        )

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Average the parameter count across metric observations."""
        del targets
        return predictions.mean()


class _DifferentiableDiagnosticMetric(Metric):
    """Base metric for scalar differentiable diagnostics from AdaptiveESM."""

    diagnostic_name: str

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the configured tensor without severing its gradient."""
        if len(outputs) < 2 or not isinstance(outputs[1], dict):
            raise ValueError(
                f"{type(self).__name__} requires AdaptiveESM diagnostics."
            )
        value = outputs[1].get(self.diagnostic_name)
        if not torch.is_tensor(value):
            raise ValueError(
                "AdaptiveESM diagnostics must contain tensor "
                f"{self.diagnostic_name!r}."
            )
        return value.reshape(-1), targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Average the scalar diagnostic across metric observations."""
        del targets
        return predictions.mean()


class SoftTotalParameters(_DifferentiableDiagnosticMetric):
    """Track the differentiable total parameter count at the soft widths."""

    diagnostic_name = "soft_total_parameters"


class ParameterCompressionRatio(_DifferentiableDiagnosticMetric):
    """Track soft total parameters relative to the initial architecture."""

    diagnostic_name = "parameter_compression_ratio"


class ParameterCompressionPenalty(_DifferentiableDiagnosticMetric):
    """Track the scaled squared deviation from the configured ratio budget."""

    diagnostic_name = "parameter_compression_penalty"


class SoftFLOPs(_DifferentiableDiagnosticMetric):
    """Track differentiable per-sequence Transformer matmul FLOPs."""

    diagnostic_name = "soft_flops"


class FLOPsCompressionRatio(_DifferentiableDiagnosticMetric):
    """Track soft Transformer FLOPs relative to the initial architecture."""

    diagnostic_name = "flops_compression_ratio"


class FLOPsCompressionPenalty(_DifferentiableDiagnosticMetric):
    """Track the scaled squared deviation from the configured FLOPs budget."""

    diagnostic_name = "flops_compression_penalty"


class AdaptiveWidthMetricsPlotter(Plotter):
    """Persist forward time and transformer FFN width histories.

    ``model_forward_time_ms`` is a scalar epoch history averaged over all
    training and evaluation forwards completed before ``on_epoch_end``.
    ``step/model_forward_time_ms`` contains the individual training-batch
    forward duration at every sampled global training step.
    ``model_widths`` is an ``epochs x transformer blocks`` matrix containing
    the materialized FFN hidden dimension of each block. When step logging is
    enabled, ``step/model_widths/layer_N`` stores the corresponding width at
    every sampled global training step.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._epoch_forward_times_ms: list[float] = []

    @staticmethod
    def _model_widths(model: torch.nn.Module) -> list[int]:
        """Return the materialized FFN hidden width of every transformer block."""
        model = getattr(model, "module", model)
        width_reader = getattr(model, "instantiated_widths", None)
        if not callable(width_reader):
            raise TypeError(
                "AdaptiveWidthMetricsPlotter requires a model with an "
                "instantiated_widths() method."
            )
        return [int(width) for width in width_reader()]

    def on_epoch_start(self, state: State) -> None:
        """Reset forward measurements before processing the next epoch."""
        super().on_epoch_start(state)
        self._epoch_forward_times_ms = []

    def _record_forward_time(self, state: State) -> None:
        """Append the most recent forward duration to the current epoch."""
        if not self.main_process or not self.store_on_disk:
            return
        duration = getattr(state, "model_forward_time_ms", None)
        if duration is None:
            raise RuntimeError(
                "No model forward duration was recorded. Configure "
                "esm_surrogate.metrics.TimedEngineCallback as engine_callback."
            )
        duration = float(duration)
        if duration < 0.0:
            raise ValueError("Model forward duration cannot be negative.")
        self._epoch_forward_times_ms.append(duration)

    def on_training_batch_end(self, state: State) -> None:
        """Record a completed training forward and persist sampled metrics."""
        self._record_forward_time(state)
        super().on_training_batch_end(state)
        self._store_sampled_forward_time()
        self._store_sampled_widths(state)

    def _store_sampled_forward_time(self) -> None:
        """Align the current batch's forward duration with a sampled step."""
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
            raise RuntimeError(
                "Cannot store sampled forward time without a step value."
            )
        if not self._epoch_forward_times_ms:
            raise RuntimeError("Cannot store sampled forward time without a duration.")

        sample_count = len(steps)
        history = step_metrics.setdefault(
            "model_forward_time_ms", [None] * sample_count
        )
        if not isinstance(history, list):
            raise TypeError(
                "Expected `step/model_forward_time_ms` to be a list."
            )
        history.extend([None] * (sample_count - len(history)))
        history[sample_count - 1] = self._epoch_forward_times_ms[-1]

    def _store_sampled_widths(self, state: State) -> None:
        """Align current block widths with a newly persisted step sample."""
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

        sample_count = len(steps)
        width_histories = step_metrics.setdefault("model_widths", {})
        if not isinstance(width_histories, dict):
            raise TypeError("Expected `step/model_widths` to be a dictionary.")

        current_widths = self._model_widths(state.model)
        for history in width_histories.values():
            if not isinstance(history, list):
                raise TypeError("Expected every sampled width history to be a list.")
            history.extend([None] * (sample_count - len(history)))

        for index, width in enumerate(current_widths, start=1):
            name = f"layer_{index}"
            history = width_histories.setdefault(name, [None] * sample_count)
            if not isinstance(history, list):
                raise TypeError(
                    f"Expected sampled width history `{name}` to be a list."
                )
            history.extend([None] * (sample_count - len(history)))
            history[sample_count - 1] = width

        # The base Plotter flushes losses and scores before this custom history
        # is appended. Re-flush the same atomic artifact with aligned widths.
        self._store_metrics()

    def on_eval_batch_end(self, state: State) -> None:
        """Record a completed evaluation forward."""
        self._record_forward_time(state)

    def on_epoch_end(self, state: State) -> None:
        """Store ordinary metrics, mean forward time, and per-block widths."""
        super().on_epoch_end(state)
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
        self.stored_metrics.setdefault("model_widths", []).append(
            self._model_widths(state.model)
        )

        if self.store_every_N_epochs is not None and (
            (state.epoch + 1) % self.store_every_N_epochs == 0
        ):
            # Plotter has already flushed losses/scores. Flush again so the
            # model metrics from this epoch land in the same artifact.
            self._store_metrics()
