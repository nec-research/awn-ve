#        Adaptive-width ESM embedding distillation
#
#   File:     tests/test_metrics.py
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

"""Tests for adaptive-width and model-timing metric callbacks."""

from pathlib import Path

import pytest
import torch
from mlwiz.config_loader import load_experiment_config
from mlwiz.static import LOSSES, SCORES
from mlwiz.training.event.state import State
from mlwiz.ui.dashboard import ResultsRepository

import esm_surrogate.metrics as metrics_module
from esm_surrogate.metrics import AdaptiveWidthMetricsPlotter, TimedEngineCallback
from esm_surrogate.metrics import (
    AdaptiveWidthHuberLoss,
    AdaptiveWidthL1Loss,
    AdaptiveWidthMSELoss,
    AdaptiveWidthRMSELoss,
    GlobalWidthLogPrior,
    RootMeanSquaredError,
    SmoothL1Loss,
    ThetaLogPrior,
    TotalParameters,
    TotalWidth,
)


def test_smooth_l1_loss_wraps_pytorch_loss() -> None:
    """The wrapper should expose Smooth L1 through the MLWiz metric contract."""
    predictions = torch.tensor([[0.0, 3.0]], requires_grad=True)
    targets = torch.zeros_like(predictions)
    loss = SmoothL1Loss(use_as_loss=True, beta=2.0)

    result = loss(targets, predictions)[loss.name]

    assert result.item() == pytest.approx(1.0)
    result.backward()
    assert predictions.grad is not None


def test_root_mean_squared_error_matches_global_rmse() -> None:
    """The reporting metric should aggregate samples and coordinates."""
    predictions = torch.tensor([[3.0, 4.0], [0.0, 0.0]], requires_grad=True)
    targets = torch.zeros_like(predictions)
    metric = RootMeanSquaredError(use_as_loss=False)

    result = metric(targets, predictions)[metric.name]

    assert result.item() == pytest.approx(2.5)
    result.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


class _WidthModel(torch.nn.Module):
    """Minimal model exposing the adaptive-width metric contract."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)

    def instantiated_widths(self) -> list[int]:
        return [4, 7, 5]


def _epoch_state(model: torch.nn.Module) -> State:
    """Build the state fields consumed by the metrics plotter."""
    state = State(model=model, optimizer=None, device="cpu")
    state.update(
        epoch_results={
            LOSSES: {"training_main_loss": torch.tensor(0.5)},
            SCORES: {"training_main_score": torch.tensor(0.5)},
        },
        model_forward_time_ms=0.0,
    )
    return state


def test_timed_engine_callback_records_completed_forward(monkeypatch) -> None:
    """The engine callback should store outputs and elapsed milliseconds."""
    clock = iter((4.0, 4.0125))
    monkeypatch.setattr(metrics_module, "perf_counter", lambda: next(clock))

    model = _WidthModel()
    state = State(model=model, optimizer=None, device="cpu")
    state.update(batch_input=torch.ones(1, 2))
    callback = TimedEngineCallback(store_last_checkpoint=False)

    callback.on_forward(state)

    assert state.batch_outputs.shape == (1, 2)
    assert state.model_forward_time_ms == pytest.approx(12.5)


def test_metrics_plotter_stores_epoch_time_and_transformer_widths(tmp_path) -> None:
    """Forward durations and one width per block should share the epoch artifact."""
    state = _epoch_state(_WidthModel())
    plotter = AdaptiveWidthMetricsPlotter(str(tmp_path))
    plotter.on_epoch_start(state)

    for duration in (3.0, 6.0):
        state.model_forward_time_ms = duration
        plotter.on_training_batch_end(state)
    state.model_forward_time_ms = 12.0
    plotter.on_eval_batch_end(state)
    plotter.on_epoch_end(state)

    stored = torch.load(tmp_path / "metrics_data.torch", weights_only=True)
    assert stored["model_forward_time_ms"] == pytest.approx([7.0])
    assert stored["model_widths"] == [[4, 7, 5]]


def test_metrics_plotter_flushes_batch_steps_before_epoch_end(tmp_path) -> None:
    """Sampled batch histories should be dashboard-ready during a long epoch."""
    state = _epoch_state(_WidthModel())
    state.batch_loss = {"main_loss": torch.tensor(0.5)}
    state.batch_score = {"main_score": torch.tensor(0.4)}
    plotter = AdaptiveWidthMetricsPlotter(str(tmp_path), store_every_N_steps=2)
    plotter.on_epoch_start(state)

    for loss, score in ((0.5, 0.4), (0.3, 0.7)):
        state.batch_loss["main_loss"] = torch.tensor(loss)
        state.batch_score["main_score"] = torch.tensor(score)
        state.model_forward_time_ms = 1.0
        plotter.on_training_batch_end(state)

    stored = torch.load(tmp_path / "metrics_data.torch", weights_only=True)
    assert stored["losses"] == {}
    assert stored["scores"] == {}
    assert stored["step"]["steps"] == [2]
    assert stored["step"]["losses"]["training_main_loss"] == pytest.approx([0.3])
    assert stored["step"]["scores"]["training_main_score"] == pytest.approx([0.7])
    assert stored["step"]["model_forward_time_ms"] == pytest.approx([1.0])
    assert stored["step"]["model_widths"] == {
        "layer_1": [4],
        "layer_2": [7],
        "layer_3": [5],
    }

    details = ResultsRepository(tmp_path.parent).details(tmp_path.name)
    step_series = [item for item in details["series"] if item["unit"] == "step"]
    assert {
        item["name"] for item in step_series if item["group"] != "model_widths"
    } == {
        "training_main_loss",
        "training_main_score",
        "model_forward_time_ms",
    }
    width_series = [item for item in step_series if item["group"] == "model_widths"]
    assert {item["name"] for item in width_series} == {
        "layer_1",
        "layer_2",
        "layer_3",
    }
    assert {tuple(item["x_values"]) for item in step_series} == {(2.0,)}


def test_metrics_plotter_backfills_width_gaps_when_step_history_exists(
    tmp_path,
) -> None:
    """Width curves added after a resume should retain the global step axis."""
    torch.save(
        {
            "losses": {},
            "scores": {},
            "step": {
                "steps": [2, 4],
                "last_step": 4,
                "losses": {"training_main_loss": [0.8, 0.6]},
                "scores": {"training_main_score": [0.2, 0.4]},
            },
        },
        tmp_path / "metrics_data.torch",
    )
    state = _epoch_state(_WidthModel())
    state.batch_loss = {"main_loss": torch.tensor(0.5)}
    state.batch_score = {"main_score": torch.tensor(0.6)}
    state.model_forward_time_ms = 1.0
    plotter = AdaptiveWidthMetricsPlotter(str(tmp_path), store_every_N_steps=2)

    plotter.on_training_batch_end(state)
    plotter.on_training_batch_end(state)

    stored = torch.load(tmp_path / "metrics_data.torch", weights_only=True)
    assert stored["step"]["steps"] == [2, 4, 6]
    assert stored["step"]["model_forward_time_ms"] == [None, None, 1.0]
    assert stored["step"]["model_widths"]["layer_1"] == [None, None, 4]

    details = ResultsRepository(tmp_path.parent).details(tmp_path.name)
    layer = next(
        item
        for item in details["series"]
        if item["unit"] == "step"
        and item["group"] == "model_widths"
        and item["name"] == "layer_1"
    )
    assert layer["x_values"] == pytest.approx([2, 4, 6])
    assert layer["values"] == [None, None, 4.0]


def test_metrics_plotter_requires_timed_engine_callback(tmp_path) -> None:
    """A missing timing callback should fail with an actionable message."""
    state = _epoch_state(_WidthModel())
    del state.model_forward_time_ms
    plotter = AdaptiveWidthMetricsPlotter(str(tmp_path))

    with pytest.raises(RuntimeError, match="TimedEngineCallback"):
        plotter.on_training_batch_end(state)


def test_regularized_l1_loss_matches_dataset_scaled_elbo() -> None:
    """The loss should scale L1 by dataset size before adding global priors."""
    predictions = torch.tensor([[1.0, 3.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0]])
    theta_log_prior = torch.tensor([-8.0], requires_grad=True)
    width_log_prior = torch.tensor([-2.0], requires_grad=True)
    loss = AdaptiveWidthL1Loss(
        use_as_loss=True,
        reduction="mean",
        accumulate_over_epoch=False,
    )

    result = loss(
        targets,
        predictions,
        {"n_observations": 5},
        theta_log_prior,
        width_log_prior,
    )[loss.name]
    larger_dataset_result = loss(
        targets,
        predictions,
        {"n_observations": 10},
        theta_log_prior,
        width_log_prior,
    )[loss.name]

    assert result.item() == pytest.approx(3.5)  # mean L1 1.5 + prior penalty 2.0
    assert larger_dataset_result.item() == pytest.approx(2.5)
    result.backward()
    assert theta_log_prior.grad.item() == pytest.approx(-0.2)
    assert width_log_prior.grad.item() == pytest.approx(-0.2)


def test_regularized_mse_loss_matches_dataset_scaled_elbo() -> None:
    """Adaptive MSE should retain the existing global-prior scaling."""
    predictions = torch.tensor([[1.0, 3.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0]])
    theta_log_prior = torch.tensor([-8.0], requires_grad=True)
    width_log_prior = torch.tensor([-2.0], requires_grad=True)
    loss = AdaptiveWidthMSELoss(
        use_as_loss=True,
        reduction="mean",
        accumulate_over_epoch=False,
    )

    result = loss(
        targets,
        predictions,
        {"n_observations": 5},
        theta_log_prior,
        width_log_prior,
    )[loss.name]

    assert result.item() == pytest.approx(4.5)  # mean MSE 2.5 + prior penalty 2.0
    result.backward()
    assert torch.equal(predictions.grad, torch.tensor([[1.0, 2.0]]))
    assert theta_log_prior.grad.item() == pytest.approx(-0.2)
    assert width_log_prior.grad.item() == pytest.approx(-0.2)


def test_regularized_huber_loss_matches_dataset_scaled_elbo() -> None:
    """The adaptive Huber loss should retain the existing prior scaling."""
    predictions = torch.tensor([[0.25, 2.0]], requires_grad=True)
    targets = torch.zeros_like(predictions)
    theta_log_prior = torch.tensor([-8.0], requires_grad=True)
    width_log_prior = torch.tensor([-2.0], requires_grad=True)
    loss = AdaptiveWidthHuberLoss(
        use_as_loss=True,
        delta=0.5,
        reduction="mean",
        accumulate_over_epoch=False,
    )

    result = loss(
        targets,
        predictions,
        {"n_observations": 5},
        theta_log_prior,
        width_log_prior,
    )[loss.name]

    assert result.item() == pytest.approx(2.453125)
    result.backward()
    assert predictions.grad is not None
    assert theta_log_prior.grad.item() == pytest.approx(-0.2)
    assert width_log_prior.grad.item() == pytest.approx(-0.2)


def test_regularized_rmse_loss_matches_dataset_scaled_elbo() -> None:
    """Adaptive RMSE should retain the existing global-prior scaling."""
    predictions = torch.tensor([[3.0, 4.0]], requires_grad=True)
    targets = torch.zeros_like(predictions)
    theta_log_prior = torch.tensor([-8.0], requires_grad=True)
    width_log_prior = torch.tensor([-2.0], requires_grad=True)
    loss = AdaptiveWidthRMSELoss(
        use_as_loss=True,
        reduction="mean",
        accumulate_over_epoch=False,
    )

    result = loss(
        targets,
        predictions,
        {"n_observations": 5},
        theta_log_prior,
        width_log_prior,
    )[loss.name]

    assert result.item() == pytest.approx((12.5**0.5) + 2.0)
    result.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()
    assert theta_log_prior.grad.item() == pytest.approx(-0.2)
    assert width_log_prior.grad.item() == pytest.approx(-0.2)


def test_regularized_rmse_has_finite_gradient_at_exact_match() -> None:
    """An exact prediction should have zero loss and a stable zero gradient."""
    predictions = torch.zeros(1, 3, requires_grad=True)
    targets = torch.zeros_like(predictions)
    theta_log_prior = torch.zeros(1, requires_grad=True)
    width_log_prior = torch.zeros(1, requires_grad=True)
    loss = AdaptiveWidthRMSELoss(
        use_as_loss=True,
        accumulate_over_epoch=False,
    )

    result = loss(
        targets,
        predictions,
        {"n_observations": 5},
        theta_log_prior,
        width_log_prior,
    )[loss.name]
    result.backward()

    assert result.item() == pytest.approx(0.0)
    assert torch.equal(predictions.grad, torch.zeros_like(predictions))
    assert torch.isfinite(predictions.grad).all()


def test_centered_experiment_uses_mse_for_training_and_rmse_selection() -> None:
    """The centered experiment should train with MSE and select by RMSE."""
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "experiment_mean_centered_targets.yml"
    )
    config = load_experiment_config(config_path)

    assert config["grid"]["loss"] == "esm_surrogate.metrics.AdaptiveWidthMSELoss"
    assert (
        config["grid"]["scorer"]["args"]["main_scorer"]
        == "esm_surrogate.metrics.RootMeanSquaredError"
    )


@pytest.mark.parametrize(
    ("metric_class", "output_index", "expected"),
    [
        (ThetaLogPrior, 2, -7.0),
        (GlobalWidthLogPrior, 3, -1.5),
    ],
)
def test_prior_metrics_extract_raw_model_terms(
    metric_class: type, output_index: int, expected: float
) -> None:
    """Reference-style prior metrics should expose the corresponding raw term."""
    outputs: list[object] = [torch.zeros(1, 2), {}]
    outputs.extend([torch.tensor([-7.0]), torch.tensor([-1.5])])
    metric = metric_class(use_as_loss=False, accumulate_over_epoch=False)

    predictions, metric_targets = metric.get_predictions_and_targets(
        torch.zeros(1, 2), *outputs
    )

    assert predictions is outputs[output_index]
    assert metric.compute_metric(metric_targets, predictions).item() == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("metric_class", "expected"),
    [
        (TotalWidth, 16.0),
        (TotalParameters, 1234.0),
    ],
)
def test_model_size_metrics_extract_diagnostics(
    metric_class: type, expected: float
) -> None:
    """Model-size metrics should expose current widths and parameter count."""
    targets = torch.zeros(1, 2)
    outputs = [
        torch.zeros(1, 2),
        {"widths": [4, 7, 5], "total_parameters": 1234},
    ]
    metric = metric_class(use_as_loss=False, accumulate_over_epoch=False)

    predictions, metric_targets = metric.get_predictions_and_targets(
        targets, *outputs
    )

    assert metric.compute_metric(metric_targets, predictions).item() == pytest.approx(
        expected
    )
