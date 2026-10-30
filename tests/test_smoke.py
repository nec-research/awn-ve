#        Adaptive-width ESM embedding distillation
#
#   File:     tests/test_smoke.py
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

"""Small end-to-end MLWiz distillation smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from esm_surrogate.baseline import DynamicMLPBaseline
from esm_surrogate.data import (
    ESMProteinDataLoader,
    ProteinEmbeddingDataset,
    label_to_filename,
)
from esm_surrogate.experiment import AdaptiveWidthExperiment
from esm_surrogate.model import AdaptiveESM


def _write_example_data(
    root: Path,
    *,
    target_normalization: str = "standard",
) -> ProteinEmbeddingDataset:
    fasta = root / "tiny.fasta"
    fasta.write_text(">p1\nACDE\n>p2\nMKT\n", encoding="utf-8")
    teacher_dir = root / "teachers"
    teacher_dir.mkdir()
    for index, label in enumerate(("p1", "p2")):
        torch.save(
            {
                "label": label,
                "encodings": torch.full((8,), float(index) / 2.0),
            },
            teacher_dir / f"{label_to_filename(label)}.pt",
        )
    dataset = ProteinEmbeddingDataset(
        storage_folder=str(root / "data"),
        fasta_path=str(fasta),
        teacher_embeddings_dir=str(teacher_dir),
        max_sequence_length=8,
    )
    dataset.set_target_normalization(
        range(len(dataset)),
        mode=target_normalization,
    )
    return dataset


@pytest.mark.parametrize(
    ("target_normalization", "loss_class", "scorer_class"),
    [
        (
            "standard",
            "esm_surrogate.metrics.AdaptiveWidthL1Loss",
            "mlwiz.training.callback.metric.MeanAbsoluteError",
        ),
        (
            "center",
            "esm_surrogate.metrics.AdaptiveWidthMSELoss",
            "esm_surrogate.metrics.RootMeanSquaredError",
        ),
    ],
)
def test_one_epoch_runs_through_mlwiz(
    tmp_path: Path,
    target_normalization: str,
    loss_class: str,
    scorer_class: str,
) -> None:
    """Both target/loss variants should train through the complete MLWiz path."""
    dataset = _write_example_data(
        tmp_path,
        target_normalization=target_normalization,
    )
    loader = ESMProteinDataLoader(
        dataset, batch_size=2, max_sequence_length=8, shuffle=False
    )
    model_config = {
        "architecture": "esm2",
        "num_layers": 1,
        "embed_dim": 8,
        "attention_heads": 2,
        "token_dropout": False,
        "activation": "torch.nn.functional.gelu",
        "quantile": 0.9,
        "n_observations": len(dataset),
        "theta_prior_scale": 1.0,
        "total_width_prior_mean": 3,
        "total_width_prior_scale": 1.0,
        "truncated_distribution": {
            "class_name": "distribution.TruncatedDistribution",
            "args": {
                "discretized_distribution": {
                    "class_name": "distribution.DiscretizedDistribution",
                    "args": {
                        "base_distribution": {
                            "class_name": "distribution.Exponential",
                            "args": {"starting_size": 3, "max_size": 6},
                        }
                    },
                }
            },
        },
        "attention_truncated_distribution": {
            "class_name": "distribution.TruncatedDistribution",
            "args": {
                "discretized_distribution": {
                    "class_name": "distribution.DiscretizedDistribution",
                    "args": {
                        "base_distribution": {
                            "class_name": "distribution.Exponential",
                            "args": {"starting_size": 2, "max_size": 4},
                        }
                    },
                }
            },
        },
    }
    configuration = {
        **model_config,
        "device": "cpu",
        "evaluate_every": 1,
        "optimizer": {
            "class_name": "mlwiz.training.callback.optimizer.Optimizer",
            "args": {
                "optimizer_class_name": "torch.optim.Adam",
                "lr": 1e-3,
            },
        },
        "loss": {
            "class_name": loss_class,
            "args": {"accumulate_over_epoch": False},
        },
        "scorer": {
            "class_name": "mlwiz.training.callback.metric.MultiScore",
            "args": {
                "accumulate_over_epoch": False,
                "main_scorer": {
                    "class_name": scorer_class,
                    "args": {"accumulate_over_epoch": False},
                },
                "theta_log_prior": {
                    "class_name": "esm_surrogate.metrics.ThetaLogPrior",
                    "args": {},
                },
                "global_width_log_prior": {
                    "class_name": "esm_surrogate.metrics.GlobalWidthLogPrior",
                    "args": {},
                },
                "total_width": {
                    "class_name": "esm_surrogate.metrics.TotalWidth",
                    "args": {},
                },
                "total_parameters": {
                    "class_name": "esm_surrogate.metrics.TotalParameters",
                    "args": {},
                },
            },
        },
        "engine": {
            "class_name": "mlwiz.training.engine.TrainingEngine",
            "args": {"engine_callback": "esm_surrogate.metrics.TimedEngineCallback"},
        },
        "gradient_clipper": None,
        "early_stopper": None,
        "plotter": {
            "class_name": (
                "esm_surrogate.metrics_dynamic_attention.DynamicAttentionMetricsPlotter"
            ),
            "args": {"store_every_N_epochs": 1},
        },
        "checkpoint": False,
    }
    model = AdaptiveESM(8, dataset.dim_target, configuration)
    experiment = AdaptiveWidthExperiment(
        configuration, exp_path=str(tmp_path / "run"), exp_seed=4
    )
    engine = experiment.create_engine(experiment.model_config, model)
    results = engine.train(
        train_loader=loader,
        validation_loader=loader,
        max_epochs=1,
        training_timeout_seconds=-1,
    )

    train_loss, train_score, validation_loss, validation_score, _, _ = results
    assert torch.isfinite(train_loss["main_loss"])
    assert torch.isfinite(train_score["main_score"])
    assert torch.isfinite(validation_loss["main_loss"])
    assert torch.isfinite(validation_score["main_score"])
    assert model.torch_optimizer is engine.optimizer.optimizer
    assert model.n_observations == len(loader.dataset)

    metrics = torch.load(tmp_path / "run" / "metrics_data.torch", weights_only=True)
    assert len(metrics["model_forward_time_ms"]) == 1
    assert metrics["model_forward_time_ms"][0] > 0.0
    assert metrics["mlp_widths"] == [model.instantiated_mlp_widths()]
    assert metrics["attention_widths"] == [model.instantiated_attention_widths()]
    assert len(metrics["scores"]["training_ThetaLogPrior"]) == 1
    assert len(metrics["scores"]["training_GlobalWidthLogPrior"]) == 1
    assert metrics["scores"]["training_TotalWidth"] == [
        sum(model.instantiated_widths())
    ]
    assert metrics["scores"]["training_TotalParameters"] == [
        sum(parameter.numel() for parameter in model.parameters())
    ]


def test_dynamic_mlp_baseline_runs_one_epoch_through_mlwiz(tmp_path: Path) -> None:
    """Token flattening, direct embedding regression, and MLWiz should run."""
    dataset = _write_example_data(tmp_path)
    loader = ESMProteinDataLoader(
        dataset, batch_size=2, max_sequence_length=8, shuffle=False
    )
    model_config = {
        "num_hidden_layers": 2,
        "token_embedding_dim": 4,
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
                            "args": {"starting_size": 3, "max_size": 6},
                        }
                    },
                }
            },
        },
    }
    configuration = {
        **model_config,
        "device": "cpu",
        "evaluate_every": 1,
        "optimizer": {
            "class_name": "mlwiz.training.callback.optimizer.Optimizer",
            "args": {
                "optimizer_class_name": "torch.optim.Adam",
                "lr": 1e-3,
            },
        },
        "loss": {
            "class_name": "esm_surrogate.metrics.SmoothL1Loss",
            "args": {"accumulate_over_epoch": False},
        },
        "scorer": {
            "class_name": "mlwiz.training.callback.metric.MultiScore",
            "args": {
                "accumulate_over_epoch": False,
                "main_scorer": {
                    "class_name": ("mlwiz.training.callback.metric.MeanAbsoluteError"),
                    "args": {"accumulate_over_epoch": False},
                },
            },
        },
        "engine": {
            "class_name": "mlwiz.training.engine.TrainingEngine",
            "args": {"engine_callback": "esm_surrogate.metrics.TimedEngineCallback"},
        },
        "scheduler": None,
        "gradient_clipper": None,
        "early_stopper": None,
        "plotter": None,
        "checkpoint": False,
    }
    model = DynamicMLPBaseline(8, dataset.dim_target, configuration)
    experiment = AdaptiveWidthExperiment(
        configuration, exp_path=str(tmp_path / "baseline_run"), exp_seed=4
    )
    engine = experiment.create_engine(experiment.model_config, model)

    results = engine.train(
        train_loader=loader,
        validation_loader=loader,
        max_epochs=1,
        training_timeout_seconds=-1,
    )

    train_loss, train_score, validation_loss, validation_score, _, _ = results
    assert torch.isfinite(train_loss["main_loss"])
    assert torch.isfinite(train_score["main_score"])
    assert torch.isfinite(validation_loss["main_loss"])
    assert torch.isfinite(validation_score["main_score"])
    assert model.torch_optimizer is engine.optimizer.optimizer
    assert model.instantiated_widths() == [3, 3]
