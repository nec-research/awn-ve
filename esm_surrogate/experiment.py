#        Adaptive-width ESM embedding distillation
#
#   File:     esm_surrogate/experiment.py
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

"""MLWiz experiment integration for dynamic-width student models."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from mlwiz.evaluation.config import Config
from mlwiz.experiment import Experiment
from mlwiz.model.interface import ModelInterface
from mlwiz.static import (
    BEST_CHECKPOINT_FILENAME,
    MODEL_MANIFEST_FILENAME,
    MODEL_STATE,
    SCORES,
)
from mlwiz.training.engine import TrainingEngine

from esm_surrogate.data import TARGET_NORMALIZATION_MODES

_MISSING = object()


def resolve_target_normalization_mode(
    model_config: Mapping[str, Any],
) -> str | None:
    """Resolve the target coordinate system, including legacy booleans."""
    configured_mode = model_config.get("target_normalization", _MISSING)
    legacy_option = model_config.get("normalize_targets", _MISSING)

    if configured_mode is not _MISSING:
        if legacy_option is not _MISSING:
            raise ValueError(
                "Configure only target_normalization; do not also set the "
                "legacy normalize_targets option."
            )
        if not isinstance(configured_mode, str):
            raise TypeError("target_normalization must be a string.")
        if configured_mode == "none":
            return None
        if configured_mode not in TARGET_NORMALIZATION_MODES:
            choices = ", ".join([*sorted(TARGET_NORMALIZATION_MODES), "none"])
            raise ValueError(
                f"target_normalization must be one of: {choices}; "
                f"received {configured_mode!r}."
            )
        return configured_mode

    if legacy_option is _MISSING:
        return None
    if not isinstance(legacy_option, bool):
        raise TypeError("normalize_targets must be a boolean.")
    return "standard" if legacy_option else None


class AdaptiveWidthExperiment(Experiment):
    """MLWiz experiment with optional split-aware target normalization."""

    def _prepare_target_normalization(
        self, dataset_getter: object, *, final_run: bool
    ) -> None:
        """Enable or clear normalization according to the experiment config."""
        normalization_mode = resolve_target_normalization_mode(self.model_config)

        dataset = dataset_getter._get_dataset()
        if normalization_mode is None:
            if hasattr(dataset, "clear_target_normalization"):
                dataset.clear_target_normalization()
            return

        splitter = dataset_getter._get_splitter()
        if final_run:
            dataset_getter._require_outer_k()
            training_indices = splitter.outer_folds[
                dataset_getter.outer_k
            ].train_idxs
        else:
            dataset_getter._require_outer_and_inner_k()
            training_indices = splitter.inner_folds[dataset_getter.outer_k][
                dataset_getter.inner_k
            ].train_idxs

        if not hasattr(dataset, "set_target_normalization"):
            raise TypeError(
                "AdaptiveWidthExperiment requires a dataset with "
                "set_target_normalization(training_indices, mode=...)."
            )
        dataset.set_target_normalization(
            training_indices,
            mode=normalization_mode,
        )

    def run_valid(
        self,
        dataset_getter: object,
        training_timeout_seconds: float,
        logger: object,
        progress_callback: Callable[[dict], None] | None = None,
        should_terminate: Callable[[], bool] | None = None,
    ) -> object:
        """Run model selection with the configured target coordinate system."""
        self._prepare_target_normalization(dataset_getter, final_run=False)
        return super().run_valid(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=progress_callback,
            should_terminate=should_terminate,
        )

    def _run_valid_impl(
        self,
        dataset_getter: object,
        training_timeout_seconds: float,
        logger: object,
        progress_callback: Callable[[dict], None] | None = None,
        should_terminate: Callable[[], bool] | None = None,
        ddp_rank: Optional[int] = None,
        ddp_world_size: int = 1,
    ) -> object:
        """Configure inner-fold targets inside a local or DDP worker."""
        self._prepare_target_normalization(dataset_getter, final_run=False)
        return super()._run_valid_impl(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=progress_callback,
            should_terminate=should_terminate,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

    def run_test(
        self,
        dataset_getter: object,
        training_timeout_seconds: float,
        logger: object,
        progress_callback: Callable[[dict], None] | None = None,
        should_terminate: Callable[[], bool] | None = None,
    ) -> object:
        """Run final evaluation with the configured target coordinate system."""
        self._prepare_target_normalization(dataset_getter, final_run=True)
        return super().run_test(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=progress_callback,
            should_terminate=should_terminate,
        )

    def _run_test_impl(
        self,
        dataset_getter: object,
        training_timeout_seconds: float,
        logger: object,
        progress_callback: Callable[[dict], None] | None = None,
        should_terminate: Callable[[], bool] | None = None,
        ddp_rank: Optional[int] = None,
        ddp_world_size: int = 1,
    ) -> object:
        """Configure outer-fold targets inside a local or DDP worker."""
        self._prepare_target_normalization(dataset_getter, final_run=True)
        return super()._run_test_impl(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=progress_callback,
            should_terminate=should_terminate,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

    def create_engine(self, config: Config, model: object) -> TrainingEngine:
        """Create the standard MLWiz engine and expose its optimizer to the model."""
        engine = super().create_engine(config, model)
        if hasattr(model, "set_optimizer"):
            model.set_optimizer(engine.optimizer)
        return engine


class CompressionAdaptiveWidthExperiment(AdaptiveWidthExperiment):
    """Restore the best source run, optionally within a chosen configuration."""

    _VALIDATION_METRIC = "validation_main_score"

    def __init__(
        self, model_configuration: dict[str, Any], exp_path: str, exp_seed: int
    ) -> None:
        previous_experiment = model_configuration.get("previous_experiment")
        if not isinstance(previous_experiment, str) or not previous_experiment:
            raise ValueError(
                "CompressionAdaptiveWidthExperiment requires a non-empty "
                "previous_experiment path or result-folder name."
            )

        previous_root = self._previous_experiment_root(
            previous_experiment, exp_path
        )
        previous_config_id = model_configuration.get("previous_config_id")
        if previous_config_id is not None and (
            type(previous_config_id) is not int or previous_config_id < 1
        ):
            raise ValueError("previous_config_id must be a positive integer.")
        self.previous_checkpoint, self.previous_validation_error = (
            self._best_validation_checkpoint(
                previous_root, exp_path, previous_config_id=previous_config_id
            )
        )
        manifest_path = self.previous_checkpoint.parent / MODEL_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Previous run manifest not found: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(
                f"Previous run manifest is not a mapping: {manifest_path}"
            )
        previous_config = manifest.get("config")
        if not isinstance(previous_config, dict):
            raise ValueError(
                f"Previous run manifest has no model config: {manifest_path}"
            )

        # The winning manifest supplies architecture choices such as num_layers;
        # compression-specific options in the new configuration take precedence.
        merged_config = {**previous_config, **model_configuration}
        super().__init__(merged_config, exp_path, exp_seed)

    @staticmethod
    def _previous_experiment_root(
        previous_experiment: str, exp_path: str
    ) -> Path:
        """Resolve an absolute path or a sibling MLWiz result-folder name."""
        previous_root = Path(previous_experiment).expanduser()
        if previous_root.is_absolute():
            return previous_root

        current_path = Path(exp_path)
        assessment_dir = next(
            (
                path
                for path in (current_path, *current_path.parents)
                if path.name == "MODEL_ASSESSMENT"
            ),
            None,
        )
        if assessment_dir is None:
            raise ValueError(
                "Cannot resolve a relative previous_experiment because "
                f"MODEL_ASSESSMENT is absent from the current path: {exp_path}"
            )
        return assessment_dir.parent.parent / previous_root

    @classmethod
    def _best_validation_checkpoint(
        cls,
        previous_root: Path,
        exp_path: str,
        *,
        previous_config_id: int | None = None,
    ) -> tuple[Path, float]:
        """Select a fold-matched checkpoint, optionally from a specific config."""
        outer_fold = next(
            (
                part
                for part in Path(exp_path).parts
                if part.startswith("OUTER_FOLD_")
            ),
            None,
        )
        if outer_fold is None:
            raise ValueError(
                f"Cannot locate an outer fold in the current run path: {exp_path}"
            )
        inner_fold = next(
            (
                part
                for part in Path(exp_path).parts
                if part.startswith("INNER_FOLD_")
            ),
            None,
        )

        config_folder = (
            "config_*" if previous_config_id is None else f"config_{previous_config_id}"
        )
        candidates = sorted(
            (
                previous_root
                / "MODEL_ASSESSMENT"
                / outer_fold
                / "MODEL_SELECTION"
            ).glob(f"{config_folder}/**/{BEST_CHECKPOINT_FILENAME}")
        )
        if inner_fold is not None:
            candidates = [
                path for path in candidates if inner_fold in path.parts
            ]
        if not candidates:
            fold_description = "/".join(
                part for part in (outer_fold, inner_fold) if part is not None
            )
            raise FileNotFoundError(
                "No model-selection best checkpoints found for "
                f"{fold_description or 'the requested fold'} "
                f"({config_folder}) under {previous_root}."
            )

        scored_candidates: list[tuple[float, Path]] = []
        for checkpoint_path in candidates:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            try:
                validation_error = float(
                    checkpoint[SCORES][cls._VALIDATION_METRIC]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Checkpoint has no scalar {cls._VALIDATION_METRIC!r}: "
                    f"{checkpoint_path}"
                ) from error
            scored_candidates.append((validation_error, checkpoint_path))
        validation_error, checkpoint_path = min(
            scored_candidates, key=lambda item: item[0]
        )
        return checkpoint_path, validation_error

    def create_model(
        self,
        dim_input_features: int | tuple[int, ...],
        dim_target: int,
        config: Config,
    ) -> ModelInterface:
        """Construct the winning architecture and strictly restore its weights."""
        model = super().create_model(dim_input_features, dim_target, config)
        checkpoint = torch.load(
            self.previous_checkpoint, map_location="cpu", weights_only=True
        )
        model_state = checkpoint.get(MODEL_STATE)
        if not isinstance(model_state, Mapping):
            raise ValueError(
                f"Checkpoint has no {MODEL_STATE!r} mapping: "
                f"{self.previous_checkpoint}"
            )
        model.load_state_dict(model_state, strict=True)
        set_compression_reference = getattr(model, "set_compression_reference", None)
        if not callable(set_compression_reference):
            raise TypeError(
                "CompressionAdaptiveWidthExperiment requires a model with "
                "set_compression_reference()."
            )
        set_compression_reference()
        return model
