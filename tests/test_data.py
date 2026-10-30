#        Adaptive-width ESM embedding distillation
#
#   File:     tests/test_data.py
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

"""Tests for FASTA/teacher matching and protein-ID split behavior."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mlwiz.data.dataset import DatasetInterface
from mlwiz.data.util import load_dataset
from mlwiz.util import atomic_dill_save

from esm_surrogate.data import (
    ProteinEmbeddingDataset,
    ProteinIDSplitter,
    label_to_filename,
    load_teacher_embedding,
    read_fasta,
    restore_target_coordinates,
    transform_target_coordinates,
)
from esm_surrogate.experiment import (
    AdaptiveWidthExperiment,
    resolve_target_normalization_mode,
)


def _write_teacher(directory: Path, label: str, values: torch.Tensor) -> None:
    path = directory / f"{label_to_filename(label)}.pt"
    torch.save({"label": label, "encodings": values}, path)


@pytest.mark.parametrize("mode", [None, "center", "standard"])
def test_target_coordinate_transforms_round_trip(mode: str | None) -> None:
    """Notebook post-processing should exactly invert the training transform."""
    target = torch.tensor([[1.0, 7.0], [5.0, 3.0]])
    target_mean = torch.tensor([3.0, 4.0])
    target_std = torch.tensor([2.0, 3.0])

    transformed = transform_target_coordinates(
        target,
        target_mean,
        target_std,
        mode=mode,
    )
    restored = restore_target_coordinates(
        transformed,
        target_mean,
        target_std,
        mode=mode,
    )

    assert torch.equal(restored, target)


def test_dataset_matches_labels_and_normalizes_fasta(tmp_path: Path) -> None:
    """Dataset should reuse encoder naming/normalization and fixed vectors."""
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(
        ">protein_A description\nac.d-\n>protein_B\nMKT\n",
        encoding="utf-8",
    )
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    _write_teacher(
        teacher_dir, "protein_A description", torch.arange(4, dtype=torch.float32)
    )
    _write_teacher(teacher_dir, "protein_B", torch.ones(4))

    dataset = ProteinEmbeddingDataset(
        storage_folder=str(tmp_path / "cache"),
        fasta_path=str(fasta_path),
        teacher_embeddings_dir=str(teacher_dir),
        teacher_layer=33,
        max_sequence_length=8,
    )

    assert dataset.dim_input_features == 8
    assert dataset.dim_target == 4
    assert dataset.protein_ids == ["protein_A", "protein_B"]
    statistics_path = dataset.set_target_normalization([0, 1])
    assert torch.equal(dataset.target_mean, torch.tensor([0.5, 1.0, 1.5, 2.0]))
    assert torch.equal(dataset.target_std, torch.tensor([0.5, 1.0, 0.5, 1.0]))
    assert dataset.target_normalization_mode == "standard"
    (label, sequence), target = dataset[0]
    assert label == "protein_A description"
    assert sequence == "AC-D-"
    assert torch.equal(target, torch.tensor([-1.0, 0.0, 1.0, 1.0]))
    assert not target.requires_grad

    cached_payload = DatasetInterface._load_dataset(statistics_path)
    assert torch.equal(cached_payload["target_mean"], dataset.target_mean)
    assert torch.equal(cached_payload["target_std"], dataset.target_std)


def test_target_normalization_can_only_subtract_training_mean(
    tmp_path: Path,
) -> None:
    """Center mode should preserve each coordinate's original scale."""
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(">a\nAA\n>b\nCC\n", encoding="utf-8")
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    _write_teacher(teacher_dir, "a", torch.tensor([0.0, 2.0]))
    _write_teacher(teacher_dir, "b", torch.tensor([4.0, 2.0]))
    dataset = ProteinEmbeddingDataset(
        storage_folder=str(tmp_path / "cache"),
        fasta_path=str(fasta_path),
        teacher_embeddings_dir=str(teacher_dir),
    )

    dataset.set_target_normalization([0, 1], mode="center")

    assert dataset.target_normalization_mode == "center"
    assert torch.equal(dataset.target_mean, torch.tensor([2.0, 2.0]))
    assert torch.equal(dataset.target_std, torch.tensor([2.0, 1.0]))
    assert torch.equal(dataset[0][1], torch.tensor([-2.0, 0.0]))
    assert torch.equal(dataset[1][1], torch.tensor([2.0, 0.0]))


def test_target_normalization_uses_only_selected_training_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation targets must not affect fitted normalization statistics."""
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(
        ">train_a\nAA\n>train_b\nCC\n>validation\nDD\n", encoding="utf-8"
    )
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    _write_teacher(teacher_dir, "train_a", torch.tensor([0.0, 2.0]))
    _write_teacher(teacher_dir, "train_b", torch.tensor([2.0, 2.0]))
    _write_teacher(teacher_dir, "validation", torch.tensor([100.0, 10.0]))
    dataset = ProteinEmbeddingDataset(
        storage_folder=str(tmp_path / "cache"),
        fasta_path=str(fasta_path),
        teacher_embeddings_dir=str(teacher_dir),
    )

    statistics_path = dataset.set_target_normalization([0, 1])

    assert torch.equal(dataset.target_mean, torch.tensor([1.0, 2.0]))
    assert torch.equal(dataset.target_std, torch.ones(2))
    assert torch.equal(dataset[2][1], torch.tensor([99.0, 8.0]))
    assert statistics_path.is_file()

    reloaded = ProteinEmbeddingDataset(
        storage_folder=str(tmp_path / "cache"),
        fasta_path=str(fasta_path),
        teacher_embeddings_dir=str(teacher_dir),
    )

    def fail_if_recomputed(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError(f"statistics should be loaded, not fitted from {indices}")

    monkeypatch.setattr(reloaded, "_compute_target_statistics", fail_if_recomputed)
    assert reloaded.set_target_normalization([1, 0]) == statistics_path
    assert torch.equal(reloaded.target_mean, dataset.target_mean)
    assert torch.equal(reloaded.target_std, dataset.target_std)


def test_experiment_selects_inner_or_outer_training_statistics(
    tmp_path: Path,
) -> None:
    """Model-selection and final runs should fit their own training folds."""
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(">a\nAA\n>b\nCC\n>c\nDD\n", encoding="utf-8")
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    _write_teacher(teacher_dir, "a", torch.tensor([0.0]))
    _write_teacher(teacher_dir, "b", torch.tensor([2.0]))
    _write_teacher(teacher_dir, "c", torch.tensor([10.0]))
    dataset = ProteinEmbeddingDataset(
        storage_folder=str(tmp_path / "cache"),
        fasta_path=str(fasta_path),
        teacher_embeddings_dir=str(teacher_dir),
    )
    splitter = SimpleNamespace(
        inner_folds=[[SimpleNamespace(train_idxs=[0, 1])]],
        outer_folds=[SimpleNamespace(train_idxs=[1, 2])],
    )

    class Provider:
        outer_k = 0
        inner_k = 0

        @staticmethod
        def _get_splitter() -> object:
            return splitter

        @staticmethod
        def _get_dataset() -> ProteinEmbeddingDataset:
            return dataset

        @staticmethod
        def _require_outer_k() -> None:
            return None

        @staticmethod
        def _require_outer_and_inner_k() -> None:
            return None

    provider = Provider()
    experiment = AdaptiveWidthExperiment(
        {"normalize_targets": True}, exp_path=str(tmp_path / "run"), exp_seed=1
    )
    experiment._prepare_target_normalization(provider, final_run=False)
    assert torch.equal(dataset.target_mean, torch.tensor([1.0]))
    assert dataset.target_normalization_mode == "standard"

    experiment._prepare_target_normalization(provider, final_run=True)
    assert torch.equal(dataset.target_mean, torch.tensor([6.0]))

    centered_experiment = AdaptiveWidthExperiment(
        {"target_normalization": "center"},
        exp_path=str(tmp_path / "centered_run"),
        exp_seed=1,
    )
    centered_experiment._prepare_target_normalization(provider, final_run=False)
    assert dataset.target_normalization_mode == "center"
    assert torch.equal(dataset[0][1], torch.tensor([-1.0]))

    unnormalized_experiment = AdaptiveWidthExperiment(
        {},
        exp_path=str(tmp_path / "raw_run"),
        exp_seed=1,
    )
    unnormalized_experiment._prepare_target_normalization(
        provider, final_run=False
    )
    assert torch.equal(dataset[0][1], torch.tensor([0.0]))
    with pytest.raises(RuntimeError, match="statistics have not been set"):
        _ = dataset.target_mean


def test_experiment_rejects_non_boolean_target_normalization_option(
    tmp_path: Path,
) -> None:
    """String-like booleans must not silently enable normalization."""
    experiment = AdaptiveWidthExperiment(
        {"normalize_targets": "false"}, exp_path=str(tmp_path), exp_seed=1
    )
    provider = SimpleNamespace(_get_dataset=lambda: object())

    with pytest.raises(TypeError, match="normalize_targets must be a boolean"):
        experiment._prepare_target_normalization(provider, final_run=False)


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        (
            {"target_normalization": True},
            TypeError,
            "target_normalization must be a string",
        ),
        (
            {"target_normalization": "invalid"},
            ValueError,
            "target_normalization must be one of",
        ),
        (
            {"target_normalization": "center", "normalize_targets": True},
            ValueError,
            "Configure only target_normalization",
        ),
    ],
)
def test_target_normalization_mode_validation(
    config: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """The new mode option should reject ambiguous or misspelled values."""
    with pytest.raises(error, match=message):
        resolve_target_normalization_mode(config)


def test_dataset_cache_has_stable_name_and_loads_without_reprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dynamic reconstruction should find the same cache without a FASTA scan."""
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(">protein_A\nACDE\n", encoding="utf-8")
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    _write_teacher(teacher_dir, "protein_A", torch.ones(4))
    storage_folder = tmp_path / "cache"
    arguments = {
        "storage_folder": str(storage_folder),
        "fasta_path": str(fasta_path),
        "teacher_embeddings_dir": str(teacher_dir),
        "teacher_layer": 33,
        "max_sequence_length": 8,
    }

    created = ProteinEmbeddingDataset(**arguments)
    assert created.dataset_filename == "ProteinEmbeddingDataset_processed_dataset.pt"
    assert created.dataset_filepath.is_file()
    atomic_dill_save(
        arguments,
        created.dataset_folder / "dataset_kwargs.pt",
    )

    def fail_if_reprocessed(self) -> list[dict[str, str]]:
        raise AssertionError("process_dataset should not run for a matching cache")

    monkeypatch.setattr(
        ProteinEmbeddingDataset,
        "process_dataset",
        fail_if_reprocessed,
    )
    loaded = load_dataset(
        str(storage_folder),
        ProteinEmbeddingDataset,
    )

    assert loaded.protein_ids == ["protein_A"]


def test_stable_dataset_cache_rebuilds_for_changed_teacher_path(
    tmp_path: Path,
) -> None:
    """Internal fingerprints should prevent stale targets under the stable name."""
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(">protein_A\nACDE\n", encoding="utf-8")
    first_teacher_dir = tmp_path / "teachers_one"
    second_teacher_dir = tmp_path / "teachers_two"
    first_teacher_dir.mkdir()
    second_teacher_dir.mkdir()
    _write_teacher(first_teacher_dir, "protein_A", torch.ones(4))
    _write_teacher(second_teacher_dir, "protein_A", torch.full((4,), 2.0))
    storage_folder = tmp_path / "cache"

    first = ProteinEmbeddingDataset(
        storage_folder=str(storage_folder),
        fasta_path=str(fasta_path),
        teacher_embeddings_dir=str(first_teacher_dir),
    )
    second = ProteinEmbeddingDataset(
        storage_folder=str(storage_folder),
        fasta_path=str(fasta_path),
        teacher_embeddings_dir=str(second_teacher_dir),
    )

    assert first.dataset_filepath == second.dataset_filepath
    assert torch.equal(second[0][1], torch.full((4,), 2.0))


def test_dataset_migrates_matching_hashed_cache_without_reprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old hashed cache should be copied to the stable discoverable name."""
    fasta_path = (tmp_path / "proteins.fasta").resolve()
    teacher_dir = (tmp_path / "teachers").resolve()
    teacher_dir.mkdir()
    storage_folder = tmp_path / "cache"
    dataset_folder = storage_folder / "ProteinEmbeddingDataset"
    dataset_folder.mkdir(parents=True)
    fingerprint_payload = f"{fasta_path}|{teacher_dir}|33|8"
    legacy_fingerprint = hashlib.sha256(
        fingerprint_payload.encode("utf-8")
    ).hexdigest()[:12]
    legacy_path = dataset_folder / (f"ProteinEmbeddingDataset_{legacy_fingerprint}.pt")
    cached_records = [
        {
            "protein_id": "protein_A",
            "label": "protein_A",
            "sequence": "ACDE",
            "teacher_path": str(teacher_dir / "protein_A.pt"),
        }
    ]
    DatasetInterface._save_dataset(cached_records, legacy_path)
    _write_teacher(teacher_dir, "protein_A", torch.ones(4))

    def fail_if_reprocessed(self) -> list[dict[str, str]]:
        raise AssertionError("legacy cache migration should not scan the FASTA")

    monkeypatch.setattr(
        ProteinEmbeddingDataset,
        "process_dataset",
        fail_if_reprocessed,
    )
    loaded = ProteinEmbeddingDataset(
        storage_folder=str(storage_folder),
        fasta_path=str(fasta_path),
        teacher_embeddings_dir=str(teacher_dir),
        teacher_layer=33,
        max_sequence_length=8,
    )

    assert loaded.protein_ids == ["protein_A"]
    assert (
        loaded.dataset_filepath.name == "ProteinEmbeddingDataset_processed_dataset.pt"
    )
    assert loaded.dataset_filepath.is_file()
    assert legacy_path.is_file()


def test_missing_teacher_is_reported_by_label(tmp_path: Path) -> None:
    """Missing targets must fail rather than shift target indices silently."""
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(">present\nAAAA\n>missing\nCCCC\n", encoding="utf-8")
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    _write_teacher(teacher_dir, "present", torch.zeros(3))

    with pytest.raises(FileNotFoundError, match="missing"):
        ProteinEmbeddingDataset(
            storage_folder=str(tmp_path / "cache"),
            fasta_path=str(fasta_path),
            teacher_embeddings_dir=str(teacher_dir),
        )


def test_teacher_label_mismatch_detects_truncated_filename_collision(
    tmp_path: Path,
) -> None:
    """Stored labels should guard against collisions after filename truncation."""
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"label": "different", "encodings": torch.zeros(2)}, teacher_path)
    with pytest.raises(ValueError, match="label mismatch"):
        load_teacher_embedding(teacher_path, expected_label="expected")


def test_read_fasta_rejects_sequence_before_header(tmp_path: Path) -> None:
    """Malformed FASTA input should produce a useful error."""
    fasta_path = tmp_path / "bad.fasta"
    fasta_path.write_text("AAAA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before a FASTA header"):
        read_fasta(fasta_path)


@dataclass
class _IDDataset:
    protein_ids: list[str]

    def __len__(self) -> int:
        return len(self.protein_ids)


def _partition_ids(dataset: _IDDataset, splitter: ProteinIDSplitter) -> dict:
    fold = splitter.outer_folds[0]
    return {
        "train": {dataset.protein_ids[index] for index in fold.train_idxs},
        "validation": {dataset.protein_ids[index] for index in fold.val_idxs},
        "test": {dataset.protein_ids[index] for index in fold.test_idxs},
    }


def test_id_splits_are_order_independent_and_group_duplicates() -> None:
    """Reordering records cannot alter membership or split duplicate IDs."""
    ids = [f"protein_{index}" for index in range(20)] + ["protein_3"]
    first_dataset = _IDDataset(ids)
    second_dataset = _IDDataset(list(reversed(ids)))

    def make_splitter() -> ProteinIDSplitter:
        return ProteinIDSplitter(
            n_outer_folds=1,
            n_inner_folds=1,
            seed=17,
            test_ratio=0.2,
            outer_val_ratio=0.25,
            inner_val_ratio=0.25,
        )

    first_splitter = make_splitter()
    second_splitter = make_splitter()
    first_splitter.split(first_dataset)
    second_splitter.split(second_dataset)

    assert _partition_ids(first_dataset, first_splitter) == _partition_ids(
        second_dataset, second_splitter
    )
    partitions = _partition_ids(first_dataset, first_splitter)
    assert partitions["train"].isdisjoint(partitions["validation"])
    assert partitions["train"].isdisjoint(partitions["test"])
    assert partitions["validation"].isdisjoint(partitions["test"])
    first_splitter.check_splits_overlap()
