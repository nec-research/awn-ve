#        Adaptive-width ESM embedding distillation
#
#   File:     esm_surrogate/data.py
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

"""FASTA, teacher-embedding, tokenization, and protein-ID split utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from mlwiz.data.dataset import DatasetInterface
from mlwiz.data.splitter import InnerFold, OuterFold, Splitter
from torch.utils.data import DataLoader


MAX_TEACHER_FILENAME_LENGTH = 100
TARGET_NORMALIZATION_MODES = frozenset({"center", "standard"})


def transform_target_coordinates(
    target: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    *,
    mode: str | None,
) -> torch.Tensor:
    """Transform raw targets into the configured training coordinates."""
    if mode is None:
        return target
    if mode not in TARGET_NORMALIZATION_MODES:
        choices = ", ".join(sorted(TARGET_NORMALIZATION_MODES))
        raise ValueError(f"Target normalization mode must be one of: {choices}.")
    transformed = target - target_mean
    return transformed / target_std if mode == "standard" else transformed


def restore_target_coordinates(
    target: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    *,
    mode: str | None,
) -> torch.Tensor:
    """Restore transformed targets to their original coordinates."""
    if mode is None:
        return target
    if mode not in TARGET_NORMALIZATION_MODES:
        choices = ", ".join(sorted(TARGET_NORMALIZATION_MODES))
        raise ValueError(f"Target normalization mode must be one of: {choices}.")
    restored = target * target_std if mode == "standard" else target
    return restored + target_mean


def label_to_filename(label: str, max_length: int = MAX_TEACHER_FILENAME_LENGTH) -> str:
    """Convert a FASTA label to the filename used by the teacher encoder."""
    return (
        label.replace("/", "-")
        .replace(":", "_")
        .replace(" ", "")
        .replace("'", "")
        .replace(",", "_")
    )[:max_length]


def protein_id_from_label(label: str) -> str:
    """Return the standard FASTA protein ID (text before first whitespace)."""
    protein_id = label.split(maxsplit=1)[0]
    if not protein_id:
        raise ValueError("FASTA labels must contain a non-empty protein ID.")
    return protein_id


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read FASTA records while preserving the full label text."""
    records: list[tuple[str, str]] = []
    label: str | None = None
    sequence_parts: list[str] = []

    def flush_record() -> None:
        if label is None:
            return
        sequence = "".join(sequence_parts).upper().replace(".", "-")
        if not sequence:
            raise ValueError(f"Protein '{label}' has an empty FASTA sequence.")
        records.append((label, sequence))

    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                label = line[1:].strip() or f"seqnum{line_number:09d}"
                sequence_parts = []
            else:
                if label is None:
                    raise ValueError(
                        f"Sequence data appears before a FASTA header at line "
                        f"{line_number} in '{path}'."
                    )
                sequence_parts.append(line)
    flush_record()

    if not records:
        raise ValueError(f"No FASTA records found in '{path}'.")
    return records


def load_teacher_embedding(
    path: str | Path,
    teacher_layer: int | str | None = None,
    expected_label: str | None = None,
) -> torch.Tensor:
    """Load one fixed teacher vector from the encoder's ``.pt`` format."""
    try:
        record = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        record = torch.load(path, map_location="cpu")

    if (
        expected_label is not None
        and isinstance(record, dict)
        and record.get("label", expected_label) != expected_label
    ):
        raise ValueError(
            f"Teacher label mismatch for '{path}': expected "
            f"'{expected_label}', found '{record.get('label')}'. This can "
            "indicate a collision caused by truncated filenames."
        )

    encoding = record.get("encodings") if isinstance(record, dict) else record
    if isinstance(encoding, dict):
        if teacher_layer is None:
            if len(encoding) != 1:
                raise ValueError(
                    f"Teacher file '{path}' contains multiple layers; set "
                    "teacher_layer explicitly."
                )
            encoding = next(iter(encoding.values()))
        else:
            keys = (teacher_layer, str(teacher_layer))
            encoding = next((encoding[key] for key in keys if key in encoding), None)

    if not torch.is_tensor(encoding):
        raise ValueError(f"Teacher file '{path}' does not contain a tensor encoding.")
    if encoding.ndim != 1:
        raise ValueError(
            f"Expected a pooled 1D teacher embedding in '{path}', found "
            f"shape {tuple(encoding.shape)}."
        )
    return encoding.detach().to(dtype=torch.get_default_dtype()).contiguous()


def _stable_id_order(
    protein_ids: Iterable[str], seed: int, namespace: str
) -> list[str]:
    """Order unique IDs by a seeded stable digest, independently of file order."""
    unique_ids = set(protein_ids)

    def key(protein_id: str) -> tuple[bytes, str]:
        payload = f"{seed}:{namespace}:{protein_id}".encode("utf-8")
        return hashlib.sha256(payload).digest(), protein_id

    return sorted(unique_ids, key=key)


def _evaluation_count(size: int, ratio: float) -> int:
    """Convert a hold-out ratio to a useful deterministic group count."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"Split ratios must be in [0, 1), found {ratio}.")
    if size < 2 or ratio == 0.0:
        return 0
    return min(max(int(round(size * ratio)), 1), size - 1)


def deterministic_id_holdout(
    protein_ids: Sequence[str], ratio: float, seed: int, namespace: str
) -> tuple[set[str], set[str]]:
    """Split protein-ID groups into retained and held-out sets reproducibly."""
    ordered_ids = _stable_id_order(protein_ids, seed=seed, namespace=namespace)
    held_out_count = _evaluation_count(len(ordered_ids), ratio)
    held_out = set(ordered_ids[:held_out_count])
    retained = set(ordered_ids[held_out_count:])
    return retained, held_out


class ProteinEmbeddingDataset(DatasetInterface):
    """FASTA dataset whose targets are fixed, precomputed teacher embeddings.

    The processed cache contains only labels, sequences, IDs, and target paths.
    Teacher tensors remain separate and are loaded lazily, so the original
    embedding directory is never copied or incorporated into the training graph.
    Target statistics are fitted separately from a run's training indices and
    stored in a keyed sidecar file for reuse. The cache has a stable filename
    for MLWiz dynamic loading; its payload holds a fingerprint that detects
    changes to metadata-producing input paths.
    """

    _CACHE_FORMAT_VERSION = 1

    def __init__(
        self,
        storage_folder: str,
        fasta_path: str,
        teacher_embeddings_dir: str,
        teacher_layer: int | str | None = None,
        max_sequence_length: int = 1307,
        **kwargs: object,
    ) -> None:
        self.fasta_path = str(Path(fasta_path).expanduser().resolve())
        self.teacher_embeddings_dir = str(
            Path(teacher_embeddings_dir).expanduser().resolve()
        )
        self.teacher_layer = teacher_layer
        self.max_sequence_length = int(max_sequence_length)
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive.")

        metadata_fingerprint_payload = (
            f"{self._CACHE_FORMAT_VERSION}|{self.fasta_path}|"
            f"{self.teacher_embeddings_dir}"
        )
        self._cache_fingerprint = hashlib.sha256(
            metadata_fingerprint_payload.encode("utf-8")
        ).hexdigest()

        # Releases before the stable cache name embedded a short configuration
        # digest in the filename. Reuse an exact legacy match without scanning
        # the FASTA again, then leave the old cache untouched.
        legacy_fingerprint_payload = (
            f"{self.fasta_path}|{self.teacher_embeddings_dir}|"
            f"{self.teacher_layer}|{self.max_sequence_length}"
        )
        legacy_fingerprint = hashlib.sha256(
            legacy_fingerprint_payload.encode("utf-8")
        ).hexdigest()[:12]
        self._migrate_legacy_cache(storage_folder, legacy_fingerprint)
        super().__init__(storage_folder=storage_folder, **kwargs)

        if len(self.dataset) == 0:
            raise ValueError("The processed protein dataset is empty.")
        self._dim_target = int(
            load_teacher_embedding(
                self.dataset[0]["teacher_path"],
                self.teacher_layer,
                expected_label=self.dataset[0]["label"],
            ).numel()
        )
        self._target_mean: torch.Tensor | None = None
        self._target_std: torch.Tensor | None = None
        self._target_normalization_mode: str | None = None

    def _migrate_legacy_cache(
        self, storage_folder: str, legacy_fingerprint: str
    ) -> None:
        """Copy an exact legacy hashed cache into the stable cache location."""
        dataset_folder = Path(storage_folder, self.__class__.__name__)
        stable_path = dataset_folder / (
            f"{self.__class__.__name__}_processed_dataset.pt"
        )
        legacy_path = dataset_folder / (
            f"{self.__class__.__name__}_{legacy_fingerprint}.pt"
        )
        if stable_path.exists() or not legacy_path.is_file():
            return

        legacy_dataset = DatasetInterface._load_dataset(legacy_path)
        payload = {
            "cache_format_version": self._CACHE_FORMAT_VERSION,
            "fingerprint": self._cache_fingerprint,
            "dataset": legacy_dataset,
        }
        DatasetInterface._save_dataset(payload, stable_path)

    def _save_dataset(self, dataset: object, dataset_filepath: str | Path) -> None:
        """Store metadata under a stable name with its validation fingerprint."""
        payload = {
            "cache_format_version": self._CACHE_FORMAT_VERSION,
            "fingerprint": self._cache_fingerprint,
            "dataset": dataset,
        }
        DatasetInterface._save_dataset(payload, dataset_filepath)

    def _load_dataset(self, dataset_filepath: str | Path) -> object:
        """Load matching metadata or rebuild it when source paths changed."""
        payload = DatasetInterface._load_dataset(dataset_filepath)
        if (
            isinstance(payload, dict)
            and payload.get("cache_format_version") == self._CACHE_FORMAT_VERSION
            and payload.get("fingerprint") == self._cache_fingerprint
            and "dataset" in payload
        ):
            return payload["dataset"]

        print(
            f"Cache metadata in {dataset_filepath} does not match the configured "
            "FASTA/teacher paths; rebuilding it..."
        )
        dataset = self.process_dataset()
        if self.pre_transform is not None:
            dataset = [self.pre_transform(record) for record in dataset]
        self._save_dataset(dataset, dataset_filepath)
        return dataset

    @property
    def dim_input_features(self) -> int:
        """Return the configured maximum amino-acid sequence length."""
        return self.max_sequence_length

    @property
    def dim_target(self) -> int:
        """Return the teacher embedding dimension."""
        return self._dim_target

    @property
    def target_mean(self) -> torch.Tensor:
        """Return the active training-set teacher-embedding mean."""
        if self._target_mean is None:
            raise RuntimeError("Target normalization statistics have not been set.")
        return self._target_mean

    @property
    def target_std(self) -> torch.Tensor:
        """Return the active training-set teacher-embedding standard deviation."""
        if self._target_std is None:
            raise RuntimeError("Target normalization statistics have not been set.")
        return self._target_std

    @property
    def target_normalization_mode(self) -> str | None:
        """Return the active target transformation, if any."""
        return self._target_normalization_mode

    @property
    def protein_ids(self) -> list[str]:
        """Return one protein ID per dataset record."""
        return [record["protein_id"] for record in self.dataset]

    def process_dataset(self) -> list[dict[str, str]]:
        """Match FASTA records to teacher files by label-derived identity."""
        fasta_path = Path(self.fasta_path)
        teacher_dir = Path(self.teacher_embeddings_dir)
        if not fasta_path.is_file():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        if not teacher_dir.is_dir():
            raise FileNotFoundError(
                f"Teacher embedding directory not found: {teacher_dir}"
            )

        records: list[dict[str, str]] = []
        missing: list[str] = []
        for label, sequence in read_fasta(fasta_path):
            teacher_path = teacher_dir / f"{label_to_filename(label)}.pt"
            if not teacher_path.is_file():
                missing.append(label)
                continue
            records.append(
                {
                    "protein_id": protein_id_from_label(label),
                    "label": label,
                    "sequence": sequence,
                    "teacher_path": str(teacher_path),
                }
            )

        if missing:
            preview = ", ".join(repr(label) for label in missing[:3])
            raise FileNotFoundError(
                f"Missing teacher embeddings for {len(missing)} FASTA records "
                f"in '{teacher_dir}'. First missing labels: {preview}"
            )
        return records

    def _compute_target_statistics(
        self, indices: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute population statistics without loading all targets at once."""
        if not indices:
            raise ValueError("Target statistics require at least one training index.")

        mean: torch.Tensor | None = None
        sum_squared_deviations: torch.Tensor | None = None
        for count, index in enumerate(indices, start=1):
            record = self.dataset[index]
            target = load_teacher_embedding(
                record["teacher_path"],
                teacher_layer=self.teacher_layer,
                expected_label=record["label"],
            ).to(dtype=torch.float64)
            if not torch.isfinite(target).all():
                raise ValueError(
                    f"Teacher embedding for '{record['label']}' contains "
                    "non-finite values."
                )
            if mean is None:
                mean = torch.zeros_like(target)
                sum_squared_deviations = torch.zeros_like(target)
            elif target.shape != mean.shape:
                raise ValueError(
                    "Teacher embeddings must have a consistent dimension; "
                    f"expected {mean.numel()}, found {target.numel()} for "
                    f"'{record['label']}'."
                )

            delta = target - mean
            mean = mean + delta / count
            sum_squared_deviations = sum_squared_deviations + delta * (
                target - mean
            )

        assert mean is not None and sum_squared_deviations is not None
        target_dtype = torch.get_default_dtype()
        mean = mean.to(dtype=target_dtype)
        target_std = (sum_squared_deviations / len(indices)).clamp_min(0).sqrt()
        target_std = target_std.to(dtype=target_dtype)
        # Constant dimensions are already centered at zero. A unit scale keeps
        # them at zero without introducing infinities during normalization.
        target_std = torch.where(
            target_std > 0, target_std, torch.ones_like(target_std)
        )
        return mean.contiguous(), target_std.contiguous()

    def set_target_normalization(
        self,
        training_indices: Sequence[int],
        *,
        mode: str = "standard",
    ) -> Path:
        """Fit or load normalization vectors for a specific training split.

        Statistics are keyed by the source data, teacher layer, and exact set
        of training indices. This lets model-selection and final runs use their
        respective training folds without leaking validation or test targets.
        """
        if not isinstance(mode, str):
            raise TypeError("Target normalization mode must be a string.")
        if mode not in TARGET_NORMALIZATION_MODES:
            choices = ", ".join(sorted(TARGET_NORMALIZATION_MODES))
            raise ValueError(
                f"Target normalization mode must be one of: {choices}; "
                f"received {mode!r}."
            )
        indices = sorted({int(index) for index in training_indices})
        if not indices:
            raise ValueError("Target normalization requires a non-empty training set.")
        if indices[0] < 0 or indices[-1] >= len(self.dataset):
            raise IndexError("Target normalization indices are outside the dataset.")

        digest = hashlib.sha256()
        statistics_source = (
            f"{self.fasta_path}\0{self.teacher_embeddings_dir}\0"
            f"{self.teacher_layer}\0"
        )
        digest.update(statistics_source.encode("utf-8"))
        for index in indices:
            digest.update(index.to_bytes(8, byteorder="big", signed=False))
        statistics_key = digest.hexdigest()
        statistics_path = self.dataset_folder / (
            f"target_statistics_{statistics_key[:16]}.pt"
        )

        if statistics_path.is_file():
            payload = DatasetInterface._load_dataset(statistics_path)
            if (
                isinstance(payload, dict)
                and payload.get("statistics_key") == statistics_key
                and torch.is_tensor(payload.get("target_mean"))
                and torch.is_tensor(payload.get("target_std"))
            ):
                self._target_mean = payload["target_mean"]
                self._target_std = payload["target_std"]
                self._target_normalization_mode = mode
                return statistics_path

        self._target_mean, self._target_std = self._compute_target_statistics(indices)
        self._target_normalization_mode = mode
        DatasetInterface._save_dataset(
            {
                "statistics_key": statistics_key,
                "num_training_samples": len(indices),
                "target_mean": self._target_mean,
                "target_std": self._target_std,
            },
            statistics_path,
        )
        return statistics_path

    def clear_target_normalization(self) -> None:
        """Return targets in their original, unnormalized coordinate system."""
        self._target_mean = None
        self._target_std = None
        self._target_normalization_mode = None


    def __getitem__(self, index: int) -> tuple[tuple[str, str], torch.Tensor]:
        """Return a teacher vector, normalized when statistics are active."""
        record = self.dataset[index]
        teacher = load_teacher_embedding(
            record["teacher_path"],
            teacher_layer=self.teacher_layer,
            expected_label=record["label"],
        )
        if self._target_normalization_mode is not None:
            assert self._target_mean is not None and self._target_std is not None
            teacher = transform_target_coordinates(
                teacher,
                self._target_mean,
                self._target_std,
                mode=self._target_normalization_mode,
            )
        return (record["label"], record["sequence"]), teacher


class ESMBatchCollator:
    """Convert raw protein strings with fair-esm's native batch converter."""

    def __init__(self, max_sequence_length: int) -> None:
        try:
            import esm
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "fair-esm is required for protein tokenization; install the "
                "project dependencies first."
            ) from error
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = alphabet.get_batch_converter(max_sequence_length)

    def __call__(
        self, batch: Sequence[tuple[tuple[str, str], torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and pad a batch, returning MLWiz ``(input, target)``."""
        proteins, teacher_embeddings = zip(*batch)
        _, _, tokens = self.batch_converter(proteins)
        return tokens, torch.stack(teacher_embeddings, dim=0)


class ESMProteinDataLoader(DataLoader):
    """PyTorch loader that performs fair-esm tokenization during collation."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *args: object,
        max_sequence_length: int = 1307,
        **kwargs: object,
    ) -> None:
        if "collate_fn" in kwargs:
            raise ValueError("ESMProteinDataLoader defines its own collate_fn.")
        super().__init__(
            dataset,
            *args,
            collate_fn=ESMBatchCollator(max_sequence_length),
            **kwargs,
        )


class ProteinIDSplitter(Splitter):
    """MLWiz hold-out splitter that assigns complete protein-ID groups.

    Hash ordering is based on ``(seed, split namespace, protein ID)``. Therefore
    membership is reproducible and invariant to FASTA order or target indexing.
    Duplicate records with the same ID always remain in the same partition.
    """

    def __init__(
        self,
        n_outer_folds: int,
        n_inner_folds: int,
        seed: int,
        stratify: bool = False,
        shuffle: bool = True,
        inner_val_ratio: float = 0.1,
        outer_val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        if n_outer_folds != 1 or n_inner_folds != 1:
            raise NotImplementedError(
                "ProteinIDSplitter currently implements deterministic hold-out "
                "splits only (n_outer_folds=n_inner_folds=1)."
            )
        if stratify:
            raise ValueError("Protein embedding distillation is not stratified.")
        requested_seed = seed
        super().__init__(
            n_outer_folds=n_outer_folds,
            n_inner_folds=n_inner_folds,
            seed=seed,
            stratify=False,
            shuffle=shuffle,
            inner_val_ratio=inner_val_ratio,
            outer_val_ratio=outer_val_ratio,
            test_ratio=test_ratio,
        )
        # Base Splitter clears the seed when shuffle=False. ID hashing does not
        # depend on sample shuffling, so retain the requested reproducibility key.
        self.seed = requested_seed

    @staticmethod
    def _indices_for_ids(
        protein_ids: Sequence[str], selected_ids: set[str]
    ) -> list[int]:
        return [
            index
            for index, protein_id in enumerate(protein_ids)
            if protein_id in selected_ids
        ]

    def split(self, dataset: DatasetInterface, targets: object = None) -> None:
        """Create MLWiz outer and inner folds exclusively from protein IDs."""
        del targets
        if not hasattr(dataset, "protein_ids"):
            raise TypeError("ProteinIDSplitter requires dataset.protein_ids.")
        protein_ids = list(dataset.protein_ids)
        if not protein_ids:
            raise ValueError("Cannot split an empty protein dataset.")

        outer_ids, test_ids = deterministic_id_holdout(
            protein_ids,
            ratio=self.test_ratio,
            seed=int(self.seed),
            namespace="outer-test",
        )
        outer_train_ids, outer_val_ids = deterministic_id_holdout(
            sorted(outer_ids),
            ratio=self.outer_val_ratio,
            seed=int(self.seed),
            namespace="outer-validation",
        )
        inner_train_ids, inner_val_ids = deterministic_id_holdout(
            sorted(outer_ids),
            ratio=self.inner_val_ratio,
            seed=int(self.seed),
            namespace="inner-validation",
        )

        self.outer_folds = [
            OuterFold(
                train_idxs=self._indices_for_ids(protein_ids, outer_train_ids),
                val_idxs=self._indices_for_ids(protein_ids, outer_val_ids),
                test_idxs=self._indices_for_ids(protein_ids, test_ids),
            )
        ]
        self.inner_folds = [
            [
                InnerFold(
                    train_idxs=self._indices_for_ids(protein_ids, inner_train_ids),
                    val_idxs=self._indices_for_ids(protein_ids, inner_val_ids),
                )
            ]
        ]
