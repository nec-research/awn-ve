#        Adaptive-width ESM embedding distillation
#
#   File:     encode_proteins_from_fasta.py
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

"""
Encode protein sequences from a FASTA file using an ESM language model.

Disk space quick reference (ESM1B, hidden=1280, spike ~1273 AA, 3 layers):
  2D token-level  : ~18.6 MB/sequence  →  ~12.5 TB for 670k sequences
  1D pooled       : ~15 KB/sequence    →  ~10 GB   for 670k sequences  (×1000 smaller)

By default only pooled 1D embeddings are written to disk.  Pass --save-2d to also
keep the full token-level tensors (use only for small datasets or targeted analysis).

Output layout:
  <output-dir>/<pooling_strategy>/<layer>/<label>.pt   ← pooled 1D (default)
  <output-dir>/<label>.pt                              ← 2D token-level (--save-2d)

Example – recommended invocation for a large dataset (single 10 GB GPU):

    python encode_proteins_from_fasta.py \\
        --input protein_data/filtered_msa.fasta \\
        --output-dir protein_data/teacher_embeddings \\
        --model ESM1B \\
        --layers 12 33 34 \\
        --gpu 0 \\
        --token-budget 16384 \\
        --aggregate-pooling average_token cls_token \\
        --skip-existing

Example – multi-GPU with DataParallel (3 GPUs, budget scaled ×3):

    python encode_proteins_from_fasta.py \\
        --input filtered_msa.fasta \\
        --output-dir ./encodings \\
        --gpu 0 1 2 \\
        --token-budget 49152 \\
        --aggregate-pooling average_token cls_token \\
        --skip-existing

Example – manual sharding across 4 GPUs (run each line in a separate terminal):

    python encode_proteins_from_fasta.py --gpu 0 --batch-offset 0    --max-batches 500 ...
    python encode_proteins_from_fasta.py --gpu 1 --batch-offset 500  --max-batches 500 ...
    python encode_proteins_from_fasta.py --gpu 2 --batch-offset 1000 --max-batches 500 ...
    python encode_proteins_from_fasta.py --gpu 3 --batch-offset 1500 --max-batches 500 ...

Example – also retain 2D embeddings (small dataset / per-position analysis):

    python encode_proteins_from_fasta.py \\
        --input sequences.fasta \\
        --output-dir ./encodings \\
        --token-budget 16384 \\
        --aggregate-pooling average_token \\
        --save-2d
"""

import gc
import logging
import argparse
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
import esm
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ["ESM1B", "ESM2B"]
SUPPORTED_POOLING = ["cls_token", "first_token", "average_token", "average_sequence"]
DEFAULT_LAYERS_ESM1B = [12, 33, 34]
DEFAULT_LAYERS_ESM2B = [12, 33]
DEFAULT_SEQ_LENGTH = 1307
MAX_FILENAME_LEN = 100
# Token budget controls how many tokens are packed into one batch.
# ESM's dataloader fills batches until this limit is hit, so longer sequences
# automatically result in fewer sequences per batch.
# Rule of thumb for ESM1B (float32):
#   GPU mem  |  safe budget  |  spike seqs/batch
#   ---------|---------------|------------------
#    10 GB   |  16000-24000  |  12-18
#    24 GB   |  40000-65000  |  30-50
#    40 GB   |  80000        |  60+
DEFAULT_TOKEN_BUDGET = 4096   # conservative; override with --token-budget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_to_filename(label: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """Sanitise a sequence label for use as a filename."""
    return (
        label
        .replace("/", "-")
        .replace(":", "_")
        .replace(" ", "")
        .replace("'", "")
        .replace(",", "_")
    )[:max_len]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

class ModelFactory:
    @staticmethod
    def load_model(model_type: str):
        if model_type == "ESM1B":
            model, alphabet = esm.pretrained.esm1_t34_670M_UR100()
        elif model_type == "ESM2B":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Choose from {SUPPORTED_MODELS}."
            )
        return model, alphabet


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ESMProteinEncoder:
    """Encodes protein sequences using a pre-trained ESM model."""

    def __init__(self, model, truncate_seq_length: int):
        self.model = model
        self.seq_length = truncate_seq_length
        self.repr_layers: list = []
        self.representations: list = []

    def load_data(self, dataset, alphabet, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.batches = dataset.get_batch_indices(token_budget, extra_toks_per_seq=1)
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(self.seq_length),
            batch_sampler=self.batches,
        )

    def _apply_pooling(self, enc_2d, pooling_strategy):
        """Pool an already-extracted 2D [seq+1, hidden] tensor to a 1D vector.

        Assumes enc_2d includes the CLS token at row 0, matching the layout
        produced by _pool(..., pooling_strategy=None).

        All branches return a *contiguous* 1D clone so that torch.save writes
        only the pooled vector (≈ hidden_dim × 4 bytes) rather than the full
        underlying storage of enc_2d.
        """
        if pooling_strategy == "cls_token":
            # Row 0 is the CLS token; clone() breaks the view to avoid saving
            # the full [seq_len, hidden] storage (would inflate files ~1000×).
            return enc_2d[0, :].clone()
        elif pooling_strategy == "first_token":
            return enc_2d[1, :].clone()
        elif pooling_strategy == "average_sequence":
            # Mean over sequence positions → [hidden_dim]
            return enc_2d[1:, :].mean(0)
        elif pooling_strategy == "average_token":
            return enc_2d[1:, :].mean(0)
        else:
            raise ValueError(
                f"Unknown pooling strategy: '{pooling_strategy}'. "
                f"Choose from {SUPPORTED_POOLING}."
            )

    def _pool(self, token_repr, pooling_strategy, truncate_len):
        """Apply pooling to a single sequence's token representations.

        token_repr: 2D tensor [seq_len+2, hidden_dim] for one sequence and one
                    layer, as returned by ESM (CLS at row 0, EOS at the last row).
        truncate_len: number of amino-acid tokens to keep (excludes CLS / EOS).
        """
        if pooling_strategy == "cls_token":
            # Row 0 is the CLS token; return 1D [hidden_dim].
            return token_repr[0, :]
        elif pooling_strategy == "first_token":
            # Row 1 is the first amino-acid token; return 1D [hidden_dim].
            return token_repr[1, :]
        elif pooling_strategy == "average_sequence":
            # Mean over amino-acid positions → [hidden_dim]
            return token_repr[1:truncate_len + 1].mean(0)
        elif pooling_strategy == "average_token":
            return token_repr[1:truncate_len + 1].mean(0)
        elif pooling_strategy is None:
            # Return full 2D embeddings including CLS token [truncate_len+1, hidden_dim]
            return token_repr[0:truncate_len + 1]
        else:
            raise ValueError(
                f"Unknown pooling strategy: '{pooling_strategy}'. "
                f"Choose from {SUPPORTED_POOLING} or omit for 2D embeddings."
            )

    def _batch_all_exist(
        self,
        labels: list,
        output_dir: Path,
        pooling_strategy,
        inline_pooling_strategies,
        repr_layers: list,
        save_2d: bool,
    ) -> bool:
        """Return True only when every output file for every label in the batch
        already exists on disk.  Used to skip the forward pass entirely."""
        for label in labels:
            file_id = label_to_filename(label)
            if inline_pooling_strategies and pooling_strategy is None:
                for strategy in inline_pooling_strategies:
                    for layer in repr_layers:
                        if not (
                            output_dir / str(strategy) / str(layer) / f"{file_id}.pt"
                        ).exists():
                            return False
            if save_2d and pooling_strategy is None:
                if not (output_dir / f"{file_id}.pt").exists():
                    return False
            if pooling_strategy is not None:
                if not (output_dir / f"{file_id}.pt").exists():
                    return False
        return True

    def encode(
        self,
        repr_layers: list,
        gpus: list = None,
        pooling_strategy=None,
        output_dir: Path = None,
        skip_existing: bool = False,
        max_batches: int = None,
        batch_offset: int = 0,
        inline_pooling_strategies: list = None,
        save_2d: bool = False,
    ):
        """Run inference and (optionally) save embeddings to disk.

        Args:
            repr_layers:               Transformer layers to extract.
            gpus:                      List of GPU device indices to use.
                                       - Single GPU [0]: standard inference.
                                       - Multiple GPUs [0, 1, 2]: DataParallel splits
                                         each batch across all listed GPUs, results
                                         gathered on gpus[0]. Scale --token-budget by
                                         the number of GPUs to fill them all.
                                       - Empty list or None: CPU inference.
            pooling_strategy:          Pooling applied per sequence. None → full 2D
                                       token-level layout used internally; whether it
                                       is written to disk is controlled by save_2d.
            output_dir:                Directory to write embedding files.
            skip_existing:             If True, skip sequences whose output file exists.
            max_batches:               Stop after this many batches. None means all.
                                       Useful for dry-runs and throughput estimation.
            batch_offset:              Skip the first N batches before processing.
                                       Used together with max_batches to shard the
                                       dataset across independent processes/GPUs
                                       without pre-splitting the FASTA file.
            inline_pooling_strategies: Pool the 2D tensors with these strategies while
                                       they are already in CPU memory (right after each
                                       batch), avoiding a separate disk-read pass.
                                       Files are written to
                                       ``output_dir/<strategy>/<layer>/<file_id>.pt``.
                                       Only active when pooling_strategy is None.
            save_2d:                   Write the full 2D token-level .pt files to
                                       output_dir. Default False — at large scale
                                       (670k sequences, ESM1B) these files total ~12 TB.
                                       Set True only for small datasets or when
                                       per-position analysis is needed.
        """
        self.repr_layers = repr_layers
        self.model.eval()

        gpus = [g for g in (gpus or []) if g >= 0]  # -1 means CPU
        use_cuda = len(gpus) > 0 and torch.cuda.is_available()
        primary_device = f"cuda:{gpus[0]}" if use_cuda else "cpu"

        if use_cuda:
            self.model = self.model.to(primary_device)
            if len(gpus) > 1:
                logging.info(f"DataParallel across GPUs {gpus}.")
                self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
            else:
                logging.info(f"Running inference on GPU {gpus[0]}.")
        else:
            logging.info("Running inference on CPU.")

        # Pre-create subdirectories for inline pooling outputs
        if inline_pooling_strategies and output_dir is not None and pooling_strategy is None:
            for strategy in inline_pooling_strategies:
                for layer in repr_layers:
                    (output_dir / str(strategy) / str(layer)).mkdir(parents=True, exist_ok=True)

        self.representations = []
        start_time = time.time()

        n_batches_total = len(self.data_loader)
        # effective window after skipping batch_offset batches
        available = n_batches_total - batch_offset
        n_batches_run = (
            min(max_batches, available) if max_batches is not None else available
        )
        if batch_offset:
            logging.info(f"Skipping first {batch_offset} batches (batch_offset).")
        if max_batches is not None:
            logging.info(
                f"max_batches={max_batches}: will process "
                f"{n_batches_run}/{n_batches_total} batches."
            )

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(
                tqdm(self.data_loader, desc="Encoding batches", total=n_batches_run)
            ):
                # skip batches before the requested offset
                if batch_idx < batch_offset:
                    continue

                if max_batches is not None and (batch_idx - batch_offset) >= max_batches:
                    logging.info(f"Reached max_batches={max_batches}, stopping.")
                    break

                # Skip inference entirely when every output file for this batch
                # already exists.  Checking labels (available before the forward
                # pass) avoids wasting GPU time on sequences already encoded.
                if skip_existing and output_dir is not None:
                    if self._batch_all_exist(
                        labels, output_dir, pooling_strategy,
                        inline_pooling_strategies, repr_layers, save_2d,
                    ):
                        logging.info(
                            f"Batch {batch_idx + 1}/{n_batches_total}: "
                            f"all {len(labels)} output files exist — skipping inference."
                        )
                        continue

                logging.info(
                    f"Batch {batch_idx + 1}/{n_batches_total} "
                    f"({len(labels)} sequences)"
                )

                if use_cuda:
                    toks = toks.to(device=primary_device, non_blocking=True)

                iter_start = time.time()
                out = self.model(toks, repr_layers=repr_layers, return_contacts=False)

                # Move all layer tensors to CPU immediately to free GPU memory.
                # With DataParallel the gather already happened on primary_device,
                # so .to("cpu") works the same as the single-GPU case.
                representations = {
                    layer: t.to(device="cpu")
                    for layer, t in out["representations"].items()
                }

                for i, label in enumerate(labels):
                    truncate_len = min(self.seq_length, len(strs[i]))

                    encodings = {
                        layer: self._pool(t[i], pooling_strategy, truncate_len).clone()
                        for layer, t in representations.items()
                    }
                    result = {
                        "label": label,
                        "encodings": encodings,
                        "pooling_strategy": pooling_strategy,
                    }

                    if output_dir is not None:
                        file_id = label_to_filename(label)

                        # 2D token-level file — opt-in only (can be ~18 MB/sequence)
                        if save_2d and pooling_strategy is None:
                            out_file = output_dir / f"{file_id}.pt"
                            if not (skip_existing and out_file.exists()):
                                torch.save(result, out_file)
                                logging.debug(f"Saved 2D: {out_file.name}")

                        # Inline pooling: tensors are already in CPU memory, so pool
                        # and write now rather than doing a second disk-read pass later.
                        if inline_pooling_strategies and pooling_strategy is None:
                            for strategy in inline_pooling_strategies:
                                for layer, enc_2d in encodings.items():
                                    pooled = self._apply_pooling(enc_2d, strategy)
                                    pool_file = (
                                        output_dir / str(strategy) / str(layer)
                                        / f"{file_id}.pt"
                                    )
                                    if not (skip_existing and pool_file.exists()):
                                        torch.save(
                                            {"label": label, "encodings": pooled},
                                            pool_file,
                                        )
                        # Tensors saved to disk — drop them from memory immediately.
                        # Accumulating all results in self.representations would consume
                        # ~10 GB RAM for 670k sequences and cause OOM on large runs.
                        del encodings, result
                    else:
                        # No output_dir: caller wants in-memory results (small runs only)
                        self.representations.append(result)

                iter_elapsed = time.time() - iter_start
                logging.info(f"Batch {batch_idx + 1} done in {iter_elapsed:.1f}s")

                # Free intermediate tensors
                del representations, out, toks
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()

        total_elapsed = time.time() - start_time
        logging.info(f"Encoding complete in {total_elapsed:.1f}s.")


# ---------------------------------------------------------------------------
# Post-hoc aggregation of saved 2D embeddings
# ---------------------------------------------------------------------------

def aggregate_encodings(
    input_dir: Path,
    output_dir: Path,
    pooling_strategies: list,
    skip_existing: bool = False,
):
    """Pool saved 2D token-level embeddings and write 1D files.

    Output layout: ``output_dir / pooling_strategy / layer / <file_id>.pt``

    Each output .pt file contains ``{"label": str, "encodings": Tensor}``.
    """
    pt_files = sorted(f for f in input_dir.iterdir() if f.suffix == ".pt")
    logging.info(
        f"Aggregating {len(pt_files)} files from '{input_dir}' "
        f"with strategies {pooling_strategies}."
    )

    for fpath in tqdm(pt_files, desc="Aggregating"):
        try:
            record = torch.load(fpath)
        except Exception as exc:
            logging.error(f"Cannot load '{fpath.name}': {exc}")
            continue

        label = record.get("label", fpath.stem)
        encodings_rec = record.get("encodings")

        if not isinstance(encodings_rec, dict):
            logging.warning(f"Skipping '{fpath.name}': 'encodings' is not a dict.")
            continue

        layers = list(encodings_rec.keys())
        if not layers:
            logging.warning(f"Skipping '{fpath.name}': no layers found.")
            continue

        file_id = label_to_filename(label)

        for pooling_strategy, layer in product(pooling_strategies, layers):
            enc = encodings_rec[layer]

            if enc.ndim == 1:
                logging.debug(
                    f"'{fpath.name}' layer {layer} already 1D, skipping."
                )
                continue

            if enc.ndim != 2:
                logging.warning(
                    f"Unexpected shape {enc.shape} in '{fpath.name}' layer {layer}."
                )
                continue

            if pooling_strategy == "cls_token":
                pooled = enc[0, :]
            elif pooling_strategy == "first_token":
                pooled = enc[1, :]
            elif pooling_strategy == "average_sequence":
                pooled = enc[1:, :].mean(1)
            elif pooling_strategy == "average_token":
                pooled = enc[1:, :].mean(0)
            else:
                raise ValueError(f"Unknown pooling strategy: '{pooling_strategy}'")

            out_subdir = output_dir / str(pooling_strategy) / str(layer)
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_file = out_subdir / f"{file_id}.pt"

            if skip_existing and out_file.exists():
                continue

            torch.save({"label": label, "encodings": pooled}, out_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Encode protein sequences from a FASTA file with an ESM model "
            "and save per-sequence embeddings as .pt files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    parser.add_argument(
        "--input", required=True, metavar="FASTA",
        help="Path to the input FASTA file.",
    )
    parser.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Directory where per-sequence .pt embedding files are saved.",
    )

    # Model configuration
    parser.add_argument(
        "--model", default="ESM1B", choices=SUPPORTED_MODELS,
        help="ESM model variant.",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None, metavar="L",
        help=(
            "Transformer layers to extract. "
            f"Default: {DEFAULT_LAYERS_ESM1B} (ESM1B) / "
            f"{DEFAULT_LAYERS_ESM2B} (ESM2B)."
        ),
    )
    parser.add_argument(
        "--seq-length", type=int, default=DEFAULT_SEQ_LENGTH, metavar="N",
        help="Maximum sequence length (longer sequences are truncated).",
    )
    parser.add_argument(
        "--gpu", nargs="+", type=int, default=[0], metavar="ID",
        help=(
            "GPU device index/indices. "
            "Single value (default: 0): standard single-GPU inference. "
            "Multiple values (e.g. --gpu 0 1 2): DataParallel across all listed GPUs "
            "— each batch is split evenly; scale --token-budget by the number of GPUs. "
            "Pass -1 to force CPU (e.g. --gpu -1)."
        ),
    )
    parser.add_argument(
        "--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET, metavar="N",
        help=(
            "Maximum total tokens packed into one batch. "
            "Higher values increase GPU utilisation and throughput but use more memory. "
            "For spike proteins (~1274 tokens) on ESM1B: "
            "4096→~3 seqs/batch (~3.6 GB), "
            "16384→~12 seqs/batch (~5.2 GB), "
            "32768→~25 seqs/batch (~7.5 GB). "
            "With N GPUs via DataParallel, multiply the budget by N. "
            f"Default: {DEFAULT_TOKEN_BUDGET} (conservative)."
        ),
    )
    parser.add_argument(
        "--batch-offset", type=int, default=0, metavar="N",
        help=(
            "Skip the first N batches before starting to encode. "
            "Use together with --max-batches to shard one FASTA across "
            "independent processes without pre-splitting the file. "
            "Example — 4-GPU manual sharding (run in parallel): "
            "  --batch-offset 0  --max-batches 500 --gpu 0  "
            "  --batch-offset 500 --max-batches 500 --gpu 1  "
            "  --batch-offset 1000 --max-batches 500 --gpu 2  "
            "  --batch-offset 1500 --max-batches 500 --gpu 3"
        ),
    )

    # Pooling
    parser.add_argument(
        "--pooling-strategy", default=None, choices=SUPPORTED_POOLING,
        metavar="STRATEGY",
        help=(
            "Pooling applied during encoding. Omit to save full 2D "
            "(token-level) embeddings. "
            f"Choices: {SUPPORTED_POOLING}."
        ),
    )
    parser.add_argument(
        "--aggregate-pooling", nargs="+", default=None,
        choices=SUPPORTED_POOLING, metavar="STRATEGY",
        help=(
            "Pool the saved 2D embeddings with these strategies. "
            "By default this is done inline during encoding (no extra disk read). "
            "Use --no-inline-pooling to run it as a separate pass instead. "
            f"Choices: {SUPPORTED_POOLING}."
        ),
    )
    parser.add_argument(
        "--no-inline-pooling", action="store_true",
        help=(
            "When --aggregate-pooling is set, run pooling as a separate pass "
            "over saved 2D files instead of inline during encoding. "
            "Useful when adding pooling to an already-completed encoding run."
        ),
    )
    parser.add_argument(
        "--save-2d", action="store_true",
        help=(
            "Also write full 2D token-level embedding files to <output-dir>. "
            "WARNING: at ESM1B scale these are ~18.6 MB per sequence "
            "(~12.5 TB for 670k sequences). "
            "Omit this flag for large datasets and use --aggregate-pooling instead."
        ),
    )

    # Misc
    parser.add_argument(
        "--max-batches", type=int, default=None, metavar="N",
        help=(
            "Stop after processing N batches. Omit to process the full dataset. "
            "Useful for dry-runs and throughput estimation before a full run."
        ),
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip sequences whose output .pt file already exists (resume mode).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser.parse_args()


def setup_logging(output_dir: Path, level: str):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"encode_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info(f"Log file: {log_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir, args.log_level)

    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)

    layers = args.layers or (
        DEFAULT_LAYERS_ESM1B if args.model == "ESM1B" else DEFAULT_LAYERS_ESM2B
    )

    logging.info("=" * 60)
    logging.info("Protein embedding pipeline")
    logging.info(f"  Input FASTA  : {input_path}")
    logging.info(f"  Output dir   : {output_dir}")
    logging.info(f"  Model        : {args.model}")
    logging.info(f"  Layers       : {layers}")
    logging.info(f"  Seq length   : {args.seq_length}")
    logging.info(f"  GPU(s)       : {args.gpu if args.gpu != [-1] else 'CPU'}")
    logging.info(f"  Token budget : {args.token_budget}")
    logging.info(f"  Batch offset : {args.batch_offset}")
    logging.info(f"  Pooling      : {args.pooling_strategy}")
    logging.info(f"  Agg. pooling : {args.aggregate_pooling or 'none'}")
    logging.info(f"  Inline pool  : {not args.no_inline_pooling}")
    logging.info(f"  Save 2D      : {args.save_2d}")
    logging.info(f"  Max batches  : {args.max_batches or 'all'}")
    logging.info(f"  Skip existing: {args.skip_existing}")
    logging.info("=" * 60)

    # Load dataset
    logging.info("Loading FASTA dataset...")
    dataset = esm.data.FastaBatchedDataset.from_file(str(input_path))
    logging.info(f"  {len(dataset)} sequences loaded.")

    # ESM expects uppercase amino acid letters; lowercase residues (e.g. soft-masked
    # regions from MAFFT/MUSCLE alignments) would be mapped to <unk> tokens otherwise.
    # Dots are hmmalign insertion-column markers — semantically identical to gap ("-")
    # but not present in ESM's alphabet, so replace them before tokenisation.
    dataset.sequence_strs = [
        seq.upper().replace(".", "-") for seq in dataset.sequence_strs
    ]
    logging.info("  Sequences uppercased and '.' → '-' substitution applied.")

    # Load model
    logging.info(f"Loading {args.model} model...")
    model, alphabet = ModelFactory.load_model(args.model)

    # Encode and save
    encoder = ESMProteinEncoder(model, truncate_seq_length=args.seq_length)
    encoder.load_data(dataset, alphabet, token_budget=args.token_budget)
    inline_pooling = (
        args.aggregate_pooling
        if args.aggregate_pooling and not args.no_inline_pooling
        else None
    )
    encoder.encode(
        repr_layers=layers,
        gpus=args.gpu,
        pooling_strategy=args.pooling_strategy,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
        max_batches=args.max_batches,
        batch_offset=args.batch_offset,
        inline_pooling_strategies=inline_pooling,
        save_2d=args.save_2d,
    )

    # Separate pooling pass: only when --no-inline-pooling is explicitly requested,
    # or when running pooling against an already-completed encoding (no model loaded).
    if args.aggregate_pooling and args.no_inline_pooling:
        logging.info(
            f"Running separate pooling pass with strategies: {args.aggregate_pooling}"
        )
        aggregate_encodings(
            input_dir=output_dir,
            output_dir=output_dir,
            pooling_strategies=args.aggregate_pooling,
            skip_existing=args.skip_existing,
        )

    logging.info("All done.")


if __name__ == "__main__":
    main()
