#        Adaptive-width ESM embedding distillation
#
#   File:     scripts/phase2_compression_report.py
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

"""Export phase-2 compression summaries and benchmark inference scaling.

The module is intentionally usable both from the phase-2 notebook and from the
command line.  Checkpoint files are treated as read-only; only report assets are
written below ``reports/esm_surrogate_progress_compression``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports/esm_surrogate_progress_compression"
FIGURE_DIR = REPORT_DIR / "figures"
DATA_DIR = REPORT_DIR / "data"
FLOP_SEQUENCE_LENGTH = 1307
BENCHMARK_RESIDUES = 1300
BENCHMARK_WARMUPS = int(os.environ.get("ESM_COMPRESSION_BENCHMARK_WARMUPS", "3"))
BENCHMARK_STEPS = int(os.environ.get("ESM_COMPRESSION_BENCHMARK_STEPS", "10"))
MAX_BATCH_POWER = int(os.environ.get("ESM_COMPRESSION_MAX_BATCH_POWER", "12"))

FOUNDATION_MODELS = {
    "esm1": {
        "label": "Original ESM-1 670M UR100 (1,024 tokens)",
        "loader": "esm1_t34_670M_UR100",
        # ESM-1 has 1,024 learned positions, including BOS and EOS.
        "benchmark_residues": 1022,
    },
    "esm1b": {
        "label": "Original ESM-1b 650M (1,024 tokens)",
        "loader": "esm1b_t33_650M_UR50S",
        # ESM-1b has 1,024 learned positions, including BOS and EOS.
        "benchmark_residues": 1022,
    },
    "foundation": {
        "label": "Foundation ESM-2 650M",
        "loader": "esm2_t33_650M_UR50D",
    },
}


def _benchmark_residues(variant: str) -> int:
    """Return the longest valid residue sequence for a benchmark variant."""
    return FOUNDATION_MODELS.get(variant, {}).get(
        "benchmark_residues", BENCHMARK_RESIDUES
    )


def _results_root() -> Path:
    override = os.environ.get("ESM_RESULTS_ROOT")
    if override:
        return Path(override).expanduser()
    root = REPO_ROOT / "RESULTS"
    if root.is_dir():
        return root
    raise FileNotFoundError(
        "Could not find the phase-2 result tree. Set ESM_RESULTS_ROOT."
    )


def _run_root() -> Path:
    return (
        _results_root()
        / "esm_distillation_compression_mean_centered_targets_ProteinEmbeddingDataset"
        / "MODEL_ASSESSMENT/OUTER_FOLD_1/MODEL_SELECTION"
    )


def _report_paths() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _scalar(value: Any) -> float:
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def _run_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in _run_root().rglob("model_manifest.json"):
        run_dir = manifest_path.parent
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        target = manifest.get("config", {}).get("parameter_compression_target_ratio")
        if manifest.get("model") != "esm_surrogate.model.AdaptiveESM" or target is None:
            continue
        config_name = next(
            (parent.name for parent in run_dir.parents if parent.name.startswith("config_")),
            run_dir.name,
        )
        records.append(
            {
                "configuration": config_name,
                "target_ratio": float(target),
                "run_directory": run_dir,
                "manifest_path": manifest_path,
                "manifest": manifest,
            }
        )
    records.sort(key=lambda record: record["target_ratio"])
    if len(records) != 3:
        raise RuntimeError(
            f"Expected three compression runs, found {len(records)} under {_run_root()}"
        )
    return records


def _instantiate_checkpoint(record: dict[str, Any], filename: str):
    from mlwiz.util import s2c

    checkpoint_path = record["run_directory"] / filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    manifest = record["manifest"]
    model_class = s2c(manifest["model"])
    model = model_class(
        dim_input_features=manifest["dim_input_features"],
        dim_target=manifest["dim_target"],
        config=dict(manifest["config"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, checkpoint_path


def _summarize_checkpoint(record: dict[str, Any], filename: str, role: str) -> dict[str, Any]:
    model, checkpoint, checkpoint_path = _instantiate_checkpoint(record, filename)
    with torch.no_grad():
        initial_parameters = float(model._initial_parameter_count.item())
        materialized_parameters = float(sum(parameter.numel() for parameter in model.parameters()))
        soft_parameters = float(model.soft_parameter_count().item())
        soft_flops = float(model.soft_flops(FLOP_SEQUENCE_LENGTH).item())
        initial_flops = float(
            model._flops_for_widths(
                FLOP_SEQUENCE_LENGTH,
                model._initial_mlp_widths,
                model._initial_attention_head_widths,
            ).item()
        )
    mlp_widths = [int(width) for width in model.instantiated_mlp_widths()]
    attention_widths = [int(width) for width in model.instantiated_attention_head_widths()]
    checkpoint_scores = checkpoint.get("scores", {})
    row = {
        "configuration": record["configuration"],
        "target_ratio": record["target_ratio"],
        "checkpoint_role": role,
        "checkpoint": str(Path("RESULTS") / checkpoint_path.relative_to(_results_root())),
        "epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", -1))) + 1,
        "validation_main_score": _scalar(checkpoint_scores["validation_main_score"]),
        "materialized_parameters": int(materialized_parameters),
        "soft_parameters": soft_parameters,
        "parameter_ratio": materialized_parameters / initial_parameters,
        "soft_parameter_ratio": soft_parameters / initial_parameters,
        "soft_flops": soft_flops,
        "initial_flops": initial_flops,
        "soft_flops_ratio": soft_flops / initial_flops,
        "mlp_min": min(mlp_widths),
        "mlp_mean": float(np.mean(mlp_widths)),
        "mlp_max": max(mlp_widths),
        "attention_head_min": min(attention_widths),
        "attention_head_mean": float(np.mean(attention_widths)),
        "attention_head_max": max(attention_widths),
        "attention_projection_min": min(attention_widths) * 20,
        "attention_projection_mean": float(np.mean(attention_widths)) * 20,
        "attention_projection_max": max(attention_widths) * 20,
        "mlp_widths": mlp_widths,
        "attention_head_widths": attention_widths,
    }
    del model, checkpoint
    return row


def _extract_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = _run_records()
    rows = []
    for record in records:
        rows.append(_summarize_checkpoint(record, "last_checkpoint.pth", "terminal"))
        rows.append(_summarize_checkpoint(record, "best_checkpoint.pth", "validation_best"))
    all_checkpoints = pd.DataFrame(rows).sort_values(
        ["target_ratio", "checkpoint_role"]
    )
    terminal = all_checkpoints[all_checkpoints["checkpoint_role"] == "terminal"].copy()
    best = all_checkpoints[all_checkpoints["checkpoint_role"] == "validation_best"].copy()
    quality = terminal[
        ["configuration", "target_ratio", "epoch", "validation_main_score", "soft_flops_ratio", "parameter_ratio"]
    ].rename(
        columns={
            "epoch": "terminal_epoch",
            "validation_main_score": "terminal_validation_rmse",
            "soft_flops_ratio": "terminal_soft_flops_ratio",
            "parameter_ratio": "terminal_parameter_ratio",
        }
    ).merge(
        best[
            ["configuration", "target_ratio", "epoch", "validation_main_score", "soft_flops_ratio", "parameter_ratio"]
        ].rename(
            columns={
                "epoch": "best_epoch",
                "validation_main_score": "best_validation_rmse",
                "soft_flops_ratio": "best_soft_flops_ratio",
                "parameter_ratio": "best_parameter_ratio",
            }
        ),
        on=["configuration", "target_ratio"],
    ).sort_values("target_ratio")
    _report_paths()
    terminal.to_csv(DATA_DIR / "compression_terminal_summary.csv", index=False)
    all_checkpoints.to_csv(DATA_DIR / "compression_checkpoint_summary.csv", index=False)
    quality.to_csv(DATA_DIR / "compression_best_vs_terminal.csv", index=False)
    with (DATA_DIR / "compression_checkpoint_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(all_checkpoints.to_dict(orient="records"), stream, indent=2)
    return terminal, quality


def _plot_compression(terminal: pd.DataFrame) -> Path:
    x = terminal["target_ratio"].to_numpy(dtype=float) * 100.0
    actual = terminal["soft_flops_ratio"].to_numpy(dtype=float) * 100.0
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    axes[0].plot([0, 55], [0, 55], "k--", linewidth=1, label="target")
    axes[0].plot(x, actual, "o-", color="#4C72B0", linewidth=2, label="realized soft FLOPs")
    for configuration, target, realized in zip(terminal["configuration"], x, actual):
        axes[0].annotate(
            f"{configuration}\n{realized:.1f}%",
            (target, realized),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set(xlabel="Configured compression target (%)", ylabel="Soft FLOPs ratio (%)", xlim=(0, 55), ylim=(0, 55), title="Compression budget")
    axes[0].legend(fontsize=8)

    def errorbar(axis, mean_key: str, min_key: str, max_key: str, title: str, ylabel: str, color: str) -> None:
        means = terminal[mean_key].to_numpy(dtype=float)
        lower = means - terminal[min_key].to_numpy(dtype=float)
        upper = terminal[max_key].to_numpy(dtype=float) - means
        axis.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="o-", capsize=4, color=color, linewidth=2)
        axis.set(xlabel="Configured compression target (%)", ylabel=ylabel, title=title)
        axis.set_xticks(
            x,
            [
                f"{int(target)}%\n{configuration}"
                for target, configuration in zip(x, terminal["configuration"])
            ],
        )
        axis.grid(True, alpha=0.3)

    errorbar(axes[1], "mlp_mean", "mlp_min", "mlp_max", "Adaptive MLP widths", "width", "#DD8452")
    errorbar(axes[2], "attention_head_mean", "attention_head_min", "attention_head_max", "Adaptive Q/K widths", "channels per head", "#55A868")
    fig.suptitle("Phase-2 compression and width adaptation", fontsize=14)
    path = FIGURE_DIR / "compression_ratio_and_width_trends.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _select_gpu() -> str:
    override = os.environ.get("ESM_SURROGATE_BENCHMARK_DEVICE")
    if override:
        return override
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        candidates = []
        for line in result.stdout.splitlines():
            index, free = [part.strip() for part in line.split(",")]
            candidates.append((int(free), int(index)))
        if candidates:
            return f"cuda:{max(candidates)[1]}"
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    return "cuda:0"


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _call_model(model: torch.nn.Module, tokens: torch.Tensor, kind: str) -> Any:
    if kind == "foundation":
        return model(tokens, repr_layers=[33], return_contacts=False)
    return model(tokens)


def _benchmark_child(variant: str, device_name: str, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The compression benchmark requires CUDA.")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    import esm

    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    residues = _benchmark_residues(variant)
    converter = alphabet.get_batch_converter(residues)
    _, _, base_tokens = converter([("benchmark", "A" * residues)])
    base_tokens = base_tokens.to(device)

    if variant in FOUNDATION_MODELS:
        foundation = FOUNDATION_MODELS[variant]
        model, _ = getattr(esm.pretrained, foundation["loader"])()
        model = model.eval().to(device)
        label = foundation["label"]
        kind = "foundation"
    else:
        record = next(record for record in _run_records() if record["configuration"] == variant)
        model, _, _ = _instantiate_checkpoint(record, "last_checkpoint.pth")
        model = model.eval().to(device)
        label = f"Adaptive ESM ({record['target_ratio']:.0%} target)"
        kind = "surrogate"
    model_parameters = sum(parameter.numel() for parameter in model.parameters())

    rows: list[dict[str, Any]] = []
    for power in range(MAX_BATCH_POWER + 1):
        batch_size = 2**power
        tokens = base_tokens.repeat(batch_size, 1)
        try:
            with torch.inference_mode():
                for _ in range(BENCHMARK_WARMUPS):
                    output = _call_model(model, tokens, kind)
                del output
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                elapsed: list[float] = []
                for _ in range(BENCHMARK_STEPS):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = _call_model(model, tokens, kind)
                    end.record()
                    end.synchronize()
                    elapsed.append(float(start.elapsed_time(end)))
                    del output
            peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
            mean_ms = statistics.fmean(elapsed)
            std_ms = statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0
            rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "model_parameters": model_parameters,
                    "batch_size": batch_size,
                    "tokens_per_sequence": int(base_tokens.shape[1]),
                    "mean_latency_ms": mean_ms,
                    "std_latency_ms": std_ms,
                    "peak_allocated_gib": peak_gib,
                    "throughput_proteins_per_second": batch_size / (mean_ms / 1000.0),
                    "status": "ok",
                    "device": torch.cuda.get_device_name(device),
                    "warmups": BENCHMARK_WARMUPS,
                    "timed_steps": BENCHMARK_STEPS,
                }
            )
            del tokens
            torch.cuda.empty_cache()
        except RuntimeError as error:
            if not _is_oom(error):
                raise
            rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "model_parameters": model_parameters,
                    "batch_size": batch_size,
                    "tokens_per_sequence": int(base_tokens.shape[1]),
                    "mean_latency_ms": np.nan,
                    "std_latency_ms": np.nan,
                    "peak_allocated_gib": np.nan,
                    "throughput_proteins_per_second": np.nan,
                    "status": "oom",
                    "device": torch.cuda.get_device_name(device),
                    "warmups": BENCHMARK_WARMUPS,
                    "timed_steps": BENCHMARK_STEPS,
                }
            )
            del tokens
            torch.cuda.empty_cache()
            break
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)


def _run_benchmark() -> pd.DataFrame:
    _report_paths()
    device = _select_gpu()
    variants = list(FOUNDATION_MODELS) + [
        record["configuration"] for record in _run_records()
    ]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        child_output = DATA_DIR / f"benchmark_{variant}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--benchmark-child", variant, "--device", device, "--output", str(child_output)]
        result = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
        if result.returncode != 0 or not child_output.is_file():
            rows.append({"variant": variant, "status": "error", "device": device, "error": result.stderr[-2000:]})
            continue
        with child_output.open(encoding="utf-8") as stream:
            rows.extend(json.load(stream))
    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(DATA_DIR / "inference_batch_scaling.csv", index=False)
    with (DATA_DIR / "inference_batch_scaling.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    return benchmark


def _plot_benchmark(benchmark: pd.DataFrame) -> Path | None:
    successful = benchmark[benchmark["status"] == "ok"].copy()
    if successful.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    colors = {
        "esm1": "#CCB974",
        "esm1b": "#8C564B",
        "foundation": "#4C72B0",
        "config_1": "#C44E52",
        "config_2": "#8172B2",
        "config_3": "#55A868",
    }
    labels = {
        "esm1": "Original ESM-1 670M UR100 (1,024 tokens)",
        "esm1b": "Original ESM-1b 650M (1,024 tokens)",
        "foundation": "Foundation ESM-2 650M",
        "config_1": "Adaptive ESM, config_1 / 50% target",
        "config_2": "Adaptive ESM, config_2 / 10% target",
        "config_3": "Adaptive ESM, config_3 / 30% target",
    }
    for variant, group in successful.groupby("variant", sort=False):
        group = group.sort_values("batch_size")
        color = colors.get(variant, "#333333")
        label = labels.get(variant, variant)
        axes[0].plot(group["batch_size"], group["mean_latency_ms"], "o-", color=color, label=label)
        axes[1].plot(group["batch_size"], group["peak_allocated_gib"], "o-", color=color, label=label)
    oom_batches = sorted(benchmark.loc[benchmark["status"] == "oom", "batch_size"].dropna().unique())
    plotted_batches = sorted(set(successful["batch_size"].astype(int)) | set(int(value) for value in oom_batches))
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(plotted_batches)
        axis.get_xaxis().set_major_formatter(lambda value, _: f"{int(value)}")
        axis.set_xlim(0.8, max(plotted_batches) * 1.15)
        axis.set_xlabel("Batch size (power of 2)")
        axis.grid(True, alpha=0.3)
        for oom_batch in oom_batches:
            axis.axvline(oom_batch, color="#555555", linestyle=":", linewidth=1, alpha=0.7)
            axis.text(
                oom_batch,
                0.97,
                f"OOM @ {int(oom_batch)}",
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="top",
                rotation=90,
                fontsize=8,
                color="#444444",
            )
    axes[0].set_ylabel("Mean model-forward latency (ms)")
    axes[0].set_title("Latency versus batch size")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Peak allocated CUDA memory (GiB)")
    axes[1].set_title("Memory occupation versus batch size")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Foundation models versus compressed surrogates", fontsize=14)
    path = FIGURE_DIR / "inference_latency_memory_by_batch_size.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _select_diagnostic_indices(
    available_indices: np.ndarray,
    maximum_samples: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic, sorted subset without changing the split."""
    if maximum_samples < 1:
        raise ValueError("maximum_samples must be positive.")
    if maximum_samples >= available_indices.size:
        return np.sort(available_indices.copy())
    generator = np.random.default_rng(seed)
    return np.sort(
        generator.choice(available_indices, size=maximum_samples, replace=False)
    )


def _norm_r2(expected: np.ndarray, predicted: np.ndarray) -> float:
    """Return the coefficient of determination for one-dimensional norms."""
    residual_sum_of_squares = np.square(expected - predicted).sum(dtype=np.float64)
    total_sum_of_squares = np.square(expected - expected.mean()).sum(dtype=np.float64)
    if total_sum_of_squares <= 0:
        return float("nan")
    return float(1.0 - residual_sum_of_squares / total_sum_of_squares)


def generate_norm_calibration_outputs(
    max_samples_per_split: int = 250,
    inference_batch_size: int = 32,
    device: str | None = None,
) -> pd.DataFrame:
    """Plot original-scale norm calibration for every terminal compression run.

    Each configuration is evaluated on the same deterministic sample from the
    inner-training, inner-validation, and outer-test splits.  The function
    writes both a 3-by-3 qualitative calibration figure and its panel metrics.
    """
    from esm_surrogate.data import (
        ESMProteinDataLoader,
        ProteinEmbeddingDataset,
        restore_target_coordinates,
    )
    from esm_surrogate.experiment import resolve_target_normalization_mode
    from mlwiz.data.splitter import Splitter
    from mlwiz.data.util import load_dataset

    _report_paths()
    records = _run_records()
    device_name = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluation_device = torch.device(device_name)
    if evaluation_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive.")

    manifests: dict[str, dict[str, Any]] = {}
    normalization_modes: set[str | None] = set()
    for record in records:
        with record["manifest_path"].open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifests[record["configuration"]] = manifest
        normalization_modes.add(resolve_target_normalization_mode(manifest["config"]))
    if len(normalization_modes) != 1:
        raise ValueError(
            "Norm diagnostics require one shared target normalization across "
            "compression runs, found "
            f"{sorted(str(mode) for mode in normalization_modes)}."
        )
    target_normalization = normalization_modes.pop()
    if target_normalization is None:
        raise ValueError(
            "Compression checkpoints must declare a target normalization mode."
        )

    reference_config = manifests[records[0]["configuration"]]["config"]
    storage_folder = Path(reference_config["storage_folder"])
    if not storage_folder.is_absolute():
        storage_folder = REPO_ROOT / storage_folder
    dataset = load_dataset(str(storage_folder), ProteinEmbeddingDataset)
    splits_path = (
        REPO_ROOT
        / "DATA_SPLITS/ProteinEmbeddingDataset"
        / "ProteinEmbeddingDataset_outer1_inner1.splits"
    )
    splitter = Splitter.load(str(splits_path))
    inner_fold = splitter.inner_folds[0][0]
    outer_fold = splitter.outer_folds[0]
    split_indices = {
        "training": np.asarray(inner_fold.train_idxs, dtype=np.int64),
        "validation": np.asarray(inner_fold.val_idxs, dtype=np.int64),
        "test": np.asarray(outer_fold.test_idxs, dtype=np.int64),
    }
    dataset.set_target_normalization(inner_fold.train_idxs, mode=target_normalization)
    target_mean = dataset.target_mean.detach().to(dtype=torch.float32, device="cpu")
    target_std = dataset.target_std.detach().to(dtype=torch.float32, device="cpu")
    selected_indices = {
        split_name: _select_diagnostic_indices(
            indices, max_samples_per_split, 42 + offset
        )
        for offset, (split_name, indices) in enumerate(split_indices.items())
    }

    def extract_predictions(model_output: Any) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output
        if (
            isinstance(model_output, (tuple, list))
            and model_output
            and isinstance(model_output[0], torch.Tensor)
        ):
            return model_output[0]
        raise TypeError("Expected a prediction tensor or a tuple/list containing one.")

    panel_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        configuration = record["configuration"]
        model, _, checkpoint_path = _instantiate_checkpoint(
            record, "last_checkpoint.pth"
        )
        model.to(evaluation_device)
        model.eval()
        for split_name, indices in selected_indices.items():
            loader = ESMProteinDataLoader(
                torch.utils.data.Subset(dataset, indices.tolist()),
                batch_size=inference_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=evaluation_device.type == "cuda",
                max_sequence_length=dataset.max_sequence_length,
            )
            target_batches: list[torch.Tensor] = []
            prediction_batches: list[torch.Tensor] = []
            with torch.inference_mode():
                for tokens, targets in loader:
                    predictions = extract_predictions(
                        model(tokens.to(evaluation_device, non_blocking=True))
                    )
                    target_batches.append(
                        targets.to(dtype=torch.float32, device="cpu")
                    )
                    prediction_batches.append(
                        predictions.to(dtype=torch.float32, device="cpu")
                    )
            teacher = restore_target_coordinates(
                torch.cat(target_batches),
                target_mean,
                target_std,
                mode=target_normalization,
            ).numpy()
            predicted = restore_target_coordinates(
                torch.cat(prediction_batches),
                target_mean,
                target_std,
                mode=target_normalization,
            ).numpy()
            teacher_norm = np.linalg.norm(teacher, axis=1)
            predicted_norm = np.linalg.norm(predicted, axis=1)
            panel_data[(configuration, split_name)] = (teacher_norm, predicted_norm)
            rows.append(
                {
                    "configuration": configuration,
                    "target_ratio": record["target_ratio"],
                    "checkpoint": str(Path("RESULTS") / checkpoint_path.relative_to(_results_root())),
                    "split": split_name,
                    "available_proteins": int(split_indices[split_name].size),
                    "evaluated_proteins": int(indices.size),
                    "norm_r2": _norm_r2(teacher_norm, predicted_norm),
                    "mean_absolute_norm_error": float(
                        np.abs(predicted_norm - teacher_norm).mean()
                    ),
                }
            )
        del model
        if evaluation_device.type == "cuda":
            torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    summary.to_csv(DATA_DIR / "compression_norm_calibration.csv", index=False)
    ordered_records = sorted(
        records, key=lambda item: item["target_ratio"], reverse=True
    )
    split_names = list(split_indices)
    fig, axes = plt.subplots(
        len(split_names),
        len(ordered_records),
        figsize=(13.5, 12),
        sharex="row",
        sharey="row",
    )
    colors = {"config_1": "#C44E52", "config_2": "#8172B2", "config_3": "#55A868"}
    for row, split_name in enumerate(split_names):
        norm_values = [
            value
            for record in ordered_records
            for values in [panel_data[(record["configuration"], split_name)]]
            for value in values
        ]
        lower = min(values.min() for values in norm_values)
        upper = max(values.max() for values in norm_values)
        margin = max(0.04 * (upper - lower), 1e-6)
        limits = (lower - margin, upper + margin)
        for column, record in enumerate(ordered_records):
            axis = axes[row, column]
            configuration = record["configuration"]
            teacher_norm, predicted_norm = panel_data[(configuration, split_name)]
            panel = summary[
                (summary["configuration"] == configuration)
                & (summary["split"] == split_name)
            ].iloc[0]
            axis.scatter(
                teacher_norm,
                predicted_norm,
                color=colors.get(configuration, "#333333"),
                s=10,
                alpha=0.35,
                linewidths=0,
            )
            axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
            axis.text(
                0.03,
                0.97,
                f"$R^2$={panel['norm_r2']:.3f}\n"
                f"MAE={panel['mean_absolute_norm_error']:.3f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
            axis.set(xlim=limits, ylim=limits)
            if row == 0:
                axis.set_title(
                    f"{configuration} / {record['target_ratio']:.0%} target"
                )
            if column == 0:
                axis.set_ylabel(f"{split_name.title()}\nPredicted L2 norm")
            if row == len(split_names) - 1:
                axis.set_xlabel("Ground-truth L2 norm")
    fig.suptitle(
        "Original-scale L2 norm calibration across compression targets",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figure_path = FIGURE_DIR / "compression_norm_calibration_by_target.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def generate_report_outputs(run_benchmark: bool = True) -> dict[str, Any]:
    """Generate all compression summaries and, optionally, run the CUDA benchmark."""
    terminal, quality = _extract_summaries()
    compression_figure = _plot_compression(terminal)
    benchmark = _run_benchmark() if run_benchmark else pd.DataFrame()
    benchmark_figure = _plot_benchmark(benchmark) if not benchmark.empty else None
    return {
        "terminal_summary": terminal,
        "best_vs_terminal": quality,
        "benchmark": benchmark,
        "compression_figure": compression_figure,
        "benchmark_figure": benchmark_figure,
        "report_dir": REPORT_DIR,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-child", metavar="VARIANT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--norm-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.benchmark_child:
        if args.output is None:
            raise ValueError("--output is required for a benchmark child.")
        _benchmark_child(args.benchmark_child, args.device, args.output)
        return
    if args.norm_diagnostics:
        print(generate_norm_calibration_outputs(device=args.device).to_string(index=False))
        return
    outputs = generate_report_outputs(run_benchmark=not args.no_benchmark)
    print(outputs["terminal_summary"].to_string(index=False))
    if not outputs["benchmark"].empty:
        print(outputs["benchmark"].to_string(index=False))


if __name__ == "__main__":
    main()
