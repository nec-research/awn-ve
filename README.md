# Adaptive-width ESM embedding distillation

We built a protein embedding student that learns to reproduce embeddings from a
pretrained ESM teacher. The teacher embeddings are computed once and kept fixed;
the student is trained from scratch on the corresponding protein sequences.
Its output is the average of the final amino-acid token representations,
excluding special tokens and padding.

The student follows the ESM transformer architecture, with adaptive widths in
each block's feed-forward network and query/key attention projections. These
widths can grow or shrink during training while preserving existing weights and
optimizer state. Value and output projections keep a fixed dimension. This lets
us explore how much model capacity is needed to approximate the teacher.

We also added training with mean-centered targets and a second training stage
that encourages a smaller parameter count
or FLOPs budget. A dynamic MLP baseline and a sliding-window attention variant
provide additional comparisons. MLWiz manages data splits, training, model
selection, checkpoints, and metrics. This is a research prototype; the repository
does not establish a particular compression or accuracy result.


## Install

You need Python 3.10 or newer and `uv`. The supplied training configurations use
a CUDA GPU. From your clone of this repository, run:

```bash
uv venv --relocatable
uv sync --locked
source .venv/bin/activate
```

Activate the environment again in each new terminal. This makes `mlwiz-data`,
`mlwiz-exp`, and `mlwiz-dashboard` available.

Create the venv with `--relocatable` so activation and installed command launchers
find Python relative to the venv instead of storing the checkout's absolute path.
To convert an existing venv and regenerate the MLWiz launchers, run:

```bash
uv venv --allow-existing --relocatable --python .venv/bin/python .venv
uv pip install --python .venv/bin/python --no-deps --reinstall 'mlwiz==1.7.6'
source .venv/bin/activate
mlwiz-exp --help
```

Other commands already installed before conversion need their owning packages
reinstalled to get relocatable launchers. After moving the checkout, run
`uv sync --locked --inexact --reinstall-package esm-surrogate` to refresh the
editable project's source paths, then activate the venv again. Moving to another
machine may require recreating the venv if its base Python is unavailable.

You can also run `uv run --locked mlwiz-exp --help` without activating the venv.

## Prepare your data

You need a FASTA file and one fixed teacher embedding per sequence. If you do
not already have embeddings, generate ESM-2 layer-33 averages with:

```bash
python encode_proteins_from_fasta.py \
  --input /path/to/sequences.fasta \
  --output-dir /path/to/teacher_embeddings \
  --model ESM2B \
  --layers 33 \
  --seq-length 1307 \
  --aggregate-pooling average_token
```

The encoder downloads the pretrained teacher weights on first use. Each output
file is stored as `average_token/33/<sanitized-label>.pt` and contains
`{"label": ..., "encodings": Tensor[D]}`. Keep the teacher and student sequence
truncation consistent, and set the student's `embed_dim` to the target dimension.

Configuration defaults use repository-relative `protein_data/` and `RESULTS/`
directories. Run commands from the repository root and edit the data paths for
your setup. Report generation and checkpoint comparison tests also accept
`ESM_RESULTS_ROOT` for an external results directory. Saved report and export
checkpoint references use the portable `RESULTS/` prefix.

Edit [configs/data.yml](configs/data.yml) to point to your files:

```yaml
dataset:
  class_name: esm_surrogate.data.ProteinEmbeddingDataset
  args:
    storage_folder: DATA
    fasta_path: /path/to/sequences.fasta
    teacher_embeddings_dir: /path/to/teacher_embeddings/average_token/33
    teacher_layer: 33
    max_sequence_length: 1307
```

Keep the existing `splitter` section, then prepare the metadata and splits:

```bash
mlwiz-data --config-file configs/data.yml
```

Splits are deterministic by protein ID: duplicate IDs stay in the same
partition, and reordering the FASTA does not change membership. Metadata is
stored in `DATA/` and splits in `DATA_SPLITS/` by default.

After moving the checkout or the protein data, update the paths in
`configs/data.yml` and rerun `mlwiz-data --config-file configs/data.yml` before
starting an experiment. MLWiz loads FASTA and teacher paths from
`DATA/ProteinEmbeddingDataset/dataset_kwargs.pt`; editing the experiment YAML
does not refresh those saved paths. The data command updates them and rebuilds
the processed metadata when the configured source paths change. It reuses
existing splits, so use the same FASTA records in the same order when relocating
data for a continuation or compression run.

## Train a student

Start with
[configs/experiment_mean_centered_targets.yml](configs/experiment_mean_centered_targets.yml).
The checked-in YAML files are examples from our experiments and contain local
paths and resource settings. Before running, update:

- `experiment.result_folder` to your output directory, such as `RESULTS`.
- `experiment.telegram_config_file` to `null` unless you use Telegram notifications.
- `resources` and `data_loading.data_loader.args.num_workers` for your machine.
- `grid.batch_size`, `grid.epochs`, and the model dimensions for your experiment.
- `grid.n_observations` to the number of training examples in your split.

Keep the experiment's dataset storage and split-file paths consistent with the
data configuration, and use the same `max_sequence_length`. YAML lists define
multiple configurations to evaluate; use a single value for one run.

```bash
mlwiz-exp --config-file configs/experiment_mean_centered_targets.yml --debug
```

This configuration learns mean-centered teacher embeddings with MSE loss and
selects checkpoints using validation RMSE. Centering statistics come only from
the active training fold. Set `grid.target_normalization` to `none` for original
targets or `standard` for per-dimension standardization.

`--debug` runs locally. Remove it to use MLWiz's Ray scheduling.

## Other experiments

Use the same `mlwiz-exp --config-file <file> --debug` command with these configs:

| Configuration in `configs/` | Purpose |
| --- | --- |
| `experiment_compression_mean_centered_targets.yml` | Compress a trained mean-centered student using a FLOPs penalty. |
| `experiment_dynamic_mlp.yml` | Train an adaptive-width MLP baseline. |

For a compression run, set `grid.previous_experiment` to the source
experiment directory or its name under the results directory. The source must
contain completed model-selection checkpoints and their model manifests.
Selection uses the best validation score within the matching data fold.
`grid.previous_config_id`, when present, restricts selection to one configuration.

Compression ratios are relative to the restored checkpoint. Adjust
`parameter_compression_target_ratio` and `parameter_compression_lambda` to set
the target and penalty strength. The supplied compression configs optimize an
estimated FLOPs ratio; a parameter-count objective is also implemented in
[esm_surrogate/metrics.py](esm_surrogate/metrics.py).

For sliding-window attention, set `grid.model` to
`esm_surrogate.model_sliding_attention.AdaptiveESM` and add
`grid.attention_window_radius` (positions on each side of a query). It accepts
dense-model checkpoints and uses FlexAttention on supported CUDA execution
paths, with a dense local fallback otherwise. See
[the benchmark notebook](notebooks/sliding_attention_benchmark.ipynb) for comparisons.

## Inspect results

Point the dashboard at the output directory you configured:

```bash
mlwiz-dashboard --logdir RESULTS
```

Runs save checkpoints, model manifests, and `metrics_data.torch` histories.
Metrics include embedding error, parameter counts, forward time, and per-block
feed-forward and attention widths. Compression runs also report their budget
ratios and penalties.

## Downstream validation

The [`downstream_validation/`](downstream_validation) package bundles standalone
inference exports of the terminal compressed checkpoints together with a
biological validation pipeline (sequence reconstruction, BLOSUM distance
preservation, and lineage classification) that evaluates on the surrogate's
held-out test fold. Inference and validation utilities use the package's runtime
dependencies; computing the fold assignments reuses
`esm_surrogate.data.ProteinIDSplitter` and requires MLWiz. Re-exporting checkpoints
also requires the training environment. See
[downstream_validation/README.md](downstream_validation/README.md) for commands
and figure generation.

## Run tests

```bash
uv pip install pytest
pytest
```

## Code release inputs and reproduction

This archive contains source code, tests, configs and unexecuted notebooks.
See `LICENSE` for the NEC noncommercial research agreement and covered files.
FASTA sequences, teacher embeddings, generated splits, checkpoints, exported
models, reports and validation results are intentionally omitted.

Run commands from this project's root. Supply the aligned FASTA as
`protein_data/filtered_msa.fasta` and place the teacher embeddings under
`protein_data/average_token/33`, or edit `configs/data.yml` to use other
project-relative locations. Follow the encoding, dataset and training commands
above to regenerate embeddings, splits and training runs under `DATA/`,
`DATA_SPLITS/` and `RESULTS/`. Dataset provenance and access must be provided
separately; no sequences or download credentials are included.

Export the three trained checkpoints with
`python -m downstream_validation.export_checkpoints --results-root RESULTS`.
For downstream experiments, set `FASTA=../protein_data/filtered_msa.fasta` when
running the numbered scripts from `downstream_validation/`. Those scripts
regenerate decoder, BLOSUM and lineage outputs and their figures.
`scripts/phase2_compression_report.py` and the notebooks regenerate training
diagnostics; report directories are created when those tools run.

Tests requiring the omitted data or model artifacts are skipped until those
inputs have been generated. The license author names and existing optimizer
attribution are retained intentionally; this archive is not author-anonymous.
