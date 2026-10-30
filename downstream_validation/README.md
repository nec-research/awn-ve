# Downstream validation

Biological validation of the compressed surrogate models, with code to export
the three terminal phase-2 model-selection checkpoints.
Inference requires only PyTorch and `fair-esm`; computing the validation splits
also requires this repository and MLWiz, as described below.

```bash
pip install -r downstream_validation/requirements.txt
python -m downstream_validation.encode_proteins_from_fasta \
  --input proteins.fasta --output-dir embeddings --model config_3 --gpu 0
```

Each output file is `<output-dir>/<sanitized FASTA label>.pt` and contains
`{"label": <label>, "encodings": <raw 1280-D tensor>}`. The model output is restored
to raw teacher-embedding coordinates by adding the model-selection training-fold mean.

After training, export the checkpoints from your results tree:

```bash
.venv/bin/python -m downstream_validation.export_checkpoints --results-root RESULTS
```

Run from the repository root, or pass your results directory with `--results-root`.
Exported checkpoint references use a portable `RESULTS/` prefix.

## Downstream validation

Beyond inference, this package contains a biological-validation pipeline that
probes how well the compressed students preserve information. All of its modules
depend only on `torch`, `fair-esm`, `numpy`, `scipy`, `matplotlib`, and `tqdm`,
with one exception: computing the split (`aligned_splits` in
[`splits.py`](splits.py)) delegates to `esm_surrogate.data.ProteinIDSplitter`,
which imports MLWiz. That import is lazy, so the decoder, BLOSUM, and lineage
modules can be imported without MLWiz. Computing fold assignments and
re-exporting checkpoints require the full `esm_surrogate` environment. Install the extra dependencies with
`pip install -r downstream_validation/requirements.txt`.

All experiments evaluate on the leakage-free held-out **test fold**. The split is
reproduced from the FASTA alone by [`splits.py`](splits.py) (`aligned_splits`),
which reuses `esm_surrogate.data.ProteinIDSplitter` (seeded SHA-256 protein-ID
hold-out, seed 42) and self-checks the resulting counts, so the decoder, BLOSUM,
and lineage experiments see exactly the proteins the surrogate never trained on.

The three experiments:

- **Sequence reconstruction** - an autoregressive decoder ([`decoder.py`](decoder.py))
  is trained on each encoder's embeddings and scored by per-position identity and
  per-sequence substitutions ([`train_decoder.py`](train_decoder.py),
  [`evaluate_decoder.py`](evaluate_decoder.py)).
- **Biological distance preservation** - pairwise embedding distances (cosine / L2)
  are correlated against normalized BLOSUM62 distances
  ([`evaluate_pairwise.py`](evaluate_pairwise.py)).
- **Lineage classification** - a linear softmax probe predicts the WHO variant
  group from mean-pooled embeddings ([`lineage_classification.py`](lineage_classification.py)).

Figures are composed by [`make_paper_figures.py`](make_paper_figures.py) (BLOSUM
hexbins, per-variant F1 with confusion matrices, and reconstruction mutation
heatmaps).

### Running the pipeline

The numbered scripts in [`scripts/`](scripts) orchestrate the stages; they read
hyperparameters from [`configs/`](configs) (`encoders.json`, `decoder.json`,
`blosum.json`, `lineage.json`), which are the single source of truth. Run them
from this `downstream_validation/` directory:

```bash
cd downstream_validation
GPU=1 bash scripts/00_embed_configs.sh   # cache teacher + AWN embeddings
GPU=1 bash scripts/10_train_decoders.sh  # train reconstruction decoders
GPU=1 bash scripts/20_eval_decoders.sh   # evaluate reconstruction
GPU=1 bash scripts/30_blosum.sh          # BLOSUM distance preservation (AWN)
GPU=1 bash scripts/35_lineage.sh         # lineage probe
bash scripts/40_make_figures.sh          # render figures
```

`scripts/run_all.sh` runs every stage, and `scripts/run_encoder.sh <encoder>`
runs one encoder's full sub-pipeline on one GPU (for parallelizing across GPUs).
Useful overrides: `GPU`, `PY`, `FASTA`, `EMB_DIR`, `RESULTS_DIR`, `PAPER_IMAGES`,
`ENCODERS`, `AWN_ENCODERS`, `SAMPLE` (BLOSUM subsample), `AR_LIMIT` (decoder-eval
subsample). Each stage can also be invoked directly from the repository root,
e.g. `python -m downstream_validation.evaluate_pairwise --help`; pass data and
output paths relative to that working directory.

### Outputs

This code release contains no FASTA, exported weights, cached embeddings, or
generated validation results. Supply the data and export the trained models
before running the validation pipeline. The numbered scripts regenerate
`results/paper/` and `results/figures/`, including confusion matrices and
reconstruction FASTA files. Use `FASTA=../protein_data/filtered_msa.fasta` when
running from this directory. Direct Python modules should be invoked from the
project root with explicit project-relative input and output arguments.
