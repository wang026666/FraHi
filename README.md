# FraHi

**Hierarchical fragment-constrained residual learning for tandem mass spectrum prediction.**

FraHi predicts tandem mass spectra from molecular structures and collision-energy
metadata. It combines an HDSE molecular graph encoder, a dense spectrum predictor,
fragment-indexed local residual refinement, and a hierarchical residual adapter.
The code also supports spectrum-similarity evaluation and CASMI-style candidate
ranking.

The project is now named **FraHi**. The full model can be selected with
`model.prediction_head: frahi`; the legacy value `hipeak` remains accepted for
configuration and checkpoint compatibility.

> Start with [`config/nist20_P_frahi.yml`](config/nist20_P_frahi.yml) for the full
> model. The original dataset configurations explicitly select the dense baseline.
> The current fragment mass calculation supports positive `[M+H]+` spectra;
> negative-ion configurations are dense baseline examples.

## Model Overview

1. **Molecular encoding.** HDSE supplies atom, community, and molecular context
   from the input molecular graph.
2. **Dense prediction.** A full-grid predictor provides the base spectrum.
3. **Local refinement.** Deterministic bridge-bond cleavages and hydrogen shifts
   supply fragment candidates at which signed intensity corrections are applied.
4. **Hierarchical refinement.** A residual adapter uses community and molecular
   context to refine the local corrections.
5. **Validation-based output selection.** The validation loss selects either the
   full prediction or its dense baseline. This choice is saved with the model
   and used during evaluation.

Fragment candidates are an incomplete set of correction coordinates. They do
not define the full output support or uniquely identify physical fragmentation
pathways. The non-degradation loss and validation fallback are safeguards;
they do not guarantee improvement on every unseen spectrum. Here, residual
learning refers to additive refinements, not reinforcement learning.

---

## Repository Structure

```text
FraHi/
├── Prepare_scripts/             # Raw-spectrum parsing and dataset preparation
├── config/
│   ├── nist20_P_frahi.yml        # Full FraHi, NIST20 positive-ion example
│   ├── mona_P_frahi.yml          # Full FraHi, MoNA positive-ion example
│   ├── massbank_P_frahi.yml      # Full FraHi, MassBank positive-ion example
│   ├── nist20_P.yml / nist20_N.yml
│   ├── mona_P.yml / mona_N.yml   # Original dense baseline configurations
│   ├── massbank_P.yml / massbank_N.yml
│   └── casmi22.yml              # Dense CASMI evaluation template
├── src/
│   ├── train.py                 # CLI and configuration validation
│   ├── runner.py                # Training, evaluation, and output selection
│   ├── model.py                 # Molecular encoder and prediction branches
│   ├── peak_decoder.py          # Fragment refinement and hierarchy adapter
│   ├── fragment_candidates.py   # Deterministic fragment candidate construction
│   ├── dataset.py               # Dataset and challenge loaders
│   └── ...
├── tests/                       # Naming compatibility and CASMI input checks
├── requirements.txt
└── README.md
```

Datasets, trained weights, graph caches, and experiment results are supplied
separately. There are no bundled pretrained weights or benchmark result files.

---

## Installation

Run commands from the project root after downloading or cloning the repository.
Use Python 3.10 for the dependency versions below.

```bash
cd FraHi
conda create -n frahi python=3.10
conda activate frahi
```

### Linux / Windows with CUDA 11.7

The following preserves the project's existing PyTorch 1.13 dependency family:

```bash
python -m pip install torch==1.13.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

python -m pip install --only-binary=:all: \
  torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

python -m pip install -r requirements.txt
python -m pip check
```

PyG uses the `torch-1.13.0` wheel index for PyTorch 1.13.*. The extension wheels
must match the PyTorch and CUDA build. See the official
[PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/)
and [PyG 2.6.1 installation guide](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html).

### CPU and macOS

For Linux/Windows CPU execution, use `torch==1.13.1+cpu`, the PyTorch
`https://download.pytorch.org/whl/cpu` index, and the PyG
`https://data.pyg.org/whl/torch-1.13.0+cpu.html` wheel index in the commands above.
For macOS, use the platform-specific instructions in the linked documentation;
availability of extension wheels depends on the CPU architecture. Select CPU
execution with `-d -1`. Python 3.12+ is outside this environment recipe because
the source still imports `distutils.util` and uses older dependency pins.

Check that the training modules import after installing the full environment:

```bash
python -c "import sys; sys.path.insert(0, 'src'); import runner; print('FraHi imports OK')"
```

### Lightweight Configuration Check

To inspect a configuration without installing the model stack or downloading
data, only PyYAML is required:

```bash
python -m pip install pyyaml==6.0.2
python src/train.py -c config/nist20_P_frahi.yml --dry_run
```

`--dry_run` validates settings and prints resolved paths. It does not load
spectra, open a checkpoint, test GPU availability, or run a model.

---

## Data Preparation

### 1. Obtain Spectrum Data

Obtain spectra and molecular structures from the data source used in your
experiment. Public sources include [MoNA](https://mona.fiehnlab.ucdavis.edu/) and
[MassBank](https://massbank.eu/MassBank/). NIST data must be obtained separately
under the applicable dataset terms.

The parser accepts MSP/MSPEC files with peaks, molecular SMILES, precursor
information, instrument metadata, ion mode, and collision energy. NIST-style
inputs can instead use matching `ID*.MOL` files. The parsing CLI requires
embedded SMILES or local MOL structures; resolve InChIKey-only inputs before
running it. Review source-specific field names against
`Prepare_scripts/parse_and_export.py`; database exports can have different schemas.

```text
data/
├── raw/                 # Original spectra and optional MOL files
├── df/                  # Parsed JSON tables
└── proc/
    ├── nist20/          # Processed NIST tables
    ├── mona/            # Processed MoNA tables
    └── massbank/        # Processed MassBank tables
```

### 2. Parse Raw Spectra

```bash
python Prepare_scripts/parse_and_export.py \
  --msp_file data/raw/example.msp \
  --output_dp data/df \
  --output_name nist_20_hr_records_df
```

Use `--mol_dir data/raw/mol` for separate NIST MOL structures. `--msp_file`
also accepts multiple files or directories. For MoNA or MassBank, use
`--output_name mona_df` or `--output_name massbank_df`.

### 3. Preprocess Spectra and Molecules

```bash
python Prepare_scripts/prepare_data.py \
  --df_dp data/df \
  --dset_names nist_20_hr_records \
  --proc_dp data/proc/nist20 \
  --nist_version 20
```

This writes `spec_20_df.pkl`, `mol_20_df.pkl`, `all_smiles.txt`, and
`all_inchi.tsv`. `nist_version` is the numeric processed-file suffix used by
the existing loader; it must agree between preparation and configuration,
including for non-NIST datasets.

| Dataset | Parsed JSON | `--dset_names` / `data.primary_dset` label | Processed directory |
| --- | --- | --- | --- |
| NIST20 | `nist_20_hr_records_df.json` | `nist_20_hr_records` | `data/proc/nist20` |
| MoNA | `mona_df.json` | `mona` | `data/proc/mona` |
| MassBank | `massbank_df.json` | `massbank` | `data/proc/massbank` |

For the other datasets, substitute the corresponding label and processed
directory in the preprocessing command. Separate output directories prevent
one dataset's `spec_20_df.pkl` and `mol_20_df.pkl` from overwriting another's.

---

## Training and Evaluation

### Train FraHi

The complete positive-ion configurations are ready to use after data preparation:

```bash
python src/train.py -c config/nist20_P_frahi.yml --dry_run

python src/train.py \
  -c config/nist20_P_frahi.yml \
  --data_dir data/proc/nist20 \
  --output_dir outputs/nist20_P_frahi \
  -d 0
```

Choose `config/mona_P_frahi.yml` or `config/massbank_P_frahi.yml` for the other
positive-ion examples. Each includes fragment candidates, protected joint
training, local and hierarchical residual settings, and validation fallback.
No manual YAML merging is needed.

These are configuration examples, not a complete record of published experiments.
They retain the original dataset filters, split fractions, and epoch budgets:
100 epochs for NIST20, 30 for MoNA, and **1 epoch for the MassBank example**.
Set the intended training budget before a full experiment. The examples filter
for FT/HCD spectra; inspect ion types, collision-energy fields, and instrument
filters against your input data.

Joint training starts from scratch with `run.pretrained: null`. With
`data.hdse_graph_cache: null`, graphs and fragment candidates are built on demand.
A compatible cache path is resolved relative to `data.proc_dp`.

### Model Variants and Compatibility

| Model | `model.prediction_head` | Prediction path |
| --- | --- | --- |
| Dense-HDSE | `dense` | Dense full-grid predictor |
| Plain-Peak-HDSE | `plain_peak` | Dense predictor plus local fragment refinement |
| FraHi | `frahi` | Dense predictor plus local and hierarchical refinement |
| Legacy full-model alias | `hipeak` | Same architecture as `frahi` |

The original `nist20_P.yml`, `nist20_N.yml`, `mona_P.yml`, `mona_N.yml`,
`massbank_P.yml`, and `massbank_N.yml` explicitly select `dense`. For a controlled
ablation, start from the same FraHi configuration and change the head while
keeping the data protocol, splits, and relevant optimization settings aligned.
The original dense examples use the legacy collision-energy merge by default;
they are not automatically matched ablation protocols.

The internal `hipeak_decoder.*` checkpoint keys and
`hipeak-hdse-graph-cache` format identifier are retained intentionally. Existing
full-model checkpoints can use the new name when architecture, feature setup,
and spectrum dimensions match. Dense checkpoints are not interchangeable with
FraHi checkpoints.

### Split Seeds and Resume

```bash
# Run one explicitly selected split.
python src/train.py -c config/nist20_P_frahi.yml --split_seed 520 -d 0

# Run the existing five-seed sequence: 520, 521, 522, 523, 524.
python src/train.py -c config/nist20_P_frahi.yml --num_splits 5 -d 0
```

To resume, pass `--resume_dir` with an existing run directory containing
`chkpt.pkl` and use the same configuration and split seed. Resume requires a
single split and cannot be combined with `-k`.

### Evaluate a Checkpoint

Use the exact training configuration and the selected `best_model_state.pth`.
The following creates an evaluation configuration from the NIST20 FraHi example;
if training used a modified configuration, substitute that file as the source.

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path('config/nist20_P_frahi.yml').read_text())
cfg['run_name'] = 'nist20_P_frahi_evaluate'
cfg['run'].update(num_epochs=0, do_test=True, test_sets=['test'],
                  do_casmi22=False, pretrained=None, resume_dir=None)
Path('config/evaluate.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

python src/train.py \
  -c config/evaluate.yml \
  -k checkpoints/best_model_state.pth \
  --data_dir data/proc/nist20 \
  --output_dir outputs/evaluate \
  -d 0
```

Place your trained state dict at the checkpoint path or replace it with the
actual file path. Keep the same dataset contents, filters, split settings,
model dimensions, and preprocessing. `-k` expects a model state dict;
`chkpt.pkl` is the separate resume checkpoint.

### CASMI-Style Candidate Ranking

The preparation script expects the following layout:

```text
data/raw/casmi_2022/
├── spectra/Challenge-001.txt        # Tab-separated m/z and intensity, no header
├── candidates/Challenge-001.csv     # SMILES column
└── challenge_solutions.csv          # ChallengeName, PRECURSOR_MZ, ION_MODE, SMILES
```

Use matching three-digit challenge identifiers. `ION_MODE` accepts `POSITIVE`
or `NEGATIVE`, ignoring surrounding whitespace and letter case. This is a
solution-aware retrospective evaluation workflow: the preparation script adds
the known solution to each candidate set. It is not a blind challenge submission.

```bash
python Prepare_scripts/prepare_casmi_data.py \
  --raw_dp data/raw \
  --proc_dp data/proc/nist20 \
  --casmi_input_dir casmi_2022 \
  --casmi_output_dir casmi_2022
```

The loader requires the base processed tables and `spec_df.pkl`, `mol_df.pkl`,
and `cand_df.pkl` under the challenge subdirectory. It retains positive
`[M+H]+` queries. Adapt other challenge exports to the schema above.

For the provided NIST20 FraHi configuration (`ce_key: nce`), create the challenge
configuration from the exact training configuration. This example preserves its
model dimensions and adds the required fingerprint and loader settings:

```bash
python - <<'PYCONFIG'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path('config/nist20_P_frahi.yml').read_text())
cfg['run_name'] = 'casmi22_frahi'
cfg['data'].update(casmi22_dp='casmi_2022', casmi_fp=True)
cfg['run'].update(num_epochs=0, do_test=False, do_casmi22=True,
                  casmi_batch_size=128, casmi_num_workers=0,
                  casmi_save_sim=True, pretrained=None, resume_dir=None)
Path('config/casmi22_frahi_evaluate.yml').write_text(
    yaml.safe_dump(cfg, sort_keys=False))
PYCONFIG

python src/train.py \
  -c config/casmi22_frahi_evaluate.yml \
  -k checkpoints/best_model_state.pth \
  --data_dir data/proc/nist20 \
  --output_dir outputs/casmi22_frahi \
  -d 0
```

Review collision-energy values in `data.casmi22_nces` for the evaluation protocol.
The current challenge loader populates NCE metadata. The MoNA and MassBank
examples use ACE; evaluating their checkpoints requires challenge energy metadata
and preprocessing consistent with training, not simply changing `ce_key`.
`config/casmi22.yml` is a separate **dense, 1-Da** template; the provided FraHi
training examples use **0.1-Da** bins, so switching directly to this template
would create an incompatible model.

### Saved Outputs

Completed runs create a uniquely named subdirectory beneath the output directory.
Depending on the configuration, it contains:

| File | Purpose |
| --- | --- |
| `best_model_state.pth` | Selected model state, including the output-selection flag |
| `chkpt.pkl` | Training state for resume |
| `resolved_config.json` | Resolved data, model, and runtime settings |
| `run_summary.json` | Completion status, selected prediction source, and metrics |
| `split_id.pkl` | Saved split identifiers |
| `save_tables/*.pt` | Saved test similarity tables |

The similarity metrics include cosine, modified cosine, and spectral entropy.
Report the selected output source and data/split protocol alongside comparisons.

---

## Checks and Visualization

Run the focused release regression checks with:

```bash
python -m unittest discover -s tests -v
```

Configuration and CASMI label checks do not need datasets. Model compatibility
checks require the model dependencies and otherwise report a skip. These checks
cover naming and input handling; they do not establish full training
reproducibility or benchmark performance.

Attention plotting helpers remain in `src/model.py`. The forward-pass plotting
block is commented out, so attention export requires connecting those helpers
to an evaluation pass with the required graph attributes. There is no standalone
attention visualization command in this release.

---

## Acknowledgments

This implementation uses HDSE molecular graph representations and builds on
PyTorch, PyTorch Geometric, RDKit, NetworkX, and the libraries listed in
`requirements.txt`. See also [MassFormer](https://github.com/Roestlab/massformer)
for related graph-transformer work on tandem mass spectrum prediction.
