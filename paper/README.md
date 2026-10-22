# Reproducing the paper figures

This folder contains the data and notebooks that produce the benchmark and molecular context figures of

> Bibekar, P., Krapp, L. F., & Dal Peraro, M. (2025). Context-aware geometric deep learning for RNA sequence design. bioRxiv https://doi.org/10.1101/2025.06.21.660801

| figure (`figures/`) | notebook | data (`data/`) |
|---|---|---|
| `recovery_boxplot_risotto_grnade_rhodesign` | `fig_benchmark.ipynb` | `recovery_sc_comparison_risotto_grnade_rhodesign.csv` |
| `sc_boxplot_risotto_grnade_rhodesign` | `fig_benchmark.ipynb` | `recovery_sc_comparison_risotto_grnade_rhodesign.csv` |
| `rmsd_boxplot_risotto_grnade_rhodesign` | `fig_benchmark.ipynb` | `rmsd_gddt_tm_comparison_risotto_grnade_rhodesign.csv` |
| `plddt_boxplot_risotto_grnade_rhodesign` | `fig_benchmark.ipynb` | `rmsd_gddt_tm_comparison_risotto_grnade_rhodesign.csv` |
| `sequence_recovery_interfaces` | `fig_context_interfaces.ipynb` | `context_interfaces/*.csv` |

Each figure is saved as `.png`, `.svg` and `.pdf`.

## Quick start: figures from the precomputed tables

```bash
pip install -r requirements.txt   # from the repository root
cd paper
jupyter nbconvert --to notebook --execute --inplace fig_benchmark.ipynb
jupyter nbconvert --to notebook --execute --inplace fig_context_interfaces.ipynb
```

This only needs the tables in `data/` (committed to the repository) and takes a few seconds.

## Full reproduction from the raw data

### 1. Raw data

Download `risotto_paper_raw.tar.gz` from Zenodo (DOI: _to be added_) and extract it in this folder:

```bash
cd paper
tar -xzf risotto_paper_raw.tar.gz   # creates paper/raw/
```

| path | content |
|---|---|
| `raw/native_pdbs/{PDBID}.pdb` | native structures (biological assemblies) of the benchmark targets and of the molecular context test set |
| `raw/af3_inputs/{method}/{PDBID}.json` | AlphaFold3 input JSONs of the designed sequences (the RhoDesign sequences are read from these) |
| `raw/af3/{method}/pdbs/{pdbid}_model.pdb` | AlphaFold3 top-ranked model of each design |
| `raw/af3/{method}/confidences/{pdbid}_confidences.json` | AlphaFold3 per-atom confidences of that model |

with `method` in `risotto`, `grnade`, `rhodesign`. The RISoTTo designed sequences are in `../predictions/{PDBID}.fasta` and the gRNAde recovery and 2D self-consistency scores (computed with gRNAde) are in `data/grnade_full_benchmark.csv`.

### 2. EternaFold (2D self-consistency)

```bash
git submodule update --init software/EternaFold
cd software/EternaFold/src && make
```

### 3. Benchmark metric tables

Set `RECOMPUTE = True` in the configuration cell of `fig_benchmark.ipynb` and run it. It rebuilds the two benchmark tables in `data/` before drawing the figures (a few minutes on a CPU).

### 4. Molecular context analysis

From the repository root (a GPU is recommended):

```bash
python test_model_context.py --pdb_dir paper/raw/native_pdbs --output_dir paper/data/context_interfaces_rerun
```

The script picks 100 structures at random (fixed seed) from the test set. It is deterministic: it reseeds the model before every call and fixes the python hash seed.

### 5. AlphaFold3 folding of the designs (optional)

The AF3 predictions in `raw/af3/` were produced with the scripts in `../scripts/`:

```bash
python scripts/prepare_af3.py --pred_dir predictions/test/per_chain --pdb_dir paper/raw/native_pdbs --out_dir risotto_af_json
JSON_DIR=risotto_af_json BASE_OUTPUT_DIR=risotto_af_output sbatch scripts/run_af3.job
```

The top-ranked `*_model.cif` files were converted to PDB with `scripts/cif_to_pdb.py`.

## Notes on reproducibility

- **3D metrics, pLDDT and sequence recovery** are recomputed exactly from the raw data.
- **2D self-consistency.** For pseudoknotted native structures, biotite returns several equally optimal dot-bracket notations in random order. The original analysis took the first one, so a handful of values changed between runs. `benchmark_lib.native_dotbracket` now takes the lexicographically smallest one. The committed table was regenerated this way: 6 of the 267 values differ from the original run, and the medians shown in the figure are unchanged (RISoTTo 0.61, gRNAde 0.80, RhoDesign 0.47).
- **Molecular context.** `data/context_interfaces/` holds the per-interface results behind the published figure (142 structures). They were accumulated over several runs of `test_model_context.py` from before it was made deterministic: the model initialises its state vectors randomly and the selection depended on which structures were available. A deterministic rerun on the provided structures evaluates 88 structures (all included in the 142). Its medians are within about 0.05 of the published ones and show the same trend.
- `build_raw_archive.py` documents how `risotto_paper_raw.tar.gz` was assembled from the original working directories.
