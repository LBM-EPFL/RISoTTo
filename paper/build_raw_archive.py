#!/usr/bin/env python3
"""Assemble the raw inputs needed to recompute the paper metric tables into paper/raw/
and pack them as a tarball for Zenodo.

This is only needed by the authors (it reads from the original working directories).
Users download the resulting tarball instead, see paper/README.md.

Layout of paper/raw/:
    native_pdbs/{PDBID}.pdb                       native structures (benchmark + context test set)
    af3_inputs/{method}/{PDBID}.json              AF3 input JSONs (contain the designed sequences)
    af3/{method}/pdbs/{pdbid}_model.pdb           AF3 top-ranked model of each design
    af3/{method}/confidences/{pdbid}_confidences.json   AF3 per-atom confidences of that model
with method in {risotto, grnade, rhodesign}.

Usage:
    python paper/build_raw_archive.py \
        --run_dir /path/to/rna_v45_benchmark_grnade_das_bpRNA_cg_d5_ss64_partial_2025-07-15_13-14 \
        --pdb_dir /path/to/all_bioassemblies/pdbs \
        --tar paper/risotto_paper_raw.tar.gz
"""

import argparse
import os
import shutil
import tarfile
from glob import glob

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
METHODS = ["risotto", "grnade", "rhodesign"]


def context_test_pdbids():
    """PDB ids considered by test_model_context.py (same selection logic)."""
    sids_sel = np.genfromtxt(os.path.join(REPO, "datasets/grnade_das_test_set.txt"), dtype=np.dtype("U"))
    sids_sel = np.unique(np.array([s.split("_")[0] for s in sids_sel]))
    sids_tr = np.genfromtxt(os.path.join(REPO, "datasets/grnade_das_train_set.txt"), dtype=np.dtype("U"))
    m_tr = np.isin(sids_sel, [s.split("_") for s in sids_tr])
    return [sid.upper() for sid in sids_sel[~m_tr]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="original benchmark run directory (contains *_af_output/)")
    ap.add_argument("--pdb_dir", required=True, help="directory with native {PDBID}.pdb bioassemblies")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "raw"))
    ap.add_argument("--tar", default=None, help="optional output .tar.gz path")
    args = ap.parse_args()

    # native structures: benchmark targets, AF3-folded targets and context test set
    pdbids = set(pd.read_csv(os.path.join(HERE, "data/grnade_full_benchmark.csv"))["pdb_id"])
    for m in METHODS:
        pdbids |= {os.path.basename(f).split("_")[0].upper()
                   for f in glob(os.path.join(args.run_dir, f"{m}_af_output/pdbs/*_model.pdb"))}
    for pdbid in context_test_pdbids():
        fp = os.path.join(args.pdb_dir, f"{pdbid}.pdb")
        # test_model_context.py discards files >= 1 MB, so they are not needed
        if os.path.exists(fp) and os.path.getsize(fp) < 1e6:
            pdbids.add(pdbid)

    os.makedirs(os.path.join(args.out_dir, "native_pdbs"), exist_ok=True)
    n_native = 0
    for pdbid in sorted(pdbids):
        fp = os.path.join(args.pdb_dir, f"{pdbid}.pdb")
        if os.path.exists(fp):
            shutil.copy2(fp, os.path.join(args.out_dir, "native_pdbs"))
            n_native += 1
        else:
            print(f"missing native structure: {pdbid}")
    print(f"native_pdbs: {n_native}")

    # AF3 inputs and outputs
    for m in METHODS:
        dst_in = os.path.join(args.out_dir, "af3_inputs", m)
        dst_pdb = os.path.join(args.out_dir, "af3", m, "pdbs")
        dst_conf = os.path.join(args.out_dir, "af3", m, "confidences")
        for d in (dst_in, dst_pdb, dst_conf):
            os.makedirs(d, exist_ok=True)
        jsons = glob(os.path.join(args.run_dir, f"{m}_af_json/*.json"))
        pdbs = glob(os.path.join(args.run_dir, f"{m}_af_output/pdbs/*_model.pdb"))
        # top-level confidences of the top-ranked model (not seed samples / summaries)
        confs = [f for f in glob(os.path.join(args.run_dir, f"{m}_af_output/*/*/*_confidences.json"))
                 if "summary" not in os.path.basename(f)]
        for fs, dst in ((jsons, dst_in), (pdbs, dst_pdb), (confs, dst_conf)):
            for f in fs:
                shutil.copy2(f, dst)
        print(f"{m}: {len(jsons)} inputs, {len(pdbs)} models, {len(confs)} confidences")

    if args.tar:
        with tarfile.open(args.tar, "w:gz") as tar:
            tar.add(args.out_dir, arcname="raw")
        print(f"wrote {args.tar} ({os.path.getsize(args.tar) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
