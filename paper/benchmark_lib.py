"""Helper functions for the RISoTTo / gRNAde / RhoDesign benchmark (paper/fig_benchmark.ipynb).

Only needed to recompute the metric tables in paper/data/ from the raw data (paper/raw/).
"""

import json
import os
import tempfile
from glob import glob

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import torch as pt
from tqdm import tqdm

import runtime as rt

ATOM_NAME = "C4'"
MAX_LEN_DIFF = 5


# ---------------------------------------------------------------------------
# 2D secondary structure
# ---------------------------------------------------------------------------
def native_dotbracket(nucleotides):
    """Native 2D structure (dot-bracket) from reference nucleotides, cleaned to
    canonical '.', '(', ')' characters (anything else -> '.').

    For pseudoknotted structures biotite returns several equally optimal solutions in
    random order; the lexicographically smallest one is taken so the result is deterministic."""
    real_ss = min(struc.dot_bracket_from_structure(nucleotides))
    return "".join([c if c in [".", "(", ")"] else "." for c in real_ss])


def load_nucleotides(pdb_path):
    structure = strucio.load_structure(pdb_path)
    return structure[struc.filter_nucleotides(structure)]


def self_consistency_2d(fasta_path, nucleotides, eternafold):
    """MCC between the EternaFold-predicted 2D structure of the designed sequence and
    the native dot-bracket (trimmed to the design length)."""
    pred_ss = rt.predict_secondary_structure(fasta_path, **eternafold)
    trimmed = native_dotbracket(nucleotides)[: len(pred_ss)]
    return float(rt.self_consistency_secondary(pred_ss, trimmed))


# ---------------------------------------------------------------------------
# Sequence recovery
# ---------------------------------------------------------------------------
def recovery_positional(design_seq, native_seq):
    """Positional sequence recovery = fraction of matching positions over the
    aligned prefix (same definition as runtime.recovery_rate). Returns None if
    either sequence is empty."""
    L = min(len(design_seq), len(native_seq))
    if L == 0:
        return None
    return float(
        np.mean(np.array(list(design_seq[:L])) == np.array(list(native_seq[:L])))
    )


def design_from_afjson(json_file):
    """Read the designed RNA sequence + chain id + name from an AF3 input JSON."""
    data = json.load(open(json_file))
    rna = data["sequences"][0]["rna"]
    return rna["sequence"], rna["id"], data.get("name")


def evaluate_fasta_method(fasta_file, pdb_path, eternafold):
    """Recovery from the RISoTTo FASTA header; 2D self-consistency via EternaFold."""
    info, _ = rt.read_fasta(fasta_file)
    recovery = float(info.split("|")[1].split("=")[1])
    sc = self_consistency_2d(fasta_file, load_nucleotides(pdb_path), eternafold)
    return recovery, sc


def evaluate_afjson_method(json_file, pdb_path, eternafold):
    """Design from AF3 input JSON; recovery vs native PDB chain; 2D self-consistency via EternaFold."""
    design_seq, chain_id, name = design_from_afjson(json_file)
    nucleotides = load_nucleotides(pdb_path)
    chain = nucleotides[nucleotides.chain_id == chain_id]
    if len(chain) == 0:
        return None, None
    rids, idx = np.unique(chain.res_id, return_index=True)
    chain = chain[idx[np.argsort(rids)]]
    native_seq = "".join(chain.res_name.astype(str))
    recovery = recovery_positional(design_seq, native_seq)
    tmp_fasta = os.path.join(tempfile.gettempdir(), os.path.basename(json_file) + ".fasta")
    with open(tmp_fasta, "w") as f:
        f.write(f">{name}\n{design_seq}\n")
    sc = self_consistency_2d(tmp_fasta, nucleotides, eternafold)
    return recovery, sc


def collect(pdb_ids, design_path, pdb_dir, evaluate, eternafold, desc):
    """Apply `evaluate` to every target that has both a design and a native structure.
    `design_path` is a format string with a {pdb_id} field."""
    rec, sc = {}, {}
    for pdb_id in tqdm(pdb_ids, desc=desc):
        design = design_path.format(pdb_id=pdb_id)
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        r = s = None
        if os.path.exists(design) and os.path.exists(pdb_path):
            try:
                r, s = evaluate(design, pdb_path, eternafold)
            except Exception as e:
                print(f"Error {pdb_id}: {e}")
        rec[pdb_id] = r
        sc[pdb_id] = s
    return rec, sc


# ---------------------------------------------------------------------------
# 3D structure recovery of the AF3-folded designs
# ---------------------------------------------------------------------------
def process_designs(design_files, pdb_dir, desc="3D"):
    """C4' RMSD / GDT_TS / TM-score of each designed chain vs the native chain, after
    optimal superposition, with a shifted-alignment fallback for designs that are up to
    MAX_LEN_DIFF residues shorter than the native chain. Keyed by {PDBID}_{chain}."""
    results = {}
    for design_file in tqdm(design_files, desc=desc):
        pdb_id = os.path.basename(design_file).split("_")[0].upper()
        ref_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(ref_file):
            continue
        try:
            design_struct = rt.load_structure(design_file)
            ref_struct = rt.load_structure(ref_file)
        except Exception as e:
            print(f"load error {pdb_id}: {e}")
            continue
        design_c4 = rt.atom_select(design_struct, design_struct["name"] == ATOM_NAME)
        ref_c4 = rt.atom_select(ref_struct, ref_struct["name"] == ATOM_NAME)
        for chain in np.unique(design_c4["chain_name"]):
            pdb_chain_id = f"{pdb_id}_{chain.split(':')[0]}"
            d_chain = rt.atom_select(design_c4, design_c4["chain_name"] == chain)
            r_chain = rt.atom_select(ref_c4, ref_c4["chain_name"] == chain)
            X_design = pt.from_numpy(d_chain["xyz"])
            X_ref = pt.from_numpy(r_chain["xyz"])
            n_design, n_ref = len(X_design), len(X_ref)
            diff = n_ref - n_design
            if diff < 0 or diff > MAX_LEN_DIFF:
                continue
            rmsd = rt.compute_rmsd(X_ref[:n_design], X_design)
            gddt = rt.compute_gdt_ts(X_ref[:n_design], X_design)
            tm = rt.compute_tm_score(X_ref[:n_design], X_design)
            rmsd_s = rt.compute_rmsd(X_ref[diff:diff + n_design], X_design)
            gddt_s = rt.compute_gdt_ts(X_ref[diff:diff + n_design], X_design)
            tm_s = rt.compute_tm_score(X_ref[diff:diff + n_design], X_design)
            if rmsd_s < rmsd:
                rmsd, gddt, tm = rmsd_s, gddt_s, tm_s
            if pt.is_complex(tm):
                tm = pt.tensor(float("nan"))
            results[pdb_chain_id] = (float(rmsd), float(gddt), float(tm))
    return results


# ---------------------------------------------------------------------------
# pLDDT
# ---------------------------------------------------------------------------
def plddt_per_chain(conf_json_path):
    """Mean pLDDT per chain from an AF3 *_confidences.json file.

    Returns dict {chain_id: mean_pLDDT} (pLDDT in 0-100).
    """
    with open(conf_json_path) as f:
        data = json.load(f)
    chains = np.array(data["atom_chain_ids"])
    pld = np.array(data["atom_plddts"], dtype=float)
    out = {}
    for ch in np.unique(chains):
        out[str(ch)] = float(np.mean(pld[chains == ch]))
    return out


def plddt_map(conf_dir):
    """{PDBID}_{chain} -> mean pLDDT for all *_confidences.json files in conf_dir."""
    d_map = {}
    for cf in sorted(glob(os.path.join(conf_dir, "*_confidences.json"))):
        pdb_id = os.path.basename(cf).replace("_confidences.json", "").upper()
        for ch, val in plddt_per_chain(cf).items():
            d_map[f"{pdb_id}_{ch}"] = val
    return d_map
