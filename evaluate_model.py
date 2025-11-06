import os
import numpy as np
import torch as pt
from tqdm import tqdm
import pandas as pd
import subprocess

import biotite
import biotite.structure as struc
import biotite.structure.io as strucio

import src as sp
import runtime as rt


def process_structure(pdb_path, model):
    """Process a PDB structure and get sequence predictions."""
    pdb_id = os.path.basename(pdb_path).split(".")[0]

    try:
        structure = rt.load_structure(pdb_path)
    except Exception as e:
        print(f"Skipping {pdb_path}, cannot read: {e}")
        return None, None, None

    # Model inference
    try:
        known_chains = [""]
        m_known = np.isin(
            [cn.split(":")[0] for cn in structure["chain_name"]], known_chains
        )
        _, p, y = model(structure, m_known=m_known)
        p, y = rt.aa_only(p, y)  # pssm for RNA
        # print(p)
        seq_ref = rt.pred_to_seq(y)
        seq_pred = rt.pred_to_seq(p)
        recovery_rate = rt.recovery_rate(y, p)
        return pdb_id, seq_ref, seq_pred, recovery_rate
    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
        return None, None, None, None


def extract_chain_data(pdb_path):
    """Extract RNA chain data from PDB file using biotite."""
    try:
        # Try loading with the default method
        atom_array = strucio.load_structure(pdb_path)
        nucleotides = atom_array[struc.filter_nucleotides(atom_array)]
    except TypeError:
        try:
            # Try loading first model if multiple models exist
            atom_array = strucio.load_structure(pdb_path)[0]
            nucleotides = atom_array[struc.filter_nucleotides(atom_array)]
        except (TypeError, IndexError):
            print(f"Skipping {pdb_path} - biotite can't load or index error")
            return None
    except (IndexError, biotite.InvalidFileError):
        print(f"Skipping {pdb_path} - biotite file error")
        return None

    # Filter for RNA nucleotides
    nucleotides = nucleotides[np.isin(nucleotides.res_name, ["G", "A", "C", "U"])]
    chain_ids = sorted(set(nucleotides.chain_id))

    return nucleotides, chain_ids


def predict_secondary_structure(
    fasta_path,
    eternafold_path="software/EternaFold/src/contrafold",
    params_path="software/EternaFold/parameters/EternaFoldParams.v1",
):
    """Run EternaFold to predict RNA secondary structure."""
    temp_output = "tmp.txt"
    cmd = (
        f"{eternafold_path} predict {fasta_path} --params {params_path} > {temp_output}"
    )

    # os.system(cmd)
    subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Parse the output to extract the secondary structure
    ss_lines = []
    try:
        with open(temp_output, "r") as f:
            for line in f:
                if (
                    any(char in line for char in [".", "(", ")"])
                    and "Predicting" not in line
                ):
                    ss_lines.append(line)
    except Exception as e:
        print(f"Error reading EternaFold output: {e}")

    return ss_lines


def write_predictions(pdb_id, chain_id, chain_seq_pred, output_dir="predictions/test"):
    """Write predicted sequence and structure to FASTA file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    fasta_file_path = f"{output_dir}/{pdb_id}_{chain_id}-predicted.fasta"

    # Write sequence
    with open(fasta_file_path, "w") as fasta_file:
        fasta_file.write(f">{pdb_id}_{chain_id}\n")
        fasta_file.write(chain_seq_pred + "\n")

    # Predict secondary structure
    ss_lines = predict_secondary_structure(fasta_file_path)

    # Append secondary structure to FASTA file
    with open(fasta_file_path, "a") as fasta_file:
        for line in ss_lines:
            fasta_file.write(line)

    return fasta_file_path


def main():
    """Main function to process the RNA test set."""
    # Set up CUDA environment
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Configuration
    test_set_path = "datasets/grnade_das_test_set.txt"
    pdb_dir = "../../data/all_bioassemblies/pdbs/"
    output_dir = "predictions/test/per_chain/"
    max_size_bytes = 1.5 * 1024 * 1024  # 1.5mb
    model_path = "save"
    model_checkpoint = "model_ckpt.pt"

    # Load test set
    pdb_filepaths = []
    sid_selection = []
    with open(test_set_path, "r") as f:
        for line in f:
            pdb_id = line.split("_")[0]
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            pdb_filepaths.append(pdb_path)
            sid_selection.append(line.strip())

    # Remove duplicates and filter by size
    pdb_filepaths = list(set(pdb_filepaths))
    pdb_filepaths = [
        filepath
        for filepath in pdb_filepaths
        if os.path.exists(filepath) and os.path.getsize(filepath) < max_size_bytes
    ]

    print(f"Number of files: {len(pdb_filepaths)}")

    # Set up device
    device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
    if pt.cuda.is_available():
        print(f"Using GPU: {pt.cuda.get_device_name(device)}")
    else:
        print("Using CPU")

    # Load model
    model = rt.SequenceModel(model_path, model_checkpoint, device=device)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each PDB file
    for pdb_path in tqdm(pdb_filepaths, desc="Processing PDB files"):
        # Get sequence predictions
        pdb_id, seq_ref, seq_pred, recovery_rate = process_structure(pdb_path, model)
        if pdb_id is None:
            continue
        else:
            # with open(f'predictions/{pdb_id}.fasta', 'w') as f:
            rt.write_fasta(
                filepath=f"predictions/{pdb_id}.fasta",
                seq=seq_pred,
                info=f"{pdb_id} | rr={recovery_rate.item():.4f}",
            )

        # Extract chain data
        chain_data = extract_chain_data(pdb_path)
        if chain_data is None:
            continue

        nucleotides, chain_ids = chain_data
        seq_offset = 0

        # Process each chain
        for chain_id in chain_ids:
            print(f"Processing {pdb_id}_{chain_id}")
            chain_nucleotides = nucleotides[nucleotides.chain_id == chain_id]
            chain_seq_len = len(
                set(chain_nucleotides.res_id)
            )  # Unique residues in the chain

            # Get the corresponding predicted sequence for the current chain
            chain_seq_pred = seq_pred[seq_offset : seq_offset + chain_seq_len]
            seq_offset += chain_seq_len  # Update offset for the next chain

            # Write predictions
            fasta_path = write_predictions(pdb_id, chain_id, chain_seq_pred, output_dir)
            print(f"Predictions written to {fasta_path}")


if __name__ == "__main__":
    main()
