import argparse
import os

# Suppress specific warnings
import warnings

import numpy as np
import torch as pt
from tqdm import tqdm

import runtime as rt
import src as sp

warnings.filterwarnings(
    "ignore", message=".*Unexpected error from cudaGetDeviceCount.*"
)
warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")
warnings.filterwarnings(
    "ignore",
    message=".*torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
)


def process_structure(module, structure):
    # extract features, encode structure
    qe, qr, qn = module.encode_features(structure)
    X, Mr = module.encode_structure(structure)

    # encode chain infromation
    # _, cids = np.unique(structure['chain_name'], return_inverse=True)

    # extract backbone mask
    m = sp.backbone_mask_cg(qr, qn, module.std_rna)

    # extract backbone
    X = X[m]
    qe = qe[m]
    qr = qr[m]
    qn = qn[m]
    Mr = Mr[m]

    # get sequence
    m_std_rna = pt.from_numpy(np.isin(module.std_resnames, module.std_rna)).to(
        qr.device
    )
    y = qr[:, :-1][:, m_std_rna][pt.max(Mr.float(), dim=0)[1]]

    return X, qe, qr, Mr, y


def apply_model(model, X, qe, qr, M, yt=None):
    # mask
    if yt is None:
        qm = pt.zeros((qr.shape[0], model.module.std_rna.shape[0]))
    else:
        qm = pt.matmul(M, yt.to(qr.device))

    # apply mask and pack features
    q = pt.cat([qe, qm], dim=1)

    # run predictions
    with pt.no_grad():
        # send to device
        X, q, M = (v.to(model.device) for v in (X, q, M))

        # compute topology
        ids_topk, _, _, _, _ = model.module.extract_topology(X, 64)

        # run model
        z = model.model(X, ids_topk, q, M)
        p = pt.sigmoid(z).cpu()

    return p


def main(
    pdb_filepath, output_filepath, num_samples, imprint_ratio=0.5, sampling="max_confidence", device="cuda:0"
):
    # device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
    device = pt.device(device)
    save_path = "./save"
    model = rt.SequenceModel(save_path, "model_ckpt.pt", device=device)

    structure = rt.load_structure(pdb_filepath, rm_wat=False)
    X, qe, qr, M, y = process_structure(model.module, structure)
    m_rna = pt.sum(y, dim=1) > 0.0

    # Reference sequence and prediction
    p0 = apply_model(model, X, qe, qr, M)[m_rna]
    #np.save('7d8o_p0.npy', p0)
    sampling_method = sampling
    if sampling_method == "max_confidence":
        seq_max_conf = rt.pred_to_seq(p0)
    elif sampling_method == "probabilistic":
        seq_max_conf = rt.sample_pred_to_seq(p0)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    seq_ref = rt.pred_to_seq(y[m_rna])
    seq_recovery = rt.recovery_rate(p0, y[m_rna])
    conf_max = rt.average_maximum_confidence(p0).item()
    print(f"Reference sequence: {seq_ref}")
    print(f"Max confidence sequence: {seq_max_conf}")
    print(f"Recovery rate: {seq_recovery:.3f}")
    print(f"Max confidence: {conf_max:.3f}")

    # Prepare FASTA output
    fasta_lines = []
    fasta_lines.append(
        f">seq_wt | imprint_ratio={imprint_ratio} sampling={sampling_method}"
    )
    fasta_lines.append(seq_ref)
    fasta_lines.append(
        f">seq_0 | recovery={seq_recovery:.3f} confidence={conf_max:.3f}"
    )

    #seq_max_conf_ = "A" + seq_max_conf[:35] + "A"
    #seq_max_conf_ = "AUUU" + seq_max_conf_[4:-4] + "GAAA"  # conserve overhangs
    fasta_lines.append(seq_max_conf)
    seqs_set = set()
    seqs_set.add(seq_max_conf)

    with tqdm(total=num_samples) as pbar:
        pbar.set_description("Sampling")
        while len(seqs_set) < num_samples + 1:
            yt = pt.zeros((m_rna.shape[0], 4))
            yt[m_rna] = rt.seq_to_tensor(seq_max_conf)
            prob = pt.rand(yt.shape[0], device=yt.device)
            mask = (prob < imprint_ratio).float()
            yt = yt * mask.unsqueeze(1)

            p = apply_model(model, X, qe, qr, M, yt)[m_rna]
            imprint = mask[m_rna].bool()
            p[imprint] = yt[m_rna][imprint]

            if sampling_method=='max_confidence':
                seq_pred = rt.pred_to_seq(p)
            elif sampling_method=='probabilistic':
                seq_pred = rt.sample_pred_to_seq(p)
            else:
                raise ValueError(f"Unknown sampling method: {sampling_method}")
            
            #seq_pred = "A" + rt.pred_to_seq(p)[:35] + "A"
            #seq_pred = "AUUU" + seq_pred[4:-4] + "GAAA"  # conserve overhangs
            if seq_pred in seqs_set:
                continue
            seqs_set.add(seq_pred)

            recovery = rt.recovery_rate(p, y[m_rna]).item()
            recover_max_conf = rt.recovery_rate(p, p0).item()
            confidence = rt.average_maximum_confidence(p).item()

            fasta_lines.append(
                f">seq_{len(seqs_set) - 1} | recovery={recovery:.3f} confidence={confidence:.3f} recover_max_conf={recover_max_conf:.3f}"
            )
            fasta_lines.append(seq_pred)

            pbar.update(1)
            pbar.set_postfix({"recovery": recovery, "confidence": confidence})

    # Write to output FASTA file
    fasta_path = os.path.join(output_filepath, f"{filename}_designs.fasta")
    with open(fasta_path, "w") as f:
        f.write("\n".join(fasta_lines))
    print(f"\nSaved {len(fasta_lines) // 2} sequences to: {fasta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA sequence design using RISoTTo.")

    parser.add_argument(
        "--pdb_filepath", type=str, required=True, help="Path to input PDB file."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output FASTA."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sequences to sample (excluding max-confidence sequence).",
    )
    parser.add_argument(
        "--imprint_ratio",
        type=float,
        default=0.5,
        help="Ratio of imprinted residues (0 to 1).",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="max_confidence",
        help="Sampling method to use ('max_confidence' or 'probabilistic').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (e.g., 'cuda:0' or 'cpu').",
    )

    args = parser.parse_args()

    filename = os.path.splitext(os.path.basename(args.pdb_filepath))[0]
    main(
        args.pdb_filepath,
        args.output_dir,
        args.num_samples,
        args.imprint_ratio,
        args.sampling,
        args.device,
    )
