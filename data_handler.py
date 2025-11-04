import h5py
import numpy as np
import torch as pt

from torch.utils import data
from src.dataset import load_sparse_mask
from src.structure import data_to_structure
# from src.data_encoding import std_names, std_backbone, std_resnames, std_aminoacids
from src.data_encoding import std_elements, std_resnames, std_names, encode_features, encode_structure, std_rna, purine_atoms_cg, pyrimidine_atoms_cg


def extract_backbone_cg(X, qe, qr, qn, M, std_rna):

    purines = ["G", "A"]
    pyrimidines = ["U", "C"]

    # Identify RNA residues
    m_rna = pt.from_numpy(np.isin(std_resnames, std_rna)).to(qr.device)
    is_rna = pt.any(qr[:,:-1][:,m_rna] > 0.5, dim=1)

    # Create masks for residue types
    m_purine = pt.from_numpy(np.isin(std_resnames, purines)).to(qr.device)
    m_pyrimidine = pt.from_numpy(np.isin(std_resnames, pyrimidines)).to(qr.device)

    # Identify purines and pyrimidines
    is_purine = pt.any(qr[:,:-1][:,m_purine] > 0.5, dim=1)
    is_pyrimidine = pt.any(qr[:,:-1][:,m_pyrimidine] > 0.5, dim=1)

    # Create masks for the specific atoms
    m_purine_atoms = pt.from_numpy(np.isin(std_names, purine_atoms_cg)).to(qn.device)
    m_pyrimidine_atoms = pt.from_numpy(np.isin(std_names, pyrimidine_atoms_cg)).to(qn.device)

    # Match atoms to their types
    has_purine_atoms = pt.any(qn[:,:-1][:,m_purine_atoms] > 0.5, dim=1)
    has_pyrimidine_atoms = pt.any(qn[:,:-1][:,m_pyrimidine_atoms] > 0.5, dim=1)

    m = (~is_rna) | ((is_purine & has_purine_atoms) | (is_pyrimidine & has_pyrimidine_atoms))

    return X[m], qe[m], qr[m], qn[m], M[m]


def process_structure(X, qe, qr, qn, M, r):
    # create inital labels
    y = qr.clone()

    if r < 1.0:
        # randomly sample residues uniformly with ratio r
        nr_sel = max(int(np.ceil(M.shape[1]*r)), 1)
        ids_sel = pt.randperm(M.shape[1])[:nr_sel]
    else:
        ids_sel = pt.arange(M.shape[1])

    # randomly select residues and mask information
    m_sel = pt.any(M[:,ids_sel] > 0.5, dim=1)
    qr[m_sel] = 0.0

    # pack features
    q = pt.cat([qe, qr], dim=1)

    # atom to residue indexing
    _, rids_sel = pt.max(M[:,ids_sel], dim=0)

    return X, q, M, y[rids_sel], ids_sel


def random_atom_motion(X, r=0.75):
    # clip and resample random normal values
    rnv = np.sqrt(r*r/ 3.0) * pt.randn((X.shape[0]*10,X.shape[1]), device=X.device)
    m = (pt.norm(rnv, dim=1) <= r)
    rnv = rnv[m][:X.shape[0]]

    # compute displacement vectors
    dX = rnv * pt.rand((X.shape[0],1), device=X.device) * pt.rand((X.shape[0],1), device=X.device)
    dX = dX - pt.mean(dX, dim=0).unsqueeze(0)
    return dX


class Dataset(data.Dataset):
    """Dataset class for loading and processing molecular structures.

    Args:
        dataset_filepath: Path to HDF5 dataset file
        r_noise: Random noise amplitude for atom positions
        virt_cb: Whether to add virtual CB atoms
        partial: Whether to randomly remove parts of structures
    """
    def __init__(self, dataset_filepath, r_noise=0.0, virt_cb=False, partial=False):
        super(Dataset, self).__init__()
        # store dataset filepath
        self.dataset_filepath = dataset_filepath

        # store parameters
        self.r_noise = r_noise
        self.virt_cb = virt_cb
        self.partial = partial

        # preload data
        with h5py.File(dataset_filepath, 'r') as hf:
            # load keys, sizes and types
            self.keys = np.array(hf["metadata/keys"]).astype(np.dtype('U'))
            self.sizes = np.array(hf["metadata/rsizes"])

            # load parameters to reconstruct data
            self.std_elements = np.array(hf["metadata/std_elements"]).astype(np.dtype('U'))
            self.std_resnames = np.array(hf["metadata/std_resnames"]).astype(np.dtype('U'))
            self.std_names = np.array(hf["metadata/std_names"]).astype(np.dtype('U'))

        # set default selection mask
        self.m = np.ones(len(self.keys), dtype=bool)

    def get_largest(self):
        i = np.argmax(self.sizes[self.m,0])
        return self[i]

    def __len__(self):
        return len(self.keys[self.m])

    def __getitem__(self, k):
        # get corresponding interface keys
        key = self.keys[self.m][k]

        try:
            # load data
            with h5py.File(self.dataset_filepath, 'r') as hf:
                # hdf5 group
                hgrp = hf['data/structures/'+key]

                # topology
                X = pt.from_numpy(np.array(hgrp['X']).astype(np.float32))
                M = load_sparse_mask(hgrp, 'Mr').float()

                # load features
                qe = load_sparse_mask(hgrp, 'qe')
                qr = load_sparse_mask(hgrp, 'qr')
                qn = load_sparse_mask(hgrp, 'qn')

            # build structure back
            q = pt.cat([qe,qr,qn], dim=1)
            structure = data_to_structure(X.numpy(), q.numpy(), M.numpy(), std_elements, std_resnames, std_names)

            # extract features, encode structure
            qe, qr, qn = encode_features(structure)
            X, M = encode_structure(structure)

            # extract backbone
            X, qe, qr, qn, M = extract_backbone_cg(X, qe, qr, qn, M, std_rna)

            if self.partial:
                # randomly sample from X^2 with X uniform the percentage of residues to remove from the structure
                r = 1.0 - np.random.uniform(0.0, 1.0) * np.random.uniform(0.0, 1.0)
            else:
                r = 1.0

            # process structure
            m_std_na = pt.from_numpy(np.isin(std_resnames, std_rna)).to(qr.device)
            X, q, M, y, rids_sel = process_structure(X, qe, qr[:,:-1][:,m_std_na], qn, M, r)

            # random atom motion
            if self.r_noise > 0.0:
                X = X + random_atom_motion(X, r=self.r_noise)

            return X, q, M, y, rids_sel, key
        except Exception as e:
            print(f"Error loading data point with key: {key}")
            # print(f"Error message: {str(e)}")
            # raise
