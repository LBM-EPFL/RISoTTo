import numpy as np
import torch as pt

from .structure_io import read_pdb
from .structure import clean_structure, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits, chain_name_to_index, remove_duplicate_tagged_subunits, concatenate_chains


def select_by_sid(dataset, sids_sel):
    # extract sids of dataset
    sids = np.array(['_'.join([s.split(':')[0] for s in key.split('/')[1::2]]) for key in dataset.keys])

    # create selection mask
    m = np.isin(sids, sids_sel)

    return m


def select_by_max_ba(dataset, max_ba):
    # extract aids of dataset
    aids = np.array([int(key.split('/')[2]) for key in dataset.keys])

    # create selection mask
    m = (aids <= max_ba)

    return m


def load_sparse_mask(hgrp, k):
    # get shape
    shape = tuple(hgrp.attrs[k+'_shape'])

    # create map
    M = pt.zeros(shape, dtype=pt.float)
    ids = pt.from_numpy(np.array(hgrp[k]).astype(np.int64))
    M.scatter_(1, ids[:,1:], 1.0)

    return M


def save_data(hgrp, attrs={}, **data):
    # store data
    for key in data:
        hgrp.create_dataset(key, data=data[key], compression="lzf")

    # save attributes
    for key in attrs:
        hgrp.attrs[key] = attrs[key]


class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, pdb_filepaths, with_preprocessing=True):
        super(StructuresDataset).__init__()
        # store dataset filepath
        self.pdb_filepaths = pdb_filepaths

        # store flag
        self.with_preprocessing = with_preprocessing

    def __len__(self):
        return len(self.pdb_filepaths)

    def __getitem__(self, i):
        # find pdb filepath
        pdb_filepath = self.pdb_filepaths[i]

        # parse pdb
        try:
            structure = read_pdb(pdb_filepath)
        except Exception as e:
            print(f"ReadError: {pdb_filepath}: {e}")
            return None, pdb_filepath

        if self.with_preprocessing:
            # process structure
            structure = clean_structure(structure)

            # update molecules chains
            structure = tag_hetatm_chains(structure)

            # change chain name to chain index
            structure = chain_name_to_index(structure)

            # split structure
            subunits = split_by_chain(structure)

            # remove non atomic structures
            subunits = filter_non_atomic_subunits(subunits)

            # remove duplicated molecules and ions
            subunits = remove_duplicate_tagged_subunits(subunits)

            # pack structure
            structure = concatenate_chains(subunits)

            return structure
        else:
            return pdb_filepath, structure
