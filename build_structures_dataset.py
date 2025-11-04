import os
import re
import h5py
import numpy as np
import torch as pt
from glob import glob
from tqdm import tqdm

from src.structure import clean_structure, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits, remove_duplicate_tagged_subunits, concatenate_chains, chain_name_to_index
from src.data_encoding import config_encoding, encode_structure, encode_features
from src.dataset import StructuresDataset, save_data

pt.multiprocessing.set_sharing_strategy('file_system')


config_dataset = {
    # parameters
    "max_num_atoms": 1024*8*2,
    "max_num_nn": 64,

    # input filepaths
    "pdb_filepaths": glob("/media/bibekar/e500c9b8-094e-4f88-983c-27f30285d37a/bibekar/kuma_backup/carbonara-rna/model/data/all_bioassemblies/pdbs/*.pdb"),

    # output filepath
    "dataset_filepath": "datasets/pdb_bprna_structures_nuc_v6.h5",
}


def pack_structure_data(X, qe, qr, qn, Mc, Mr):
    return {
        'X': X.cpu().numpy().astype(np.float32),
        'qe':pt.stack(pt.where(qe > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qr':pt.stack(pt.where(qr > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qn':pt.stack(pt.where(qn > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'Mc':pt.stack(pt.where(Mc), dim=1).cpu().numpy().astype(np.uint16),
        'Mr':pt.stack(pt.where(Mr), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        'qe_shape': qe.shape, 'qr_shape': qr.shape, 'qn_shape': qn.shape,
        'Mc_shape': Mc.shape, 'Mr_shape': Mr.shape,
    }


def pack_dataset_item(structure, max_num_nn, device=pt.device("cpu")):
    # extract features, encode structure
    qe, qr, qn = encode_features(structure)
    X, Mr, Mc = encode_structure(structure, with_chains=True, device=device)

    # store structure data
    return pack_structure_data(X, qe, qr, qn, Mc, Mr)


def store_dataset_item(hf, pdbid, bid, structure_data):
    # define store key
    key = f"{pdbid.upper()[1:3]}/{pdbid.upper()}/{bid}"

    # save structure data
    hgrp = hf.create_group(f"data/structures/{key}")
    save_data(hgrp, attrs=structure_data[1], **structure_data[0])

    # store metadata
    metadata = {
        'key': key,
        'csize': (np.max(structure_data[0]["Mc"], axis=0)+1).astype(int),
        'rsize': (np.max(structure_data[0]["Mr"], axis=0)+1).astype(int),
    }

    return metadata


if __name__ == "__main__":
    # pre-filter files by sizes
    pdb_filepaths = [fp for fp in config_dataset['pdb_filepaths'] if os.path.getsize(fp) < 1e6]
    print(f"Processing {len(pdb_filepaths)} files")

    # set up dataset
    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=False)
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=16, pin_memory=False, prefetch_factor=4)

    # define device
    device = pt.device("cuda")

    # process structure, compute features and write dataset
    with h5py.File(config_dataset['dataset_filepath'], 'w', libver='latest') as hf:
        # store dataset encoding
        for key in config_encoding:
            hf[f"metadata/{key}"] = config_encoding[key].astype(np.string_)

        # prepare and store all structures
        metadata_l = []
        pbar = tqdm(dataloader)
        for pdb_filepath, structure in pbar:
            # check that structure was loaded
            if structure is None:
                print(f"Skipping - structure is None: {pdb_filepath}")
                continue

            # parse filepath
            pdbid = pdb_filepath.split('/')[-1].split('.')[0]
            bid = 1

            # check size
            if structure['xyz'].shape[0] >= config_dataset['max_num_atoms']:
                print(f"Skipping - structure too large: {pdb_filepath}")
                continue

            # process structure
            structure = clean_structure(structure, rm_wat=False)

            # update molecules chains
            structure = tag_hetatm_chains(structure)

            # split structure
            subunits = split_by_chain(structure)

            # remove non atomic structures
            subunits = filter_non_atomic_subunits(subunits)

            # remove duplicated molecules and ions
            subunits = remove_duplicate_tagged_subunits(subunits)

            # check not empty
            if len(subunits) < 1:
                print(f"Skipping - no valid subunits: {pdb_filepath}")
                continue

            # concatenate into one structure
            structure = concatenate_chains(subunits)

            # change chain name to chain index
            structure = chain_name_to_index(structure)

            # pack data
            structure_data = pack_dataset_item(structure, config_dataset['max_num_nn'], device=device)

            # store data
            metadata = store_dataset_item(hf, pdbid, bid, structure_data)
            metadata_l.append(metadata)
            
            # Add counter to show progress
            # if len(metadata_l) % 100 == 0:
            #     print(f"Successfully processed {len(metadata_l)} structures")

        # store metadata
        hf['metadata/keys'] = np.array([m['key'] for m in metadata_l]).astype(np.string_)
        hf['metadata/csizes'] = np.array([m['csize'] for m in metadata_l])
        hf['metadata/rsizes'] = np.array([m['rsize'] for m in metadata_l])
