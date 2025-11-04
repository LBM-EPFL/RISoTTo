from typing import List
import os

import numpy as np
import gemmi
from gemmi import cif
from src.data_encoding import std_aa

def read_pdb(pdb_filepath):
    # read pdb
    doc = gemmi.read_pdb(pdb_filepath, max_line_length=80)

    # altloc memory
    altloc_l = []
    icodes = []

    # data storage
    atom_element = []
    atom_name = []
    atom_xyz = []
    residue_name = []
    seq_id = []
    het_flag = []
    chain_name = []
    bfactor = []
    # parse structure
    for mid, model in enumerate(doc):
        for a in model.all():
            # altloc check (keep first encountered)
            if a.atom.has_altloc():
                key = f"{a.chain.name}_{a.residue.seqid.num}_{a.atom.name}"
                if key in altloc_l:
                    continue
                else:
                    altloc_l.append(key)

            # insertion code (skip)
            icodes.append(a.residue.seqid.icode.strip())

            # store data
            atom_element.append(a.atom.element.name)
            atom_name.append(a.atom.name)
            atom_xyz.append([a.atom.pos.x, a.atom.pos.y, a.atom.pos.z])
            residue_name.append(a.residue.name)
            seq_id.append(a.residue.seqid.num)
            het_flag.append(a.residue.het_flag)
            chain_name.append(f"{a.chain.name}:{mid}")
            bfactor.append(a.atom.b_iso)
            
    # pack data
    return {
        'xyz': np.array(atom_xyz, dtype=np.float32),
        'name': np.array(atom_name),
        'element': np.array(atom_element),
        'resname': np.array(residue_name),
        'resid': np.array(seq_id, dtype=np.int32),
        'het_flag': np.array(het_flag),
        'chain_name': np.array(chain_name),
        'icode': np.array(icodes),
        'bfactor': np.array(bfactor),
    }


def read_molecule_cif(filepath):
    # read cif
    doc = cif.read_file(filepath)

    # parse id
    molid = doc[0].find_value('_chem_comp.id')

    # parse coordinates
    xyz = np.array([
        [x for x in doc[0].find_loop('_chem_comp_atom.model_Cartn_x')],
        [x for x in doc[0].find_loop('_chem_comp_atom.model_Cartn_y')],
        [x for x in doc[0].find_loop('_chem_comp_atom.model_Cartn_z')],
    ]).T

    # if missing coordinates use ideal coordinates instead
    if not (np.float64 == xyz.dtype):
        if np.any(xyz == "?"):
            xyz = np.array([
                [x for x in doc[0].find_loop('_chem_comp_atom.pdbx_model_Cartn_x_ideal')],
                [x for x in doc[0].find_loop('_chem_comp_atom.pdbx_model_Cartn_y_ideal')],
                [x for x in doc[0].find_loop('_chem_comp_atom.pdbx_model_Cartn_z_ideal')],
            ]).T

    # single atom case
    if xyz.shape[0] == 0:
        mol = {
            "xyz": np.zeros((1,3)),
            "element": np.array([doc[0].find_value('_chem_comp_atom.type_symbol')]),
        }
    else:
        mol = {
            "xyz": xyz.astype(float),
            "element": np.array([x for x in doc[0].find_loop('_chem_comp_atom.type_symbol')]),
        }

    return mol, molid


def extract_chains_pdb(pdb_filepath, output_dir):
    """
    Extract RNA chains from a PDB file and save them as separate PDB files.
    
    Args:
        pdb_filepath (str): Path to the input PDB file
        output_dir (str): Directory where extracted chains will be saved
    
    Returns:
        List[str]: List of paths to the extracted chain PDB files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract PDB ID from filename
    pdb_id = os.path.basename(pdb_filepath).split('.')[0]
    
    # Read PDB using existing function
    pdb_data = read_pdb(pdb_filepath)
    
    
    # Get unique chains
    unique_chains = np.unique([c.split(':')[0] for c in pdb_data['chain_name']])
    
    extracted_files = []
    
    for chain_id in unique_chains:
        # Filter data for this chain
        chain_mask = np.array([c.split(':')[0] == chain_id for c in pdb_data['chain_name']])
        
        # Check if this is an RNA chain by looking for RNA residues
        residue_names = pdb_data['resname'][chain_mask]
        if not any(rn in ['A', 'G', 'C', 'U'] for rn in residue_names):
            continue
        
        # Create subunit data for this chain
        chain_data = {
            'xyz': pdb_data['xyz'][chain_mask],
            'name': pdb_data['name'][chain_mask],
            'element': pdb_data['element'][chain_mask],
            'resname': pdb_data['resname'][chain_mask],
            'resid': pdb_data['resid'][chain_mask],
            'het_flag': pdb_data['het_flag'][chain_mask],
            'bfactor': pdb_data['bfactor'][chain_mask]
        }
        
        # Save chain to PDB file
        output_filename = f"{pdb_id}_{chain_id}.pdb"
        output_path = os.path.join(output_dir, output_filename)
        save_pdb({chain_id: chain_data}, output_path)
        extracted_files.append(output_path)
    
    return extracted_files

def save_pdb(subunits, filepath):
    # open file stream
    with open(filepath, 'w') as fs:
        for cn in subunits:
            # extract data
            N = subunits[cn]['xyz'].shape[0]
            for i in range(N):
                h = "ATOM" if subunits[cn]['het_flag'][i] == 'A' else "HETATM"
                n = subunits[cn]['name'][i]
                rn = subunits[cn]['resname'][i]
                e = subunits[cn]['element'][i]
                ri = subunits[cn]['resid'][i]
                xyz = subunits[cn]['xyz'][i]
                if "bfactor" in subunits[cn]:
                    bf = subunits[cn]['bfactor'][i]
                else:
                    bf = 0.0

                # extract single character chain name
                c = cn.split(':')[0][0]

                # format pdb line
                # pdb_line = "{:<6s}{:>5d} {:<4s} {:>3s} {:1s}{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:<2s}  ".format(h, i+1, n, rn, c, ri, xyz[0], xyz[1], xyz[2], 0.0, bf, e)
                pdb_line = "{:<6s}{:>5d}  {:<4s}{:>3s} {:1s}{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:<2s}  ".format(h, i+1, n, rn, c, ri, xyz[0], xyz[1], xyz[2], 0.0, bf, e)

                # write to file
                fs.write(pdb_line+'\n')
            fs.write("TER\n")
        fs.write("END")


def save_traj_pdb(subunits, filepath):
    # determine number of frames
    for cn in subunits:
        assert len(subunits[cn]['xyz'].shape) == 3, "no time dimension"
        num_frames = subunits[cn]['xyz'].shape[0]

    # open file stream
    with open(filepath, 'w') as fs:
        for k in range(num_frames):
            fs.write("MODEL    {:>4d}\n".format(k))
            for cn in subunits:
                assert num_frames == subunits[cn]['xyz'].shape[0], "mismatching number of frames"
                # extract data
                N = subunits[cn]['xyz'][k].shape[0]
                for i in range(N):
                    h = "ATOM" if subunits[cn]['het_flag'][i] == 'A' else "HETATM"
                    n = subunits[cn]['name'][i]
                    rn = subunits[cn]['resname'][i]
                    e = subunits[cn]['element'][i]
                    ri = subunits[cn]['resid'][i]
                    xyz = subunits[cn]['xyz'][k][i]
                    if "bfactor" in subunits[cn]:
                        bf = subunits[cn]['bfactor'][i]
                    else:
                        bf = 0.0

                    # extract single character chain name
                    c = cn.split(':')[0][0]

                    # format pdb line
                    # pdb_line = "{:<6s}{:>5d} {:<4s} {:>3s} {:1s}{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:<2s}  ".format(h, i+1, n, rn, cn, ri, xyz[0], xyz[1], xyz[2], 0.0, bf, e)
                    pdb_line = "{:<6s}{:>5d}  {:<4s}{:>3s} {:1s}{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:<2s}  ".format(h, i+1, n, rn, c, ri, xyz[0], xyz[1], xyz[2], 0.0, bf, e)

                    # write to file
                    fs.write(pdb_line+'\n')
                fs.write("TER\n")
            fs.write("ENDMDL\n")
        fs.write("END")
