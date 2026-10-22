import numpy as np
import gemmi

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


