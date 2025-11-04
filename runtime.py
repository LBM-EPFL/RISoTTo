import os
import sys
import h5py
import importlib
import numpy as np
import torch as pt
import blosum as bl
from scipy import signal
import subprocess

import src as sp
from software.ribonanzanet_sec_struct.network import RibonanzaNetSS
from software.ribonanzanet.network import RibonanzaNet

def aa_only(p: pt.Tensor, y: pt.Tensor):
    m = (pt.sum(y, dim=1) > 0.0)
    return p[m], y[m]


def recovery_rate(y: pt.Tensor, p: pt.Tensor):
    return pt.mean((pt.argmax(p, dim=1) == pt.argmax(y, dim=1)).float())


def maximum_recovery_rate(y: pt.Tensor, p: pt.Tensor):
    return pt.mean(pt.sum(pt.round(p) * y, dim=1))


def average_multiplicity(p: pt.Tensor):
    return pt.mean(pt.sum(pt.round(p), dim=1))


def average_maximum_confidence(p: pt.Tensor):
    return pt.mean(pt.max(p, dim=1)[0])


def max_pred_to_seq(p: pt.Tensor):
    return ''.join([sp.res3to1[r] for r in sp.std_rna[pt.argmax(p,dim=1).cpu().numpy()]])

def pred_to_seq(p: pt.Tensor):
    std_rna_ = ['G', 'A', 'U', 'C']
    max_indices = pt.argmax(p, dim=1)
    sequence = ''.join([std_rna_[idx] for idx in max_indices])
    return sequence

def seq_to_tensor(seq: str) -> pt.Tensor:
    std_rna_ = ['G', 'A', 'U', 'C']
    char_to_idx = {c: i for i, c in enumerate(std_rna_)}
    indices = [char_to_idx[char] for char in seq]
    one_hot = pt.nn.functional.one_hot(pt.tensor(indices), num_classes=4).float()
    return one_hot

def sequence_to_onehot(sequence: str, res3to1: dict) -> pt.Tensor:
   
   # Define the order of residues
    residue_order = ['G', 'A', 'U', 'C']

    # Convert the sequence string to a list of one-letter residue codes
    sequence_list = [res3to1[res] for res in sequence]

    # Initialize a tensor to store the one-hot encoding
    num_residues = len(residue_order)
    sequence_length = len(sequence_list)
    onehot = pt.zeros(sequence_length, num_residues)

    # Convert the sequence list to one-hot encoding
    for i, res in enumerate(sequence_list):
        if res in residue_order:
            onehot[i, residue_order.index(res)] = 1

    return onehot

def ss_to_tensor(ss: str) -> pt.Tensor:
    ss_map = {'.': 0, '(': 1, ')': 2}
    indices = [ss_map[char] for char in ss]
    one_hot = pt.nn.functional.one_hot(pt.tensor(indices), num_classes=3).float()
    return one_hot

def binary_classification(y, q):
    # flatten all predictions and labels
    y = y.flatten()
    q = q.flatten()

    TP = pt.sum(q * y)
    TN = pt.sum((1.0 - q) * (1.0 - y))
    FP = pt.sum(q * (1.0 - y))
    FN = pt.sum((1.0 - q) * y)
    P = pt.sum(y)
    N = pt.sum(1.0 - y)

    return TP, TN, FP, FN, P, N


def acc(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def f1(TP, FP, FN):
    v = 2 * TP / (2 * TP + FP + FN)
    v[pt.isinf(v)] = np.nan
    return v

def mcc(TP, TN, FP, FN):
    v = ((TP*TN) - (FP*FN)) / pt.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    v[pt.isinf(v)] = np.nan
    return v

def sample_pred_to_seq(p: pt.Tensor):
    seqm = ""
    for i in range(p.shape[0]):
        # locate positive predictions
        ids_p = pt.where(p[i] > 0.5)[0]

        # random sampling
        c = p[i][ids_p] - 0.5
        if len(c) > 0:
            k = np.random.choice(ids_p.cpu().numpy(), p=(c/pt.sum(c)).cpu().numpy())
        else:
            k = 0

        # update sequence
        seqm += sp.res3to1[sp.std_rna[k].item()]

    return seqm


def minimize_sequence_similarity(p, y):
    # sequence similarity criteria
    blm = bl.BLOSUM(62)
    seq_minsim = ""
    seq_ref = ""
    seq_score = []
    for i in range(p.shape[0]):
        # extract sequence from prediction and reference
        rs0 = sp.res3to1[sp.std_rna[pt.argmax(y[i]).cpu().numpy().item()]]
        ids_pr = pt.where(p[i] >= 0.5)[0]
        rsp_l = [sp.res3to1[r] for r in sp.std_rna[ids_pr.cpu().numpy()]]

        # compute sequence similarity and find minimum locations
        ss = np.array([blm[rs0][r] for r in rsp_l])
        ids_min = np.where(ss <= 0.0)[0]
        if len(ids_min) == 0:
            ids_min = np.where(ss == np.min(ss))[0]

        # find sequence with minimum sequence similarity and maximum probability
        k = pt.argmax(p[i][ids_pr][ids_min]).cpu().numpy().item()
        rs_ms = sp.res3to1[sp.std_rna[ids_pr[ids_min][k].cpu().item()]]

        # store results
        seq_minsim += rs_ms
        seq_ref += rs0
        seq_score.append(np.min(ss))

    return seq_minsim, seq_ref, np.array(seq_score)


def kstar(c: pt.Tensor):
    S = -pt.sum(c * pt.log2(c + 1e-6), dim=1)
    return pt.pow(pt.tensor(2.0), S)


def seq_to_features(seq):
    resnames = np.array([sp.res1to3[r] for r in list(seq)])
    return pt.from_numpy(sp.onehot(resnames, sp.std_resnames).astype(np.float32))


def sequence_identity(seq_ref, seq):
    return np.mean(np.array(list(seq_ref)) == np.array(list(seq)))


def sequence_similarity(seq_ref, seq):
    blm = bl.BLOSUM(62)
    return np.mean(np.array([blm[si][sj] for si,sj in zip(seq_ref,seq)]) > 0)


def write_fasta(filepath, seq, info=""):
    with open(filepath, 'w') as fs:
        fs.write(">{}\n{}".format(info, seq))


def read_fasta(fasta_filepath):
    # read content
    with open(fasta_filepath, 'r') as fs:
        fasta_content = fs.read().strip()[1:].split('\n')

    # parse content
    info = fasta_content[0]
    seq = ''.join(fasta_content[1:]).split(':')

    return info, seq


def traj_to_struct(traj):
    df = traj.topology.to_dataframe()[0]
    return {
        "xyz": np.transpose(traj.xyz, (1,0,2))*1e1,
        "name": df["name"].values,
        "element": df["element"].values,
        "resname": df["resName"].values,
        "resid": df["resSeq"].values,
        "het_flag": np.array(['A']*traj.xyz.shape[1]),
        "chain_name": df["chainID"].values,
        "icode": np.array([""]*df.shape[0]),
    }

def atom_select(structure, sel):
    return {key: structure[key][sel] for key in structure}

def get_TMscore(pdb_ref, pdb_pred):
    # Run USalign
    cmd = f"./software/USalign/USalign {pdb_pred} {pdb_ref}"
    result = os.popen(cmd).read()
    
    # Extract the first TM-score
    for line in result.split('\n'):
        if line.startswith('TM-score='):
            # Extract the numeric value after "TM-score="
            tm_score = float(line.split('=')[1].split()[0])
            return tm_score
        
    # if no TM-score found
    return None

def superpose(xyz_ref, xyz):
    # centering
    t = pt.mean(xyz, dim=1).unsqueeze(1)
    t_ref = pt.mean(xyz_ref, dim=1).unsqueeze(1)

    # SVD decomposition
    U, S, Vt = pt.linalg.svd(pt.matmul(pt.transpose(xyz_ref-t_ref,1,2), xyz-t))

    # reflection matrix
    Z = pt.zeros(U.shape, device=xyz.device) + pt.eye(U.shape[1], U.shape[2], device=xyz.device).unsqueeze(0)
    Z[:,-1,-1] = pt.linalg.det(U) * pt.linalg.det(Vt)

    R = pt.matmul(pt.transpose(Vt,1,2), pt.matmul(Z, pt.transpose(U,1,2)))

    return xyz_ref-t_ref, pt.matmul(xyz-t, R)

def compute_gdt_ts(xyz0: pt.Tensor, xyz1: pt.Tensor, r_thr = [1.0, 2.0, 4.0, 8.0]) -> float:
    # superpose
    xyz1_aligned, xyz0_aligned = superpose(xyz0.view(1, -1, 3), xyz1.view(1, -1, 3))

    # compute pairwise distances
    distances = pt.sqrt(pt.sum((xyz0_aligned - xyz1_aligned) ** 2, dim=2)).squeeze()

    # percentage of atoms within each threshold
    scores = []
    for threshold in r_thr:
        score = pt.sum(distances < threshold).item() / distances.shape[0] * 100
        scores.append(score)

    # compute GDT_TS as the average of scores at each threshold
    gdt_ts = sum(scores) / len(r_thr)
    return gdt_ts

def compute_rmsd(xyz0, xyz1):
    # superpose
    xyz1, xyz0 = superpose(xyz0.view(1,-1,3), xyz1.view(1,-1,3))

    # compute rmsd
    rmsd = pt.sqrt(pt.mean(pt.sum(pt.square(xyz0-xyz1), dim=2)))

    return rmsd


def compute_tm_score(X0: pt.Tensor, X1: pt.Tensor) -> pt.Tensor:
    """
    Compute TM-score between two point clouds X0 and X1 (shape: [N, 3]).
    The input coordinates are assumed to be corresponding and of equal length.
    """
    assert X0.shape == X1.shape, "Input shapes must match."

    # Add batch dimension for superpose
    X0_batched = X0.unsqueeze(0)  # (1, N, 3)
    X1_batched = X1.unsqueeze(0)  # (1, N, 3)

    # Superpose
    X0_aligned, X1_aligned = superpose(X0_batched, X1_batched)

    # Remove batch dimension
    X0_aligned = X0_aligned.squeeze(0)
    X1_aligned = X1_aligned.squeeze(0)

    # Length of the structure
    L = X0.shape[0]
    d0 = 1.24 * (L - 15)**(1/3) - 1.8
    d0 = max(d0, 0.5)

    # Compute distances after alignment
    dist = pt.sqrt(pt.sum((X0_aligned - X1_aligned)**2, dim=1))

    # TM-score formula
    score = pt.sum(1.0 / (1.0 + (dist / d0)**2)) / L

    return score


def compute_lDDT(X, X0, r_thr=[0.5, 1.0, 2.0, 4.0], R0=15.0):
    # compute distance matrices
    D = pt.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=2)
    D0 = pt.norm(X0.unsqueeze(0) - X0.unsqueeze(1), dim=2)

    # thresholds
    r_thr = pt.tensor(r_thr).to(D.device)

    # local selection mask
    M = ((D0 < R0) & (D0 > 0.0)).float()

    # compute score Local Distance Difference Test
    DD = (pt.abs(D0 - D).unsqueeze(0) < r_thr.view(-1,1,1)).float()
    lDD = pt.sum(DD * M.unsqueeze(0), dim=2) / pt.sum(M, dim=1).unsqueeze(0)
    lDDT = 1e2*pt.mean(lDD, dim=0)

    return lDDT


def process_structure(structure, rm_wat=True):
    # process structure
    structure = sp.clean_structure(structure, rm_wat=rm_wat)

    # update molecules chains
    structure = sp.tag_hetatm_chains(structure)

    # change chain name to chain index
    structure = sp.chain_name_to_index(structure)

    # split structure
    subunits = sp.split_by_chain(structure)

    # remove non atomic structures
    subunits = sp.filter_non_atomic_subunits(subunits)

    # remove duplicated molecules and ions
    subunits = sp.remove_duplicate_tagged_subunits(subunits)

    return sp.concatenate_chains(subunits)


def load_structure(pdb_filepath, rm_wat=True):
    # read structure
    structure = sp.read_pdb(pdb_filepath)

    # process structure
    structure = process_structure(structure, rm_wat=rm_wat)

    return structure


def split_by_residue(structure):
    uresids, ids = np.unique(structure['resid'], return_index=True)
    uresids = uresids[np.argsort(ids)]

    residues = [sp.atom_select(structure, structure['resid'] == resid) for resid in uresids]

    return residues


def cid_to_chain_name(structure):
    structure['chain_name'] = np.array(["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[i] for i in structure['cid']])
    return structure


def chain_masks(structure):
    _, Mr, Mc = sp.encode_structure(sp.chain_name_to_index(structure), with_chains=True)
    chain_names = [structure['chain_name'][i].split(':')[0] for i in pt.max(Mc, dim=0)[1]]
    mr_chains = (pt.matmul(Mr.T, Mc) / pt.sum(Mr, dim=0).unsqueeze(1) > 0.0)
    return mr_chains, chain_names


def subunit_type(subunit):
    if np.all([rn in sp.resname_to_categ for rn in subunit['resname']]):
        t = np.unique([sp.resname_to_categ[rn] for rn in subunit['resname'] if rn in sp.resname_to_categ])
        if len(t) == 1:
            return t.item()
        else:
            return "na"
    else:
        return "na"


def subunits_type(subunits):
    return {(subunit_type(subunits[cid]),cid) for cid in subunits}


def load_module(name, path):
    # load module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module

def predict_secondary_structure(fasta_path, eternafold_path="./software/EternaFold/src/contrafold", 
                                      params_path="./software/EternaFold/parameters/EternaFoldParams.v1"):
    """Run EternaFold to predict RNA secondary structure."""
    temp_output = "tmp.txt"
    cmd = f"{eternafold_path} predict {fasta_path} --params {params_path} > {temp_output}"
    
    # os.system(cmd)
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Parse the output to extract the secondary structure
    ss_lines = []
    try:
        with open(temp_output, 'r') as f:
            for line in f:
                if any(char in line for char in ['.', '(', ')']) and 'Predicting' not in line:
                    ss_lines.append(line)
    except Exception as e:
        print(f"Error reading EternaFold output: {e}")
    
    ss = ss_lines[-1].replace('\n', '')
    return ss

def predict_secondary_structure_ribonanzanet(fasta_path, device="cpu"):
    """Predict RNA secondary structure using RibonanzaNetSS."""
    # Load sequence from FASTA
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    
    ribonanza_net_ss = RibonanzaNetSS(
    'software/ribonanzanet_sec_struct/config.yaml',
    'software/ribonanzanet_sec_struct/ribonanzanet_ss.pt',
    device
    )
    ribonanza_net_ss = ribonanza_net_ss.to(device)
    ribonanza_net_ss.eval()
    ss = ribonanza_net_ss.predict(sequence[3:-1])[1][0]
    
    return ss

def dotbracket_to_adjacency(
        sec_struct: str,
        keep_pseudoknots: bool = False,
    ) -> np.ndarray:
    """
    Convert secondary structure in dot-bracket notation to 
    adjacency matrix.

    from https://github.com/chaitjo/geometric-rna-design/blob/main/src/data/sec_struct_utils.py
    """
    n = len(sec_struct)
    adj = np.zeros((n, n), dtype=np.int8)
        
    if keep_pseudoknots == False:
        stack = []
        for i, db_char in enumerate(sec_struct):
            if db_char == '(':
                stack.append(i)
            elif db_char == ')':
                j = stack.pop()
                adj[i, j] = 1
                adj[j, i] = 1
    else:
        stack={
            '(':[],
            '[':[],
            '<':[],
            '{':[]
        }
        pop={
            ')':'(',
            ']':'[',
            '>':"<",
            '}':'{'
        }
        for i, db_char in enumerate(sec_struct):
            if db_char in stack:
                stack[db_char].append((i, db_char))
            elif db_char in pop:
                forward_bracket = stack[pop[db_char]].pop()
                adj[forward_bracket[0], i] = 1
                adj[i, forward_bracket[0]] = 1    
    return adj

def self_consistency_secondary(ss_ref, ss_pred, keep_pseudoknots: bool = False) -> float:
    """
    Compute self-consistency score for secondary structure prediction.
    """
    # Convert secondary structure to adjacency matrix
    adj_ref = dotbracket_to_adjacency(ss_ref, keep_pseudoknots)
    adj_pred = dotbracket_to_adjacency(ss_pred, keep_pseudoknots)
    
    # Ensure both structures have the same length
    if adj_ref.shape[0] != adj_pred.shape[0]:
        raise ValueError("Reference and predicted structures have different lengths")
    
    # Convert to PyTorch tensors
    adj_ref_tensor = pt.tensor(adj_ref, dtype=pt.float)
    adj_pred_tensor = pt.tensor(adj_pred, dtype=pt.float)
    
    # Calculate binary classification metrics
    TP, TN, FP, FN, P, N = binary_classification(adj_ref_tensor, adj_pred_tensor)
    
    # Calculate Matthews Correlation Coefficient
    mcc_score = mcc(TP, TN, FP, FN).item()
    
    return mcc_score

class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, pdb_filepaths):
        super(StructuresDataset).__init__()
        # store dataset filepath
        self.pdb_filepaths = pdb_filepaths

    def __len__(self):
        return len(self.pdb_filepaths)

    def __getitem__(self, i):
        # find pdb filepath
        pdb_filepath = self.pdb_filepaths[i]

        # load structure
        structure = load_structure(pdb_filepath)

        return pdb_filepath, structure


class Dataset(pt.utils.data.Dataset):
    def __init__(self, dataset_filepath):
        super(Dataset, self).__init__()
        # store dataset filepath
        self.dataset_filepath = dataset_filepath

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

        # load data
        with h5py.File(self.dataset_filepath, 'r') as hf:
            # hdf5 group
            hgrp = hf['data/structures/'+key]

            # topology
            X = pt.from_numpy(np.array(hgrp['X']).astype(np.float32))
            Mr = sp.load_sparse_mask(hgrp, 'Mr').float()
            Mc = sp.load_sparse_mask(hgrp, 'Mc').float()

            # load features
            qe = sp.load_sparse_mask(hgrp, 'qe')
            qr = sp.load_sparse_mask(hgrp, 'qr')
            qn = sp.load_sparse_mask(hgrp, 'qn')

        # convert data to structure
        structure = sp.data_to_structure(X.numpy(), pt.cat([qe, qr, qn], dim=1).numpy(), Mr.numpy(), sp.std_elements, sp.std_resnames, sp.std_names)
        structure['cid'] = pt.argmax(Mc, dim=1).numpy()

        return key, structure


class ConfidenceMap():
    def __init__(self, cdf_filepath):
        # load prediction CDF
        Z = np.loadtxt(cdf_filepath, delimiter=",")
        self.x = Z[0]
        self.C = Z[1:]

        # smooth raw mapping (finite sampling -> noise)
        for i in range(self.C.shape[0]):
            self.C[i] = signal.savgol_filter(self.C[i], 9, 3)

    def __call__(self, p):
        # interpolated confidence
        #return np.stack([np.interp(p[:,k], self.x, self.C[k]) * np.round(p[:,k]) for k in range(p.shape[1])], axis=1)
        return np.clip(np.stack([np.interp(p[:,k], self.x, self.C[k]) for k in range(p.shape[1])], axis=1), 0.0, 1.0)


class SequenceModel():
    def __init__(self, save_path, parameters_filename, device=pt.device("cpu")):
        # load module
        self.module = load_module(os.path.basename(save_path), os.path.join(save_path, "__init__.py"))

        # create and reload model
        self.device = device
        model_filepath = os.path.join(save_path, parameters_filename)
        self.model = self.module.Model(self.module.config_model)
        self.model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))
        self.model = self.model.eval().to(self.device)

    def __call__(self, structure, m_known=None, n_skip=1):
        # extract features, encode structure
        qe, qr, qn = self.module.encode_features(structure)
        X, Mr = self.module.encode_structure(structure)
        if m_known is None:
            mr_known = pt.zeros(Mr.shape[1]).bool()
        else:
            if np.any(m_known):
                mr_known = (pt.max(Mr[pt.from_numpy(m_known).to(Mr.device)], dim=0)[0] > 0.0)
            else:
                mr_known = pt.zeros(Mr.shape[1]).bool()

        # encode chain infromation
        chain_map, cids = np.unique(structure['chain_name'], return_inverse=True)

        # extract backbone mask
        m = sp.backbone_mask_cg2(qr, qn, sp.std_rna)

        # extract backbone
        X = X[m]
        qe = qe[m]
        qr = qr[m]
        qn = qn[m]
        Mr = Mr[m]
        cids = cids[m.numpy()]

        # TODO fix with sink nodes
        if X.shape[0] < 64:
            raise ValueError("Structure is too small")
        

        # get sequence
        m_std_aa = pt.from_numpy(np.isin(self.module.std_resnames, self.module.std_rna)).to(qr.device)
        y = qr[:,:-1][:,m_std_aa][pt.max(Mr.float(), dim=0)[1]]

        # mask residue information
        yt = y.clone()
        yt[~mr_known] = 0.0

        # build structure back
        q = pt.cat([qe,qr,qn], dim=1)
        structure = self.module.data_to_structure(X.numpy(), q.numpy(), Mr.numpy(), self.module.std_elements, self.module.std_resnames, self.module.std_names)
        structure['chain_name'] = np.array([chain_map[i] for i in cids])
        structure = sp.encode_bfactor(structure, pt.sum(yt,dim=1).cpu().numpy())

        # apply mask and pack features
        qr = pt.matmul(Mr, yt)
        q = pt.cat([qe, qr], dim=1)

        # multiframe support
        if len(X.shape) < 3:
            X = X.unsqueeze(1)

        # run predictions
        P = []
        with pt.no_grad():
            for i in range(0, X.shape[1], n_skip):
                # send to device
                Xi, q, M = (v.to(self.device) for v in (X[:,i], q, Mr))

                # compute topology
                ids_topk, _, _, _, _ = self.module.extract_topology(Xi, 64)

                # run model
                z = self.model(Xi, ids_topk, q, M)

                # prediction
                p = pt.sigmoid(z).cpu()

                # store result
                P.append(p)

        return structure, pt.stack(P).squeeze(), y.cpu()
