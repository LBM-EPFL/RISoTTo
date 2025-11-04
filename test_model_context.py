import os
import numpy as np
import torch as pt
import pandas as pd
from tqdm import tqdm
import sys

import src as sp
import runtime as rt


def scoring(p, y, conf):
    # compute confidence probability
    c = pt.from_numpy(conf(p.numpy()))

    # get sequence
    seq_ref = rt.max_pred_to_seq(y)
    seq = rt.max_pred_to_seq(c)

    # assess predictions
    return {
        "size": p.shape[0],
        "recovery_rate": rt.recovery_rate(y, c).numpy().item(),
        "sequence_similarity": rt.sequence_similarity(seq_ref, seq),
        "maximum_recovery_rate": rt.maximum_recovery_rate(y, p).numpy().item(),
        "average_multiplicity": rt.average_multiplicity(p).numpy().item(),
        "average_maximum_confidence": rt.average_maximum_confidence(p).numpy().item(),
        "average_maximum_score": rt.average_maximum_confidence(c).numpy().item(),
    }


def main():
    # parameters
    device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")

    # results parameters
    output_dir = "results/data"

    # model parameters
    save_path = "save"

    # create models
    model = rt.SequenceModel(save_path, "model_ckpt.pt", device=device)

    # create confidence mapping
    conf = rt.ConfidenceMap("results/{}_cdf.csv".format(os.path.basename(save_path)))

    # parameters
    sids_selection_filepath = "datasets/grnade_das_test_set.txt"
    sids_train_filepath = "datasets/grnade_das_train_set.txt"

    # load selected sids
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype('U'))
    sids_sel = np.unique(np.array([s.split('_')[0] for s in sids_sel]))

    # mask partial in training set
    m_tr = np.isin(sids_sel, [s.split('_') for s in np.genfromtxt(sids_train_filepath, dtype=np.dtype('U'))])
    sids_sel = sids_sel[~m_tr]

    # find validation structure ids
    pdbids_sel = np.array([sid.split('_')[0].lower() for sid in sids_sel])

    # get filepaths
    pdb_filepaths = ['/media/bibekar/e500c9b8-094e-4f88-983c-27f30285d37a/bibekar/kuma_backup/carbonara-rna/data/data/all_bioassemblies/pdbs/{}.pdb'.format(pdbid.upper()) for pdbid in pdbids_sel]
    pdb_filepaths = [fp for fp in pdb_filepaths if os.path.exists(fp)]
    pdb_filepaths = [fp for fp in pdb_filepaths if os.path.getsize(fp) < 1e6]
    print("Number of structures:", len(pdb_filepaths))

    # set up dataset
    dataset = rt.StructuresDataset(pdb_filepaths)

    # parameters
    N = len(dataset)

    # sample predictions
    np.random.seed(0)
    for i in tqdm(np.random.choice(len(dataset), 100, replace=False)):
        pdb_filepath = dataset.pdb_filepaths[i]
        # print(pdb_filepath)
        out_filepath = os.path.join(output_dir, os.path.basename(pdb_filepath).split('.')[0]+".csv")
        if os.path.exists(out_filepath):
            continue
        # load structure
        key, structure = dataset[i]
        structure['chain_name'] = np.array([str(cid) for cid in structure['cid']])
        # molecule type and discard unclassified
        subunits = sp.split_by_chain(structure)
        sub_types = rt.subunits_type(subunits)
        subunits = {cid:subunits[cid] for cid in [st[1] for st in sub_types if st[0] != 'na']}
        if len(subunits) == 0:
            continue
        structure = sp.concatenate_chains(subunits)
        # find rna subunits and residue to chain mapping
        cids_rna = [st[1] for st in sub_types if st[0] == 'rna']
        if len(cids_rna) == 0:
            continue
        # print(key, structure['xyz'].shape[0])
        # max size
        if structure['xyz'].shape[0] > model.module.config_data['max_size']:
            continue
        # min size
        if len(np.unique(structure['resid'])) < model.module.config_data['min_num_res']:
            continue
        # min atom size
        if structure['xyz'].shape[0] < 64:
            continue
        # apply model on full structure
        try:
            # print(pdb_filepath)
            _, p, y = model(structure)
        except ValueError:
            print("Structure too small", pdb_filepath)
            continue
        except pt.cuda.OutOfMemoryError:
            print("Out of memory", pdb_filepath)
            continue
        # prediction split by chain
        rcids = np.array([res['chain_name'][0] for res in sp.split_by_residue(structure)])
        pr = {cid:p[rcids==cid] for cid in cids_rna}
        yr = {cid:y[rcids==cid] for cid in cids_rna}
        # apply model with binder subunits known
        pc, yc = {}, {}
        for cid in cids_rna:
            m_known = (structure['chain_name'] != cid)
            _, pi, yi = model(structure, m_known=m_known)
            pi = pi[rcids==cid]
            yi = yi[rcids==cid]
            pi, yi = rt.aa_only(pi, yi)
            pc[cid] = pi
            yc[cid] = yi
        # apply model to subunits alone
        cids_rna = [st[1] for st in sub_types if st[0] == 'rna']
        ps, ys = {}, {}
        for cid in cids_rna:
            subunit = subunits[cid]
            subunit['chain_name'] = np.array([cid]*subunit['xyz'].shape[0])
            if len(np.unique(subunit['resid'])) >= model.module.config_data['min_num_res']:
                if subunit['xyz'].shape[0] > 64:
                    # print(subunit)
                    # print(subunit['chain_name'])
                    try:
                        _, pi, yi = model(subunit)
                    except ValueError:
                        print("Structure too small", pdb_filepath)
                        continue
                    pi, yi = rt.aa_only(pi, yi)
                    ps[cid] = pi
                    ys[cid] = yi
                # else:
                #     continue
        # check that labels match perfectly
        print(pdb_filepath)
        for cid in ys:
            assert pt.sum(pt.abs(yc[cid] - ys[cid])).long().item() == 0
        # contacts
        contacts = sp.extract_all_contacts(subunits, 5.0, device=device)
        # analyse interface recovery
        results = []
        for cid in cids_rna:
            # checks
            if (cid in contacts) and (cid in ys):
                for cidb in list(contacts[cid]):
                    # atom-atom contacts indices
                    ctc_ids = contacts[cid][cidb]['ids'][:,0]
                    # convert to residue-residue contacts indices
                    _, ids = pt.unique(pt.from_numpy(subunits[cid]['resid']), return_inverse=True)
                    ctc_rids = pt.unique(ids[ctc_ids])
                    # binder type
                    btype = [st[0] for st in sub_types if st[1] == cidb][0]
                    # scoring with context
                    results.append({'key': key, 'context_level': 2, 'chain_id_scafold': cid, 'chain_id_binder': cidb, 'num_subunits': len(cids_rna), 'binder_type': btype})
                    results[-1].update(scoring(pc[cid][ctc_rids], yc[cid][ctc_rids], conf))
                    # scoring with context
                    results.append({'key': key, 'context_level': 1, 'chain_id_scafold': cid, 'chain_id_binder': cidb, 'num_subunits': len(cids_rna), 'binder_type': btype})
                    results[-1].update(scoring(pr[cid][ctc_rids], yr[cid][ctc_rids], conf))
                    # scoring without context
                    results.append({'key': key, 'context_level': 0, 'chain_id_scafold': cid, 'chain_id_binder': cidb, 'num_subunits': len(cids_rna), 'binder_type': btype})
                    results[-1].update(scoring(ps[cid][ctc_rids], ys[cid][ctc_rids], conf))
        # pack results
        if len(results) > 0:
            dfi = pd.DataFrame(results)
            dfi.to_csv(out_filepath, index=False)

        # except Exception as e:
            # print(f"Erro in file {pdb_filepath}: {e}")
            # print("ERROR", i, e)
            # sys.exit(1)

if __name__ == '__main__':
    main()
