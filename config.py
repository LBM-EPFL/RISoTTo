# import sys
from datetime import datetime


config_data = {
    'dataset_filepath': "../../datasets/pdb_bprna_structures_nuc_v6.h5",
    'train_selection_filepath': "datasets/grnade_das_train_set.txt",
    'validation_selection_filepath': "datasets/grnade_das_validation_set.txt",
    'max_ba': 1,
    'max_size': 1024*12,
    'min_num_res': 10,
    'r_noise': 0,
    'virt_cb': False,
    'partial': True,
}

config_model = {
    "em": {'N0': 34, 'N1': 64},
    "sum": sum([
        [{'Ns': 64, 'Nh': 2, 'Nk':3, 'nn': 8}]*5,
        [{'Ns': 64, 'Nh': 2, 'Nk':3, 'nn': 16}]*5,
        [{'Ns': 64, 'Nh': 2, 'Nk':3, 'nn': 32}]*5,
        [{'Ns': 64, 'Nh': 2, 'Nk':3, 'nn': 64}]*5,
    ], []),
    "spl": {'N0': 64, 'N1': 64, 'Nh': 4},
    "dm": {'N0': 64, 'N1': 64, 'N2': 4},
}

# define run name tag
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'rna_v45_benchmark_grnade_das_bpRNA_cg_d5_ss64_partial'+tag, # P, C4', O2', N9/N1
    'output_dir': 'save',
    'reload': True,
    'device': 'cuda',
    'num_epochs': 100,
    'log_step': 512,
    'eval_step': 512*8,
    'eval_size': 512,
    'learning_rate': 1e-4,
    'pos_weight_factor': 0.9,
    'comment': "",
}
