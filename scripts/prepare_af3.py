#!/usr/bin/env python3
"""Prepare AlphaFold3 JSON inputs per PDB by combining predicted RNA chain FASTAs
and protein chain sequences extracted from PDB files.

Usage:
	python scripts/prepare_af3.py --pdb_dir paper/raw/native_pdbs
	optional args: --pred_dir predictions/test/per_chain --out_dir risotto_af_json

Behavior / assumptions:
- Reads RNA FASTA files from <base>/<pred_dir> (relative paths are resolved against the working directory).
- Considers FASTA files named like 1CSL_A-predicted.fasta. Ignores files where basename contains more than one '-' character.
- Extracts pdb id (e.g. 1CSL) and chain id (e.g. A) from the part before the first '-'.
- Reads the PDB file at <pdb_dir>/{PDBID}.pdb and extracts protein chain sequences from ATOM lines.
- Produces one JSON per PDB with all protein chain entries and all RNA chain entries for that PDB.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple


# Prefer gemmi for PDB parsing when available (faster and handles MODELs/chains cleanly)
try:
	import gemmi  # type: ignore
	_HAS_GEMMI = True
except Exception:
	_HAS_GEMMI = False


AA3TO1 = {
	'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
	'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
	'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
	'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
	# common modified/resolved names - map to X (fallback) or nearest
	'MSE': 'M',
}


def read_fasta(filepath: str) -> str:
	seq_lines = []
	with open(filepath, 'r') as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith('>'):
				continue
			seq_lines.append(line)
	return ''.join(seq_lines)


def parse_pdb_for_protein_sequences(pdb_path: str) -> Dict[str, str]:
	"""Parse PDB ATOM records and return dict chain -> amino-acid sequence (one-letter).

	Only residues whose three-letter codes appear in AA3TO1 are included. Residue order
	preserved by residue number + insertion code.
	"""
	if not os.path.exists(pdb_path):
		return {}

	# Use gemmi if available — it gives easy access to models and chains
	if _HAS_GEMMI:
		try:
			# read_pdb(...)[0] returns the first model directly and respects max_line_length
			model = gemmi.read_pdb(pdb_path, max_line_length=80)[0]
			result: Dict[str, str] = {}
			chains_iter = getattr(model, 'chains', list(model))
			for chain in chains_iter:
				chain_id = getattr(chain, 'name', None) or '_'
				letters: List[str] = []
				for resid in chain:
					resname = getattr(resid, 'name', str(resid)).upper()
					if resname in AA3TO1:
						letters.append(AA3TO1.get(resname, 'X'))
				seq = ''.join(letters)
				if seq:
					result[chain_id] = seq
			return result
		except Exception:
			# fall back to text parsing
			pass

	# Fallback: text-based parsing (previous behavior). Parse only first MODEL if present.
	chains: Dict[str, OrderedDict[Tuple[int, str], str]] = defaultdict(OrderedDict)

	with open(pdb_path, 'r') as f:
		lines = f.readlines()

	has_model = any(ln.startswith('MODEL') for ln in lines)
	in_first_model = not has_model

	for line in lines:
		if has_model:
			if line.startswith('MODEL'):
				if not in_first_model:
					in_first_model = True
					continue
			if line.startswith('ENDMDL') and in_first_model:
				break
			if not in_first_model:
				continue

		if not line.startswith('ATOM'):
			continue

		try:
			resName = line[17:20].strip()
			chainID = line[21].strip()
			resSeq_field = line[22:26].strip()
			resSeq = int(resSeq_field) if resSeq_field else 0
			iCode = line[26].strip() or ''
		except Exception:
			continue

		if resName.upper() in AA3TO1:
			key = (resSeq, iCode)
			if key not in chains[chainID]:
				chains[chainID][key] = resName.upper()

	result: Dict[str, str] = {}
	for chain, residues in chains.items():
		letters = [AA3TO1.get(res, 'X') for res in residues.values()]
		seq = ''.join(letters)
		if seq:
			result[chain or '_'] = seq
	return result


def get_chains_in_first_model(pdb_path: str) -> List[str]:
	"""Return list of chain IDs present in the first MODEL (or whole file if no MODEL records)."""
	if not os.path.exists(pdb_path):
		return []
	# Prefer gemmi
	if _HAS_GEMMI:
		try:
			model = gemmi.read_pdb(pdb_path, max_line_length=80)[0]
			chains_iter = getattr(model, 'chains', list(model))
			return [getattr(c, 'name', None) or '_' for c in chains_iter]
		except Exception:
			pass

	with open(pdb_path, 'r') as f:
		lines = f.readlines()

	has_model = any(ln.startswith('MODEL') for ln in lines)
	in_first_model = not has_model
	chains = []
	seen = set()

	for line in lines:
		if has_model:
			if line.startswith('MODEL'):
				if not in_first_model:
					in_first_model = True
					continue
			if line.startswith('ENDMDL') and in_first_model:
				break
			if not in_first_model:
				continue

		if not (line.startswith('ATOM') or line.startswith('HETATM')):
			continue
		chain_id = line[21].strip() or '_'
		if chain_id not in seen:
			seen.add(chain_id)
			chains.append(chain_id)

	return chains


def build_json_for_pdb(pdb_id: str, template: dict, proteins: List[Tuple[str, str]], rnas: List[Tuple[str, str]]) -> dict:
	# Copy top-level template fields we want to keep; set name to pdb id
	out = {
		'name': pdb_id,
		'modelSeeds': template.get('modelSeeds', [1]),
		'sequences': [],
		'dialect': template.get('dialect', 'alphafold3'),
		'version': template.get('version', 2),
	}

	# Add proteins first
	for chain_id, seq in proteins:
		out['sequences'].append({
			'protein': {
				'id': chain_id,
				'sequence': seq,
				'unpairedMsa': None,
				'pairedMsa': None,
				'templates': []
			}
		})

	# Add RNAs
	for chain_id, seq in rnas:
		out['sequences'].append({
			'rna': {
				'id': chain_id,
				'sequence': seq
				# 'unpairedMsa': ''
			}
		})

	return out


def main():
	here = os.path.dirname(os.path.realpath(__file__))

	ap = argparse.ArgumentParser(description='Prepare AF3 JSON inputs')
	ap.add_argument('--pred_dir', default='predictions/test/per_chain',
					help='Directory (relative to script or absolute) containing RNA per-chain FASTAs')
	ap.add_argument('--pdb_dir', required=True,
					help='Directory containing PDB files named {PDBID}.pdb')
	ap.add_argument('--template', default=os.path.join(here, 'af_input_template.json'),
					help='AF input template JSON')
	ap.add_argument('--out_dir', default='risotto_af_json',
					help='Output directory for generated JSON files')

	args = ap.parse_args()

	# normalize directories
	pred_dir = os.path.abspath(args.pred_dir)
	pdb_dir = os.path.abspath(args.pdb_dir)
	out_dir = os.path.abspath(args.out_dir)

	os.makedirs(out_dir, exist_ok=True)

	# load template
	try:
		with open(args.template, 'r') as f:
			template = json.load(f)
	except Exception:
		template = {'name': 'Protein_RNA_Run', 'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 2}

	# find fasta files
	if not os.path.isdir(pred_dir):
		print(f'Prediction directory not found: {pred_dir}')
		return

	fasta_files = [os.path.join(pred_dir, fn) for fn in os.listdir(pred_dir) if fn.endswith('.fasta')]
	if not fasta_files:
		print(f'No fasta files found in {pred_dir}')
		return

	# collect RNAs by pdb id
	rnas_by_pdb: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

	for path in fasta_files:
		base = os.path.basename(path)
		# ignore files with more than one '-'
		if base.count('-') > 1:
			# skip
			continue
		first_part = base.split('-')[0]
		if '_' not in first_part:
			# unexpected name, skip
			continue
		pdb_id, chain_id = first_part.split('_', 1)
		pdb_id = pdb_id.upper()
		chain_id = chain_id
		seq = read_fasta(path)
		if not seq:
			print(f'Warning: empty sequence in {path}')
			continue
		rnas_by_pdb[pdb_id].append((chain_id, seq))

	# For each pdb, parse the pdb file and write JSON
	written = 0
	for pdb_id, rnas in sorted(rnas_by_pdb.items()):
		# try potential pdb paths
		pdb_path = os.path.join(pdb_dir, f'{pdb_id}.pdb')
		if not os.path.exists(pdb_path):
			pdb_path_lower = os.path.join(pdb_dir, f'{pdb_id.lower()}.pdb')
			pdb_path_upper = os.path.join(pdb_dir, f'{pdb_id.upper()}.pdb')
			if os.path.exists(pdb_path_lower):
				pdb_path = pdb_path_lower
			elif os.path.exists(pdb_path_upper):
				pdb_path = pdb_path_upper
		if not os.path.exists(pdb_path):
			print(f'PDB file not found for {pdb_id}: tried {pdb_path}. Skipping.')
			continue

		proteins_map = parse_pdb_for_protein_sequences(pdb_path)

		# determine which chains are present in the first model and filter RNAs to those chains
		chains_in_model = get_chains_in_first_model(pdb_path)
		if chains_in_model:
			# keep RNAs whose chain id matches one present in model 1
			filtered_rnas = [(c, s) for (c, s) in rnas if (c in chains_in_model) or (c == '_')]
			skipped = [c for (c, s) in rnas if c not in chains_in_model and c != '_']
			if skipped:
				print(f'Warning: skipping RNA chains for {pdb_id} not present in first model: {skipped}')
		else:
			filtered_rnas = rnas

		# Convert proteins_map to list of tuples (chain_id, seq) sorted by chain id
		proteins = sorted(list(proteins_map.items()), key=lambda x: x[0])

		if not proteins:
			print(f'No protein chains found in {pdb_path} for {pdb_id}; will still emit JSON with RNAs only.')

		# build output JSON, using pdb_id as the name
		out_json = build_json_for_pdb(pdb_id, template, proteins, filtered_rnas)

		out_path = os.path.join(out_dir, f'{pdb_id}.json')
		with open(out_path, 'w') as f:
			json.dump(out_json, f, indent=2)
		written += 1

	print(f'Wrote {written} JSON files to {out_dir}')


if __name__ == '__main__':
	main()

