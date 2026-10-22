import os
import parmed as pmd
from multiprocessing import Pool, cpu_count

# .cif file to .pdb
def convert_cif_to_pdb(cif_file, output_dir):
    try:
        # Generate the output PDB filename with the output directory
        pdb_file = os.path.join(output_dir, os.path.basename(cif_file).replace('.cif', '.pdb'))
        
        # Check if the PDB file already exists
        if os.path.exists(pdb_file):
            print(f"PDB file {pdb_file} already exists. Skipping conversion.")
            return
        
        # Load the CIF file
        structure = pmd.load_file(cif_file)
        
        # Save it as a PDB file
        structure.save(pdb_file)
        print(f"Converted {cif_file} to {pdb_file}")
    
    except Exception as e:
        print(f"Failed to convert {cif_file}: {e}")

# Helper function for parallel processing
def convert_cif_to_pdb_wrapper(args):
    return convert_cif_to_pdb(*args)

# Function to run conversions in parallel with output directory
def parallel_conversion(cif_files, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a pool of workers equal to the number of CPU cores
    with Pool(cpu_count()) as pool:
        pool.map(convert_cif_to_pdb_wrapper, [(cif_file, output_dir) for cif_file in cif_files])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert all .cif files of a directory to .pdb")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # List all .cif files in the input directory
    cif_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.cif')]
    
    # Run the conversion in parallel with the output directory
    parallel_conversion(cif_files, output_dir)
