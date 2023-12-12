import amd
from joblib import Parallel, delayed
from glob import glob
import pickle

# Download ARC-MOF CIFs from here:
# https://zenodo.org/records/7600474/files/all_structures_1.tar.gz?download=1
# https://zenodo.org/records/7600474/files/all_structures_2.tar.gz?download=1

def compute_amd(cif_path):
    try:
        cif_reader = amd.CifReader(cif_path, reader="pymatgen")
        
        amds = [amd.AMD(crystal, 100) for crystal in cif_reader]
        assert len(amds) == 1

        return amds[0]
    except Exception as e:
        print(f"Error at {cif_path}: {e}")
        return None

if __name__ == "__main__":
    cif_filenames = glob("arcmof_cifs/*cif")
    arcmof_amds = Parallel(n_jobs=-1,verbose=True)(delayed(compute_amd)(cif_filename)
                                                    for cif_filename in cif_filenames)
    
    output = [[x[0], x[1]] for x in zip(cif_filenames, arcmof_amds) if x[1] is not None]
    
    with open("features/arcmof_amds/arcmof_amds_with_cifs.pkl", "wb") as f:
        pickle.dump(obj = output, file = f)
