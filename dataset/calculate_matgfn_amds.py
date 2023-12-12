import pormake as pm
import logging, warnings, sys
import amd
from pymatgen.io.cif import CifBlock
from joblib import Parallel, delayed
import pickle
import json

from matgfn.reticular import PormakeStructureBuilder

def calculate_amd_from_item(item, builder_dict):
    pm.log.logger.setLevel(logging.FATAL)
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)

    topology = item[0]
    seq = item[1]
    include_edges = any(["E" in x for x in seq])
    
    if include_edges:
        builder_dict_key = topology + "_E"
    else:
        builder_dict_key = topology
    builder = builder_dict[builder_dict_key]

    cif = builder.make_cif(seq)

    periodic_set = amd.io.periodicset_from_pymatgen_cifblock(CifBlock.from_str(cif))
    return amd.AMD(periodic_set, k=100)

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    
    topology_strings = {'asc','ats','cdl-e','cdz-e','dmg','dnq','eft','ffc','fso','tff','tsg','urj'}

    builder_dict = {}

    for topo_string in topology_strings:
        for include_edges in [True, False]:
            builder_dict_key = f'{topo_string}{"_E" if include_edges else ""}'
            builder_dict[builder_dict_key] = PormakeStructureBuilder(
                                                topology_string = topo_string, 
                                                include_edges = include_edges, 
                                                block_rmsd_cutoff= None)
            
    logging.info("Finished setting up builders")

    with open("dataset.json", "r") as f:
        all_items_dict = json.load(f)

    items = [[x[0].split(":")[0],x[0].split(":")[1].split(","),x[1]] for x in list(all_items_dict.items())]

    logging.info("Loaded dataset.")

    # Parallelise the calculation over multiple jobs
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])
    backend = sys.argv[3]

    logging.info(f"Start at {start_index}, end at {end_index}.")

    outputs = Parallel(n_jobs=-1, backend=backend, verbose=True)(
        delayed(
            lambda item: [item[0], item[1], item[2], calculate_amd_from_item(item, builder_dict=builder_dict)]
        )(item) for item in items[start_index:end_index])

    logging.info("Finished calculating outputs.")

    with open(f"amds_{start_index}_{end_index}.pkl", "wb") as f:
        pickle.dump(outputs, f)

    logging.info("Finished writing.")
    logging.info("Done")