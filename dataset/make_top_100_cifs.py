import json, os
from tqdm import tqdm
from matgfn.reticular import PormakeStructureBuilder

if __name__ == "__main__":
    with open("dataset.json", "r") as f:
        all_items_dict = json.load(f)

    items = [[x[0].split(":")[0],x[0].split(":")[1].split(","),x[1]] for x in list(all_items_dict.items())]
    items.sort(key = lambda x: x[2], reverse = True)

    items = items[0:100]

    print(items)

    index = 0
    builder_dict = {}

    os.makedirs("top_100_cifs_unrelaxed", exist_ok=True)

    for topology, sequence, reward in tqdm(items):
        include_edges = any(["E" in x for x in items[0][1]])

        if include_edges:
            builder_dict_key = topology + "E"
        else:
            builder_dict_key = topology

        if  builder_dict_key not in builder_dict:
            builder_dict[builder_dict_key] = PormakeStructureBuilder(topology_string = topology, 
                                                                     include_edges=include_edges,
                                                                     block_rmsd_cutoff= None)
        
        builder = builder_dict.get(builder_dict_key)

        cif = builder.make_cif(sequence)

        filename = f"top_100_cifs/{index:03}-{topology}-{int(reward)}.cif"
        
        with open(filename, "w") as f:
            f.write(cif)

        index = index + 1