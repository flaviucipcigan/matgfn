from openTSNE import TSNE
import pickle
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import colormaps
import logging

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    # Load ARC-MOF AMDs
    arcmof_amds = np.load("features/arcmof_amds/arcmof_amds.npy")

    # Load MatGFN AMDs
    matgfn_sequences_and_amds = []
    for filename in glob("features/matgfn_amds/amds_*.pkl"):
        with open(filename, "rb") as f:
            matgfn_sequences_and_amds = matgfn_sequences_and_amds + pickle.load(f)

    matgfn_amds_all = np.array([x[3] for x in matgfn_sequences_and_amds])
    matgfn_rewards_all = np.array([x[2] for x in matgfn_sequences_and_amds])

    matgfn_amds = matgfn_amds_all[matgfn_rewards_all != 0]
    matgfn_rewards = matgfn_rewards_all[matgfn_rewards_all != 0]

    logging.info(f"arcmof_amds.shape={arcmof_amds.shape}")
    logging.info(f"matgfn_amds.shape={matgfn_amds.shape}")
    logging.info(f"matgfn_rewards.shape={matgfn_rewards.shape}")

    all_amds = np.vstack([arcmof_amds, matgfn_amds])
    logging.info(f"all_amds.shape={all_amds.shape}")

    tsne = TSNE(n_components=2, perplexity=30, metric="chebyshev", verbose=50, n_jobs=-1,neighbors="pynndescent",n_iter=1000)
    embedding = tsne.fit(all_amds)

    arcmof_embeddings = embedding[0:arcmof_amds.shape[0]]
    matgfn_embeddings = embedding[arcmof_amds.shape[0]:]

    np.save(file = "features/embeddings/amd_embeddings_arcmof.npy", arr = np.array(arcmof_embeddings))
    np.save(file = "features/embeddings/amd_embeddings_matgfn.npy", arr = np.array(matgfn_embeddings))
    np.save(file = "features/embeddings/rewards_matgfn.npy", arr = matgfn_rewards)

    with open("features/embeddings/opentsne_embeddings.pkl", "wb") as f:
        pickle.dump(obj = embedding, file = f)