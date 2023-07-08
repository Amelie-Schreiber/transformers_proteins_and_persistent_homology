# transformers_proteins_and_persistent_homology
Transformers, Protein Sequences, and Persistent Homology

- [Clustering protein sequences](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_clustering.ipynb) using ESM-2, persistent homology, and various clustering algorithms.
- [Computing simplex trees](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/simplex_trees_esm2.ipynb) Simplex Trees from Context Vectors of ESM-2. This can be used for sequential and structural motif discovery. 
- [Visualizing Context Vectors of ESM2](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_visualization.ipynb) this can be used for discovering structural and sequential motifs.
- [Substructure and Motif Discovery](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_motifs_esm_2.ipynb) using a persistent homology informed DBSCAN of the context vectors.
- [Computing Fréchet Means](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/frechet_mean_ph_diagrams_esm_2.ipynb) of persistence diagrams. See also [this notebook](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/frechet_mean_esm_2_v2.ipynb). This can be used for anomalous protein sequence detection compared to a baseline Fréchet mean. 

Note, all of the above can be done at the level of attention probability distributions using the Jensen-Shannon distance metric, or at the level of context vectors or hidden states using the Euclidean distance metric. 
