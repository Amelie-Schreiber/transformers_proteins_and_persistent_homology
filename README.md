# transformers_proteins_and_persistent_homology
Transformers, Protein Sequences, and Persistent Homology

<details>
  <summary>Click to expand</summary>
- [Clustering protein sequences](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_clustering.ipynb) using ESM-2, persistent homology, and various clustering algorithms.
</details>

- [Computing simplex trees](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/simplex_trees_esm2.ipynb) Simplex Trees from Context Vectors of ESM-2. This can be used for sequential and structural motif discovery. 
- [Visualizing Context Vectors of ESM2](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_visualization.ipynb) this can be used for discovering structural and sequential motifs.
- [Substructure and Motif Discovery](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_motifs_esm_2.ipynb) using a persistent homology informed DBSCAN of the context vectors.
- [Computing Fréchet Means](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/frechet_mean_ph_diagrams_esm_2.ipynb) of persistence diagrams. See also [this notebook](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/frechet_mean_esm_2_v2.ipynb). This can be used for anomalous protein sequence detection compared to a baseline Fréchet mean. It is perhaps best to cluster a collection of proteins using the first notebook, then compute the Fréchet mean of some cluster of interest to use as a baseline.

Note, all of the above can be done at the level of attention probability distributions using the Jensen-Shannon distance metric, or at the level of context vectors or hidden states using the Euclidean distance metric. 

While low dimensional persistent homology features $(H_0, H_1, H_2, H_3)$ seems to be more than enough for most applications, it is almost impossible to investigate much higher dimensional homology groups $H_i$ with $i > 3$ due to the computational difficulty. Thus, this is a good opportunity to test out methods in [this article](https://arxiv.org/pdf/2209.12887.pdf), [this article](https://quantum-journal.org/papers/q-2022-12-07-873/pdf/), and [this article](https://dspace.mit.edu/bitstream/handle/1721.1/101739/Lloyd-2016-Quantum%20Algorithms.pdf;sequence=1), where there is a proven exponential speedup in computation in this case. For very long protein sequences (anything with thousands of amino acids) would also be a very obvious reason to apply this. 




