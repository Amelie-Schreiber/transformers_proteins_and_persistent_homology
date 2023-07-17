# Transformers, Proteins, and Persistent Homology

## ProTopic: Topic Modeling for Proteins

This project is the equivalent of [BERTopic](https://maartengr.github.io/BERTopic/index.html) for proteins. Specifically, we plan to add persistent homology to the [hierarchical clustering algorithms](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html) and use [ESM-2/ESMFold](https://huggingface.co/docs/transformers/model_doc/esm) in place of BERT. We also plan to eventually add other models to the list of protein language models. Topic modeling for proteins can be thought of as a kind of phenotype, tertiary structure, functional modeling that takes into account things like arrangements of secondary structure, binding sites, and the syntax and semantics of the language of life. 

## Interpretability

This is also an interpretability project. It was shown already in [BERTology Meets Biology: Interpreting Attention in Protein Language Models](https://arxiv.org/abs/2006.15222) that individual attention heads can specialize to detetct binding sites, secondary structures, and various other biologically significant information. Some shallow layer heads were shown to specialize in individual amino acids. As the layers get deep, higher order structures are learned. We are searching for similar trends and behaviors in ESM-2 and ESMFold. We have already notices that some heads seem to cluster proteins by tertiary structure and topology, others specialize in classifying proteins into functional families. We hope to find other heads that specialize in categorizing proteins in other ways. For example, proteins can be classified based on function into categories such as enzymes, transport proteins, structural proteins, signaling proteins, etc. They can also be classified based on their structure into categories such as globular proteins, fibrous proteins, and membrane proteins. Finding heads that specialize in any of these, or several of these, for some subgroup of proteins would be helpful in training models that specialize in one or several of these and would allow for knowledge distillation. 



## Transformers, Protein Sequences, and Persistent Homology

### Protein Topic Modeling Notebooks

-[Clustering Proteins](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/clustering_esm2_pipeline.ipynb)

-[Extracting and Visualizing Substructures](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_substructures_esm2.ipynb)


### Interpretability
Here are a few of the notebooks for model interpretability:

- [Clustering protein sequences](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_clustering.ipynb) using ESM-2, persistent homology, and various clustering algorithms.
- [Computing simplex trees](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/simplex_trees_esm2.ipynb) Simplex Trees from Context Vectors of ESM-2. This can be used for sequential and structural motif discovery. 
- [Visualizing Context Vectors of ESM2](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_visualization.ipynb) this can be used for discovering structural and sequential motifs.
- [Substructure and Motif Discovery](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_motifs_esm_2.ipynb) using a persistent homology informed DBSCAN of the context vectors.
- [Computing Fréchet Means](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/frechet_mean_ph_diagrams_esm_2.ipynb) of persistence diagrams. See also [this notebook](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/frechet_mean_esm_2_v2.ipynb). This can be used for anomalous protein sequence detection compared to a baseline Fréchet mean. It is perhaps best to cluster a collection of proteins using the first notebook, then compute the Fréchet mean of some cluster of interest to use as a baseline.

Note, all of the above can be done at the level of attention probability distributions using the Jensen-Shannon distance metric, or at the level of context vectors or hidden states using the Euclidean distance metric. 

While low dimensional persistent homology features $(H_0, H_1, H_2, H_3)$ seems to be more than enough for most applications, it is almost impossible to investigate much higher dimensional homology groups $H_i$ with $i > 3$ due to the computational difficulty. Thus, this is a good opportunity to test out methods in [this article](https://arxiv.org/pdf/2209.12887.pdf), [this article](https://quantum-journal.org/papers/q-2022-12-07-873/pdf/), and [this article](https://dspace.mit.edu/bitstream/handle/1721.1/101739/Lloyd-2016-Quantum%20Algorithms.pdf;sequence=1), where there is a proven exponential speedup in computation in this case. Very long protein sequences (anything with thousands of amino acids) would also be a very obvious reason to apply this. This is possibly a good opportunity for quantum computing to prove itself. 

This project has evolved into attempting to transfer the ideas present in [Prediction of multiple conformational states by combining sequence clustering with AlphaFold2](https://www.biorxiv.org/content/10.1101/2022.10.17.512570v1) to ESM-2/ESMFold using persistent homology of the internal representations of ESM-2/ESMFold in place of using a multi-scale DBSCAN of the protein sequences directly (using Levenshtein edit distance). The applications to multi-conformational "fold-switching" proteins should transfer over well and provide a way for choosing an optimal $\epsilon$ for DBSCAN. This project is also now attempting to understand what topological changes occur under mild mutations of the protein sequences. It was found in the above work that an edit of three amino acids in KaiB changed the topology of one of the conformational states of KaiB, so understanding when and why such topological changes occur, and which edits to the sequence are required to create a conformational change is now clearly within the scope of the project. With the biological topic modeling methods in place, we should have a good set of tools with which to work on this problem. 


