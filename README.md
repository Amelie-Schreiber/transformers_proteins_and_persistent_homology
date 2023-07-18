---

# ESMTopic: Transformers, Proteins, and Persistent Homology
## Topic Modeling for Proteins and Model Interpretability Project
---

<img src="https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/KaiB.png" alt="Substructure of KaiB" width="400"/>

## Introduction

This project is focused on developing a topic modeling tool for proteins, inspired by the [BERTopic](https://maartengr.github.io/BERTopic/index.html) model used in natural language processing. Our objective is to leverage hierarchical clustering algorithms like *persistent homology* and the *ESM-2* and *ESMFold* transformer models to create a *topic modeling method for proteins*. Here, a "topic" can be considered a phenotype, tertiary structure, or functional modeling attribute that reflects important characteristics such as secondary structure arrangements, binding sites, or other features in the language of life. We plan to eventually incorporate other protein language models into this approach, and other clustering methods. For most applications using persistent homology, DBSCAN, or HDBSCAN is more than enough though. 

## Understanding Protein Structures

As part of our project, we will incorporate persistent homology, a concept from topological data analysis, into our hierarchical clustering algorithms to extract substructures of amino acids deemed important by the model. Sometimes these are secondary structure motifs ($\alpha \beta \alpha$ for example), topologically important regions (like handles), binding sites, fold-switching regions, mutations cites that cause conformational or topological changes, specific subsets of amino acids, etc. This mathematical framework will help us identify and understand the high-dimensional structural patterns in the latent representation of protein sequences given by *ESM-2*.

## Model Interpretability

Our project is not only about modeling proteins but also about interpreting these models. We aim to understand how attention mechanisms in transformer models learn to recognize and represent biologically significant information. We're inspired by the study "[BERTology Meets Biology: Interpreting Attention in Protein Language Models](https://arxiv.org/abs/2006.15222)", which demonstrated that transformer models can specialize in detecting various protein features, from individual amino acids to complex structures like binding sites.

## Notebooks

We have developed several Jupyter notebooks to illustrate the methodologies and techniques used in our project:

### Protein Topic Modeling
These notebooks use the last hidden states of the model. 

1. [Extracting and Visualizing Substructures](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_substructures_esm2.ipynb): This is similar to extracting keyphrases, collocations, multiword expressions, and idioms from text. It includes a way of visualizing the substructure in the $3D$-fold predicted by ESMFold.
2. [Extracting Simplex Trees](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/simplex_trees_esm2_layer.ipynb): This is similar to constituency or dependency parse trees. It shows the importance the model assigns to each simplex. Searching for simplices with low filtration values of a particular length allows us to identify substructures of residues  of a particular length, like candidate motifs, candidate binding sites, or sites of mutations that cause topological changes in conformational states.
3. [Clustering Proteins](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/clustering_esm2_pipeline.ipynb): This is similar to text corpus and document clustering based on the themes present in the texts. It often captures tertiary structural themes, conformational states, and topology and groups proteins based on this.

<img src="https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/cluster7.png" alt="Cluster 7 of ESM-2" width="400"/>


### Model Interpretability

#### Context Vectors of Individual Attention Heads
These notebooks focus on using context vectors for individual attention heads. This gives us a way to see what individual attention heads have learned to focus on and model well at the level of context vectors, computed prior to layer normalization and the MLP. 

1. [Clustering Protein Sequences using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_clustering.ipynb)
2. [Computing Simplex Trees using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/simplex_trees_esm2.ipynb)
3. [Visualizing Context Vectors of ESM2](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_visualization.ipynb)
4. [Substructure and Motif Discovery using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_motifs_esm_2.ipynb)
5. [Computing Fréchet Means using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/frechet_mean_ph_diagrams_esm_2.ipynb)

#### Using Attention Probability Distributions
These notebooks focus on using attention probability distributions  for individual attention heads. These are given by the softmax rows of the attention matrix, using the Jensen-Shannon distance metric to compute the distance matrix for persistent homology. This gives us a way to see what individual attention heads have learned to focus on and model well. 

Notebooks coming soon...


## Future Directions

Our project is an ongoing effort, and we aim to further expand our understanding of proteins using advanced computational tools. Future directions include:

1. Investigating topological changes in protein structures under mild sequence mutations.
2. Understanding the impact of protein sequence changes on conformational states.
3. Developing topological inductive biases for fine-tuning ESMFold for generative protein diffusion models (denoising diffusion probabilistic model or DDPM) and downstream tasks like detection of point mutations that cause topological changes.
4. Developing topological loss functions as regularizers or to include a topological inductive bias such as invariance of persistent homology for motifs or important substructures. 
5. Exploring the potential of quantum computing for speeding up computations in high-dimensional homology groups, for very long protein sequences with thousands of residues or more, or for large collections of proteins (see [this paper](https://quantum-journal.org/papers/q-2022-12-07-873/), [this paper](https://arxiv.org/abs/2202.12965), and [this paper](https://www.nature.com/articles/ncomms10138)). While quantum advantage is proven in the above articles, it is not clear to what extent this is needed. 

By integrating insights from transformer models, persistent homology, and other computational tools, we hope to deepen our understanding of the complex language of life encoded in proteins.






