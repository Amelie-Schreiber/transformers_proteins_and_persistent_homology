---

# ESMTopic: Transformers, Proteins, and Persistent Homology

---

<img src="https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/KaiB.png" alt="Substructure of KaiB" width="400"/>

## Introduction

This project is focused on developing a topic modeling tool for proteins, inspired by the [BERTopic](https://maartengr.github.io/BERTopic/index.html) model used in natural language processing. Our objective is to leverage hierarchical clustering algorithms and the ESM-2/ESMFold transformer models to generate a "topic model" for proteins. Here, a "topic" can be considered a phenotype, tertiary structure, or functional modeling attribute that reflects important characteristics such as secondary structure arrangements, binding sites, or other features in the language of life. We plan to eventually incorporate other protein language models into this approach.

## Understanding Protein Structures

As part of our project, we will incorporate persistent homology, a concept from topological data analysis, into our hierarchical clustering algorithms. This mathematical framework will help us identify and understand the high-dimensional structural patterns in protein sequences.

## Model Interpretability

Our project is not only about modeling proteins but also interpreting these models. We aim to understand how attention mechanisms in transformer models learn to recognize and represent biologically significant information. We're inspired by the study "[BERTology Meets Biology: Interpreting Attention in Protein Language Models](https://arxiv.org/abs/2006.15222)", which demonstrated that transformer models can specialize in detecting various protein features, from individual amino acids to complex structures.

## Notebooks

We have developed several Jupyter notebooks to illustrate the methodologies and techniques used in our project:

### Protein Topic Modeling
These notebooks use the last hidden states of the model. 

1. [Clustering Proteins](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/clustering_esm2_pipeline.ipynb)
2. [Extracting and Visualizing Substructures](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_substructures_esm2.ipynb)

### Model Interpretability
These notebooks focus on using context vectors for individual attention heads. 

1. [Clustering Protein Sequences using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_clustering.ipynb)
2. [Computing Simplex Trees using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/simplex_trees_esm2.ipynb)
3. [Visualizing Context Vectors of ESM2](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_visualization.ipynb)
4. [Substructure and Motif Discovery using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_motifs_esm_2.ipynb)
5. [Computing Fréchet Means using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/frechet_mean_ph_diagrams_esm_2.ipynb)

## Future Directions

Our project is an ongoing effort, and we aim to further expand our understanding of proteins using advanced computational tools. Future directions include:

1. Investigating topological changes in protein structures under mild sequence mutations.
2. Understanding the impact of protein sequence changes on conformational states.
3. Developing topological inductive biases for fine-tuning ESMFold for generative protein diffusion models (denoising diffusion probabilistic model or DDPM) and downstream tasks like detection of point mutations that cause topological changes. 
4. Exploring the potential of quantum computing for speeding up computations in high-dimensional homology groups.

By integrating insights from transformer models, persistent homology, and other computational tools, we hope to deepen our understanding of the complex language of life encoded in proteins.






