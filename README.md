---

# ESMTopic: Transformers, Proteins, and Persistent Homology
## Topic Modeling for Proteins, Model Interpretability, and Topological Inductive Biases
---

<img src="https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/KaiB.png" alt="Substructure of KaiB" width="400"/>

## Introduction

This project is focused on developing a topic modeling tool for proteins, inspired by the [BERTopic](https://maartengr.github.io/BERTopic/index.html) model used in natural language processing. Our objective is to leverage hierarchical clustering algorithms like *persistent homology*, and the *ESM-2* and *ESMFold* transformer models, to create a *topic modeling method for proteins*. Here, a "topic" can be considered a phenotype, tertiary structure, or functional modeling attribute that reflects important characteristics such as secondary structure arrangements, binding sites, or other features in the language of life. We plan to eventually incorporate other protein language models into this approach, and other clustering methods. For most applications using persistent homology, DBSCAN, or HDBSCAN is more than enough though. 

## Understanding Protein Structures

As part of our project, we will incorporate persistent homology, a concept from topological data analysis, into our hierarchical clustering algorithms to extract substructures of amino acids deemed important by the model. Sometimes these are secondary structure motifs ($\alpha \beta \alpha$ for example), topologically important regions (like handles), binding sites, fold-switching regions, mutations cites that cause conformational or topological changes, specific subsets of amino acids, etc. This mathematical framework will help us identify and understand the high-dimensional structural patterns in the latent representation of protein sequences given by *ESM-2*.

## Model Interpretability

Our project is not only about modeling proteins but also about interpreting these models. We aim to understand how attention mechanisms in transformer models learn to recognize and represent biologically significant information. We're inspired by the study "[BERTology Meets Biology: Interpreting Attention in Protein Language Models](https://arxiv.org/abs/2006.15222)", which demonstrated that transformer models can specialize in detecting various protein features, from individual amino acids to complex structures like binding sites. We approach the problems of understanding attention, context vectors, and hidden states in this project, using the tools of persistent homology, multi-scale DBSCAN, and HDBSCAN, to analyze what regions of proteins ESM-2's heads and layers are paying attention to, how the model's heads and layers are clustering those regions at different scales, and how the model clusters collections of proteins. This interpretability aspect of the project can aid in the design of specialized heads or layers, or in knowledge distillation procedures. 

## Topological Inductive Biases

This study seeks to incorporate a structured inductive bias into our protein modeling process. This bias, founded on the concept of invariant persistent homology, applies to motifs and other protein substructures and is introduced through a topological loss function or regularization term, denoted as $\lambda \mathcal{L}_{topo}$. The model generates internal representations of proteins—comprising attention probability distributions, context vectors, or hidden states. Each of these representations is associated with a persistent homology persistence diagram, enabling us to compare these diagrams using the Wasserstein distance metric. A key objective of our approach is to maintain the persistent homology of protein substructures, like specialized binding sites or secondary structural motifs (for instance, $\alpha \beta \alpha$), invariant across different contexts. This implies that even when a motif appears in different proteins, its persistent homology remains consistent within the model's internal representations.

Our approach applies persistent homology not to the protein's structural data, but to the model's internal representations when processing a motif or other substructure of interest. The aim is to prompt the model to consistently recognize and represent the invariant features of the motif, irrespective of its context (the protein it is located in). While this introduces a topological constraint on the relative positioning of context vectors, it doesn't limit the values they can assume. Importantly, this method can extend to any substructure where the preservation of topological features might play a crucial role. By maintaining topological invariance, our model gains the potential to better capture and learn the essence of protein motifs, enhancing its overall performance in protein sequence analysis and prediction tasks. 

## Topological Prior for Diffusion Models

A generative diffusion model, such as a denoising diffusion probabilistic model (DDPM), generates novel data by evolving an initial random noise through a series of small diffusion steps. In this context, we're using it to generate novel protein sequences. Here's how you could incorporate a topological inductive bias using ESMFold or ESM-2 as the backbone transformer. In this setting, each step of the diffusion process involves a denoising step where the model predicts the protein sequence at the next step given the current noisy version. The backbone transformer, ESMFold or ESM-2, can be used to make this prediction. To incorporate the topological inductive bias, we can modify the loss function used in training the denoising model. For a given protein motif, we can compute the persistent homology of the model's internal representation (attention probabilities, context vectors, or hidden states) and represent it as a persistence diagram.

We can also compute the persistent homology of the motif in the original protein sequences, which we'll refer to as the "ground truth" persistence diagram. We can then measure the difference between these two persistence diagrams using the Wasserstein distance. By adding this Wasserstein distance as a term in the loss function, we are effectively encouraging the model to generate protein sequences that maintain the topological invariance of the motifs, regardless of the context in which they appear. During the generative phase, we start with a random noise and gradually denoise it through a series of steps, with the final result being a generated protein sequence. At each denoising step, we can apply the transformer model, compute the persistent homology of the current sequence, and adjust the sequence to reduce the Wasserstein distance between the computed persistence diagram and the ground truth. In this way, the DDPM not only generates plausible protein sequences but also ensures that the topological features of certain motifs remain invariant, making it more likely to generate functional protein sequences. The inductive bias introduced in this way can lead to models that generate novel protein sequences that retain important functional properties, leading to more effective protein design and the discovery of novel proteins. 

## Notebooks

---

We have developed several notebooks to illustrate the methodologies and techniques used in our project:

## Protein Topic Modeling
These notebooks use the last hidden states of the model. 

1. [Extracting and Visualizing Substructures](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_substructures_esm2.ipynb): This is similar to extracting keyphrases, collocations, multiword expressions, and idioms from text. It includes a way of visualizing the substructure in the $3D$-fold predicted by ESMFold.
2. [Extracting Simplex Trees](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/simplex_trees_esm2_layer.ipynb): This is similar to constituency or dependency parse trees. It shows the importance the model assigns to each simplex. Searching for simplices with low filtration values of a particular length allows us to identify substructures of residues  of a particular length, like candidate motifs, candidate binding sites, or sites of mutations that cause topological changes in conformational states.
3. [Clustering Proteins](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/clustering_esm2_pipeline.ipynb): This is similar to text corpus and document clustering based on the themes present in the texts. It often captures tertiary structural themes, conformational states, and topology and groups proteins based on this. Below we see the clusters given by ESM-2 for $15$ protein sequences generated by GPT-4. 

<img src="https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/cluster7.png" alt="Clusters of Proteins Given by ESM-2 and Persistent Homology Informed DBSCAN" width="400"/>

**Table**: Persistent Homology Informed DBSCAN Clustering of Protein Sequences

| **Cluster** | **Protein Sequences** |
|-------------|-----------------------|
| 0 | MAHMTQIPLSSVKRPLQVRGIVFLGICTQKGTVYGNASWVRDQARH,<br> MKHVTQIPKSSVRRPLQFRGICFLGTCQKGTVYGKASWVHDQARHA,<br> MNHITQVPLSSAKRPLQVRGICFLGICTKNGTVYGKACWVRDQARH,<br> MGGHNGWQILVKGKWTTMDFLRNAVIDQKLRRARRELKLMKAFESLK,<br> MGGHNGWQILVKGKWTTMDFLRNAVIDQKLRRARRELKLMKAFESLKN,<br> MGGHNGWQILVKGKWTTMDFLRNAVIDQKLRRARRELKLMKAFESLKNN,<br> MAQSNISDAMVQLTPAGRSLMLLVQHGSQVAAGVTFQDNQRFPGGRD,<br> MAQSNISDAMVQLTPAGRSLMLLVQHGSQVAAGVTFQDNQRFPGGRDF,<br> MAQSNISDAMVQLTPAGRSLMLLVQHGSQVAAGVTFQDNQRFPGGRDFF |
| 1 | MKLITILGLLALATLVQSTGCVTVNAAHCGVTTGQTVCAGVAKCRAE |
| 2 | MKLITILGALALATLVQSTGCVNVNAAHCVTTGQTVCAGVAKCRAET,<br> MKLITILGALALATLVQSTGCVNVNAAHCVTAGQTVCAGVAKCRAETS |
| 3 | MGSSHHHHHHSSGLVPRGSHMENITVVKFNGTQTFEVHPNVSVGQAGV,<br> MGSSHHHHHHSSGLVPRGSHMENITVVKFNGTQTFEVHPNVSVGQAGVR,<br> MGSSHHHHHHSSGLVPRGSHMENITVVKFNGTQTFEVHPNVSVGQAGVRR |




## Model Interpretability

### Context Vectors of Individual Attention Heads
These notebooks focus on using context vectors for individual attention heads. This gives us a way to see what individual attention heads have learned to focus on and model well at the level of context vectors, computed prior to layer normalization and the MLP. 

1. [Clustering Protein Sequences using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_clustering.ipynb)
2. [Computing Simplex Trees using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/simplex_trees_esm2.ipynb)
3. [Visualizing Context Vectors of ESM2](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/esm_2_visualization.ipynb)
4. [Substructure and Motif Discovery using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/extracting_motifs_esm_2.ipynb)
5. [Computing Fréchet Means using context vectors](https://github.com/Amelie-Schreiber/transformers_proteins_and_persistent_homology/blob/main/frechet_mean_ph_diagrams_esm_2.ipynb)

#### Using Attention Probability Distributions
These notebooks focus on using attention probability distributions  for individual attention heads. These are given by the softmax rows of the attention matrix, using the Jensen-Shannon distance metric to compute the distance matrix for persistent homology. This gives us a way to see what individual attention heads have learned to focus on and model well. 

Notebooks coming soon...


## Topological Inductive Biases from Topological Loss Functions

We can add a term to the loss function which includes this inductive bias into the model. 

1. For a given motif, collect a set of internal representations corresponding to its occurrence in different proteins.
2. Compute the persistent homology for each of these to generate a set of persistence diagrams.
3. Compute a "reference" persistence diagram for the motif, either by averaging the above diagrams using the Fréchet mean diagram or by some other method.
4. For each occurrence of the motif, compute the Wasserstein distance between its persistence diagram and the reference diagram.
5. Average these distances to compute a single loss value for the motif.
6. Repeat this process for each motif of interest and sum the results to produce the final topological loss term.

This term can be added to the overall model loss, alongside any other terms used to train the model (such as cross-entropy loss for predicting the next amino acid in a protein sequence). By minimizing this combined loss, the model will be encouraged to develop internal representations that consistently capture the invariant features of each motif, as encoded by their persistent homology.

To further include invariant persistent homology as an inductive bias:

1. Custom Layers: We could design custom layers that directly use the concept of persistent homology. These layers could be trained to output similar embeddings when the same motif is fed into the model, regardless of the protein in which the motif is embedded. Remember, this does not constrain the values of the hidden states, only their relative distances do one another (and thus their persistent homology). 
2. Multi-task Learning: We could design a multi-task learning setup where one task is to predict some property of the protein and the other is to minimize the variance of the persistent homology of a motif's internal representations across different contexts.
3. Contrastive Learning: We could use a contrastive learning approach, where pairs of internal representations of the same motif in different contexts are pushed to be similar (small Wasserstein distance), while representations of different motifs are pushed to be different (large Wasserstein distance).


## Denoising Diffusion Probabilistic Models for Generating Proteins

Incorporating a topological inductive bias into a generative diffusion model, specifically a Denoising Diffusion Probabilistic Model (DDPM), enables the generation of novel protein sequences that retain critical topological invariances. We plan to use ESMFold as the backbone transformer. Given a protein motif, let $\mathcal{D}_P$ denote the "ground truth" persistence diagram, which we obtain from the persistent homology of the motif within the original protein sequences. Let $\mathcal{D}_M$ denote the persistence diagram derived from the persistent homology of the model's internal representation of the motif (encompassing attention probabilities, context vectors, or hidden states). We use the Wasserstein distance, denoted as $W(\cdot, \cdot)$, to quantify the difference between these diagrams. 

Formally, the Wasserstein distance is defined as:

$$
d^{\mathcal{W}}_{p, q}(\mathcal{D}_P, \mathcal{D}M) = \left( \inf{\gamma\in \Gamma(\mathcal{D}_P, \mathcal{D}M)} \int{\mathcal{D}_P \times \mathcal{D}_M} ||x - y||^q_p d\gamma(x, y) \right)^{1/q}
$$

where $\Gamma(\mathcal{D}_P, \mathcal{D}_M)$ is the set of all joint measures on $\mathcal{D}_P \times \mathcal{D}_M$ with marginals respectively $\mathcal{D}_P$ and $\mathcal{D}_M$. The $||x - y||^q_p$ denotes the $L_p$-norm raised to the power $q$. 

The loss function used to train the model, $L_{\text{total}}$, can be augmented by a term $\lambda W(\mathcal{D}_P, \mathcal{D}_M)$, with $\lambda > 0$ being a hyperparameter. Thus, we have:

$$
L_{\text{total}} = L_{\text{task}} + \lambda d^{\mathcal{W}}_{p, q}(\mathcal{D}_P, \mathcal{D}_M)
$$

where $L_{\text{task}}$ is the base loss function, such as the negative log-likelihood. By minimizing $L_{\text{total}}$, the DDPM learns to generate protein sequences where the topological features of key motifs remain invariant, thus enhancing the biological plausibility of generated sequences. It is worth noting that this approach increases the complexity of model training, hence necessitating efficient algorithms for persistence diagram computation and optimal transport problem solving.


## Future Directions

Our project is an ongoing effort, and we aim to further expand our understanding of proteins using advanced computational tools. Future directions include:

1. Investigating topological changes in protein structures under mild sequence mutations.
2. Understanding the impact of protein sequence changes on conformational states.
3. Developing topological inductive biases for fine-tuning ESMFold for generative protein diffusion models (denoising diffusion probabilistic model or DDPM) and downstream tasks like detection of point mutations that cause topological changes.
4. Developing topological loss functions as regularizers or to include a topological inductive bias such as invariance of persistent homology for motifs or important substructures. 
5. Exploring the potential of quantum computing for speeding up computations in high-dimensional homology groups, for very long protein sequences with thousands of residues or more, or for large collections of proteins (see [this paper](https://quantum-journal.org/papers/q-2022-12-07-873/), [this paper](https://arxiv.org/abs/2202.12965), and [this paper](https://www.nature.com/articles/ncomms10138)). While quantum advantage is proven in the above articles, it is not clear to what extent this is needed. 

By integrating insights from transformer models, persistent homology, and other computational tools, we hope to deepen our understanding of the complex language of life encoded in proteins.






