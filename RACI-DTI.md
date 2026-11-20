# RACI-DTI

**Role-Aware Cross-Modal Interaction Learning for Robust Drug–Target Interaction Prediction**

RACI-DTI predicts interactions between drugs and protein targets. It encodes four molecular views separately and assigns them to two interaction pathways according to their roles. The functional–semantic pathway compares drug motifs with protein sequence context. The topology–geometry pathway compares drug topology, aligned to motif positions, with protein residue structure. Each pathway models information flow in both directions with its own attention parameters.

## Model architecture

### Molecular views

| View | Representation |
| --- | --- |
| Drug atom topology | A Transformer encodes a SMILES-derived molecular graph with 61-dimensional atom descriptors and molecular adjacency. When a valid conformer is available, radial basis expansions of interatomic distances provide attention biases. |
| Drug functional motifs | A motif-level Transformer uses functional substructure tokens and features aggregated from their member atoms using the atom–motif assignment matrix. The motif representation includes a learnable global token. |
| Protein sequence semantics | Pretrained ESM-2 produces a 1,280-dimensional contextual embedding for each residue. |
| Protein residue structure | A structural graph attention encoder uses residue distances and contacts derived from AlphaFold-predicted structures. |

Separate projection layers map the four views to a shared 256-dimensional space. The semantic and structural representations remain distinct at the input to the interaction pathways.

### Role-specific interactions

| Pathway | Input token sets | Interaction |
| --- | --- | --- |
| Functional–semantic | Drug functional motifs and ESM-2 residue embeddings | Two bidirectional cross-attention blocks using query–key compatibility, with no geometry-derived attention bias. |
| Topology–geometry | Motif-aligned drug topology and protein residue-structure embeddings | One bidirectional cross-attention block with geometry priors specific to the context supplying the keys and values. |

For topology–geometry interaction, the atom–motif assignment matrix maps projected atom features to motif positions. These features are added to the corresponding motif tokens, and the global motif token is retained. Drug-side geometry priors are computed from atom distances and propagated to motif positions through atom–motif membership. Protein-side priors are derived directly from residue distances. The drug global token receives zero geometry bias.

In an interaction written as **X ← Y**, X supplies the queries and Y supplies the keys and values. The output retains the token positions of X. Thus **Drug ← Protein** updates drug tokens from protein context, and **Protein ← Drug** updates residue tokens from drug context. In the topology–geometry pathway, the first direction uses the protein prior and the reverse direction uses the drug prior.

```text
Drug motifs + protein sequence
    └── Functional–semantic bidirectional cross-attention
          ├── Drug-query output
          └── Protein-query output

Motif-aligned drug topology + protein residue structure
    └── Topology–geometry bidirectional cross-attention
          ├── Drug-query output with protein-side key bias
          └── Protein-query output with drug-side key bias

Four outputs → masked readout for each output
    → combine the two pathway summaries separately for each entity
    → add projected pre-interaction summaries as residuals
    → pair representation → three-layer MLP → interaction probability
```

### Readout and missing structures

Masked mean pooling summarizes each of the four interaction outputs. Separate drug and protein projections combine their respective pathway summaries, with residual contributions from the pre-interaction features. The MLP receives the concatenation of the resulting drug and protein vectors, their element-wise product, and their absolute difference. The model is trained with binary cross-entropy.

If the topology–geometry pathway is unavailable for a pair, the model skips its structural interaction and geometry-prior computations. Zero-valued placeholders are used at the **readout stage** for the unavailable structural summaries. The functional–semantic pathway still operates on motifs and protein sequences. For an unavailable drug conformer, the underlying atom encoder can still use atom features and adjacency without its distance bias.

## Datasets and preprocessing

- **BindingDB.** Pairs with measured IC50 below 100 nM are labeled positive and those above 10,000 nM are labeled negative. Other records are excluded under this labeling rule. The preprocessing in the manuscript also excludes compounds represented by only one label class.
- **BioSNAP.** Known DrugBank associations supply positive pairs. Negative pairs are sampled from pairs without a reported interaction. These sampled pairs have a different evidence basis from measured negative pairs in BindingDB.

Each prepared pairing record needs drug and protein identifiers that resolve to the corresponding features and a binary interaction label. The implementation uses molecular graph and motif features, atom–motif membership, residue-level ESM-2 embeddings, and available structural features such as distance matrices and contacts. Keep features and split files aligned to the same entity identifiers.

## Repository layout

The local implementation used to verify the file layout and command-line arguments contains the following main files:

```text
.
├── dataset/
│   ├── BindingDB/
│   └── BiosNap/
├── drug/
│   ├── features.py
│   ├── model_atom_level.py
│   ├── model_motif_level.py
│   ├── mol_graph.py
│   └── utils.py
├── Protein/
│   ├── extract_protein_features.py
│   ├── protein_feature_extractor.py
│   └── protein_3d_extractor.py
├── dataset_loader.py
├── drug_encoder.py
├── interaction_model.py
├── trainer.py
├── train_dti2_kfold.py
└── map_motif_names.py
```

## Training and evaluation

### Environment and feature files

The training code uses PyTorch, NumPy, pandas, scikit-learn, and tqdm. Feature preparation also uses libraries such as RDKit, SciPy, Transformers, and Biopython. For GPU training, use a PyTorch build compatible with the local CUDA environment. Set the paths to the prepared features and motif vocabulary before running the script. The inspected local copy contains absolute paths for precomputed protein and geometry files and expects the motif vocabulary at `Classification/UnseenDDIs/token_id.json`.

### Five-fold training

From the implementation directory, select a dataset and split protocol:

```bash
python train_dti2_kfold.py --dataset BindingDB --split_type warm_start
python train_dti2_kfold.py --dataset BindingDB --split_type cold_start_drug
python train_dti2_kfold.py --dataset BindingDB --split_type cold_start_protein
```

Use `--dataset BiosNap` for BioSNAP. The script expects fold directories under `dataset/<dataset>/<split_type>/fold_<n>`.

| Setting | Split rule in the manuscript |
| --- | --- |
| S1 · Warm start | Interaction pairs are held out. The evaluated drugs and proteins can be represented in training. |
| S2 · Drug cold start | Test drugs are excluded from the training set. |
| S3 · Protein cold start | Test proteins are excluded from the training set. |

The manuscript specifies five-fold cross-validation and summarizes AUROC, AUPR, accuracy, precision, recall, and F1. A probability threshold of 0.5 is used for the threshold-dependent metrics. The reported optimization uses AdamW, learning-rate warm-up followed by cosine annealing, and early stopping based on validation AUROC. The reported label smoothing factors are 0.05 for BindingDB and 0.01 for BioSNAP.

### Results reported in the manuscript

The table lists the point estimates stated in the manuscript's Results section.

| Dataset | Setting | AUROC | AUPR | F1 |
| --- | --- | ---: | ---: | ---: |
| BindingDB | S1 | 0.960 | 0.946 | 0.886 |
| BindingDB | S2 | 0.799 | 0.728 | 0.690 |
| BindingDB | S3 | 0.622 | 0.553 | 0.596 |
| BioSNAP | S1 | 0.925 | 0.928 | 0.857 |
| BioSNAP | S2 | 0.856 | 0.861 | 0.769 |
| BioSNAP | S3 | 0.850 | 0.861 | 0.761 |

## Ablation and robustness experiments

- **Interaction controls.** Compare the full model with the sequence-only, topology-only, and structure-only interaction variants reported in the manuscript.
- **Molecular views.** Remove individual views to assess the contribution of drug atoms, drug motifs, ESM-2 sequence features, and protein residue structure.
- **Geometry priors.** Remove the protein residue prior or the drug functional-group prior, or compare the two separate priors with a shared-prior variant.
- **Structural missingness at inference.** Vary the rate of structural input removal from 0% to 100% in S2 and S3 while retaining the nonstructural inputs.
- **Composite input perturbations during training.** Apply Gaussian noise, residue masking, and channel dropout to sequence embeddings, and perturb distances and contacts on the structural side. The reported perturbation strengths are 0, 0.3, 0.6, and 0.9, with 0 denoting clean inputs.

## Citation

```bibtex
@misc{LiRACIDTI,
  title  = {{RACI-DTI}: Role-Aware Cross-Modal Interaction Learning for Robust Drug--Target Interaction Prediction},
  author = {Li, Nanxing and Hu, Sai and Zhang, Zizhao and Zhao, Bihai},
  note   = {Manuscript}
}
```
