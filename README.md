# GenDTI: Generative Drug-Target Interaction Modeling

**GenDTI** is a framework for drug discovery that combines drug-target interaction (DTI) prediction with conditional molecular generation using diffusion models.

## 🧬 Overview

GenDTI implements a multi-phase approach:

1. **Phase 1**: Train a DTI model using ESM-2 protein embeddings and GNN molecular encoders
2. **Phase 2**: Train conditional diffusion models for protein-specific molecular generation
3. **Phase 3**: Define a reward function and apply reinforcement learning (Not implemented yet)
## 🏗️ Architecture

### Phase 1: DTI Model
- **Molecular Encoder**: NNConv-based GNN processing 7D continuous atom features and 3D edge features
- **Protein Encoder**: ESM-2 transformer generating protein embeddings
- **Fusion Module**: Cross-interaction block with element-wise operations
- **Prediction Head**: Binary classification for binding prediction

### Phase 2: Diffusion Model (In Development)
- **ConditionalGraphDiffusion**: Protein-conditioned molecular generation
- **Cross-attention blocks**: Integrate protein context into molecular generation
- **Continuous features**: Compatible with DTI model feature space

## 📁 Project Structure

```
GenDTI/
├── configs/                    # Training configurations
│   ├── train.yaml             # DTI model training config
│   └── diffusion_*.yaml       # Diffusion model configs
├── data/                      # Data processing and storage
│   ├── processed/             # Cleaned BindingDB data
│   └── utils/                 # Data processing utilities
├── models/                    # Model implementations
│   ├── dti_model.py          # DTI prediction model with cross-interaction
│   ├── gnn_encoder.py        # NNConv-based molecular encoder
│   ├── esm_encoder.py        # ESM-2 protein encoder
│   └── graph_diffusion.py    # Conditional diffusion model
├── training/                  # Training scripts
│   ├── train_dti.py          # DTI model training
│   └── train_diffusion.py    # Diffusion model training
├── scripts/                   # Data preparation and training scripts
│   ├── run_training.sh       # Comprehensive DTI training script
│   ├── precompute_protein_embeddings.py
│   ├── precompute_drug_embeddings.py
│   └── split_dataset.py
└── inference/                 # Generation and evaluation
```

## 🚀 Quick Start

### 1. Environment Setup

TO BE WRITTEN

### 2. Data Preparation

```bash
# Download and process BindingDB data
python scripts/download_bindingdb.py
python scripts/split_dataset.py

# Precompute embeddings
python scripts/precompute_protein_embeddings.py
python scripts/precompute_drug_embeddings.py
```

### 3. Phase 1: Train DTI Model

**Option A: Using the Training Script (Recommended)**
```bash
# Simple training with defaults
./scripts/run_training.sh train

# Training with custom configuration and experiment name
./scripts/run_training.sh train --config configs/train.yaml --name my_experiment

# Training with Weights & Biases logging
./scripts/run_training.sh train --wandb --name wandb_experiment

# Resume training from checkpoint
./scripts/run_training.sh train --resume outputs/my_experiment/latest_checkpoint.pt

# Run experiments with different fusion methods
./scripts/run_training.sh experiment --fusion concat sum mul cross
```

**Option B: Direct Python Training**
```bash
# Train DTI prediction model directly
python training/train_dti.py --config configs/train.yaml
```

**Evaluation**
```bash
# Evaluate trained model with detailed analysis
./scripts/run_training.sh evaluate --model outputs/my_experiment/best_model.pt --detailed

# Threshold analysis
./scripts/run_training.sh evaluate --model outputs/my_experiment/best_model.pt --threshold-analysis
```

### 4. Phase 2: Train Diffusion Model (In Development)

```bash
# Stage 1: Pretraining on molecular data
python training/train_diffusion.py --config configs/diffusion_pretraining.yaml

# Stage 2: Conditional training with protein context
python training/train_diffusion.py --config configs/diffusion_conditional.yaml
```

## 📊 Features

### Training Script (`scripts/run_training.sh`)
- **Automated Pipeline**: Complete DTI training workflow with requirement checking
- **Experiment Management**: Named experiments with organized output directories
- **Multiple Training Modes**: Default training, custom configs, resume from checkpoint
- **Evaluation Tools**: Detailed model evaluation with threshold analysis and plots
- **Batch Experiments**: Compare different fusion methods automatically
- **GPU Support**: Automatic GPU detection and configuration
- **W&B Integration**: Optional Weights & Biases logging for experiment tracking

### Molecular Representation
- **7D continuous atom features**: Atomic number, degree, charge, aromaticity, hybridization, mass, radicals
- **3D continuous edge features**: Bond type, ring membership, conjugation
- **Graph structure**: NNConv layers with edge-conditioned message passing

### DTI Model Components
- **ESM-2 Integration**: Pre-trained protein language model (facebook/esm2_t6_8M_UR50D)
- **Cross-Interaction Fusion**: Element-wise multiplication with residual connections
- **Deep Classification**: Multi-layer FFN with LayerNorm and dropout

## 🔧 Configuration

### DTI Training (`configs/train.yaml`)
```yaml
model:
  fusion: "cross"        # Cross-interaction fusion
  proj_dim: 256         # Projection dimension  
  hidden_dim: 512       # Hidden layer size
  dropout: 0.1          # Dropout rate

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
```

## 📈 Current Status

### Phase 1: DTI Model ✅
- ESM-2 protein encoder implemented
- NNConv-based GNN molecular encoder
- Cross-interaction fusion mechanism
- **Complete training pipeline with `run_training.sh`**
- **Automated evaluation and experiment management**
- Training pipeline validated

### Phase 2: Diffusion Generation 🚧
- Model architecture designed
- Continuous feature compatibility ensured
- Training pipeline in development

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+
- RDKit 2022.03+
- Transformers 4.20+

### Optional Dependencies
- Weights & Biases (experiment tracking)
- Jupyter (analysis notebooks)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes  
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 📞 Contact

- **Author**: Hugo Walter
- **GitHub**: [@ziraax](https://github.com/ziraax)

---

*A focused approach to drug discovery through AI*

