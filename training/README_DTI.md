# DTI Model Training System

This directory contains an advanced training system for Drug-Target Interaction (DTI) models with comprehensive features for research and production use.

## Features

### Training Features
- **Multiple Optimizers**: Adam, AdamW, SGD, RMSprop with full parameter control
- **Learning Rate Scheduling**: Step, Exponential, Cosine, Plateau, Cosine with Warm Restarts
- **Loss Functions**: BCE, BCE with Logits, Focal Loss
- **Regularization**: Weight decay, gradient clipping, dropout
- **Early Stopping**: Configurable patience and monitoring metrics
- **Checkpointing**: Automatic saving of best models and training state
- **Resume Training**: Continue from any checkpoint

### Evaluation Features
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, MCC
- **Visualizations**: Confusion matrix, ROC curves, PR curves, probability distributions
- **Threshold Analysis**: Find optimal decision thresholds
- **Detailed Reports**: Classification reports and confusion matrices

### Experiment Tracking
- **Weights & Biases Integration**: Automatic logging of metrics, hyperparameters, and artifacts
- **Local Logging**: File-based logging with timestamps
- **Configuration Management**: YAML-based configuration system

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_training.txt
```

### 2. Basic Training
```bash
# Train with default configuration
python training/train_dti.py --config configs/train.yaml

# Quick training for testing
python training/train_dti.py --config configs/train_quick.yaml --mode train

# Production training
python training/train_dti.py --config configs/train_production.yaml --mode both
```

### 3. Using the Convenient Script
```bash
# Make script executable (first time only)
chmod +x scripts/run_training.sh

# Train with default settings
./scripts/run_training.sh train

# Train with custom name and GPU
./scripts/run_training.sh train --name my_experiment --gpu 0

# Train with Weights & Biases logging
./scripts/run_training.sh train --wandb --name wandb_experiment

# Evaluate a trained model
./scripts/run_training.sh evaluate --model outputs/my_experiment/best_model.pt --detailed

# Run experiments with different fusion methods
./scripts/run_training.sh experiment --fusion concat sum mul cross
```

## Configuration System

The training system uses YAML configuration files for maximum flexibility. Key configuration sections:

### Model Configuration
```yaml
model:
  fusion: "cross"          # Fusion method: concat, sum, mul, cross
  proj_dim: 256           # Projection dimension
  hidden_dim: 512         # Hidden layer dimension
  dropout: 0.1            # Dropout rate
```

### Optimizer Configuration
```yaml
optimizer:
  name: "adamw"           # Optimizer: adam, adamw, sgd, rmsprop
  lr: 0.001              # Learning rate
  weight_decay: 0.01     # L2 regularization
  betas: [0.9, 0.999]    # Adam beta parameters
```

### Scheduler Configuration
```yaml
scheduler:
  name: "cosine"          # Scheduler type
  T_max: 50              # Cosine annealing period
  eta_min: 0             # Minimum learning rate
```

### Early Stopping
```yaml
early_stopping:
  enabled: true           # Enable early stopping
  patience: 15           # Epochs to wait for improvement
```

## File Structure

```
training/
├── train_dti.py          # Main training script
├── evaluate_dti.py       # Standalone evaluation script
├── trainer.py            # Advanced trainer class
└── evaluator.py          # Comprehensive evaluator class

configs/
├── train.yaml           # Default training configuration
├── train_quick.yaml     # Quick training for testing
└── train_production.yaml # Production training settings

scripts/
└── run_training.sh      # Convenient training script
```

## Usage Examples

### 1. Training a Model

**Basic Training:**
```bash
python training/train_dti.py --config configs/train.yaml --mode train
```

**With Custom Settings:**
```bash
python training/train_dti.py \
    --config configs/train_production.yaml \
    --experiment_name production_model_v1 \
    --gpu 0 \
    --mode both
```

### 2. Evaluating a Model

**Basic Evaluation:**
```bash
python training/evaluate_dti.py \
    --model_path outputs/my_experiment/best_model.pt
```

**Detailed Evaluation:**
```bash
python training/evaluate_dti.py \
    --model_path outputs/my_experiment/best_model.pt \
    --detailed \
    --threshold_analysis \
    --output_dir results/evaluation
```

### 3. Hyperparameter Experiments

**Different Fusion Methods:**
```bash
for fusion in concat sum mul cross; do
    python training/train_dti.py \
        --config configs/train.yaml \
        --experiment_name fusion_${fusion} \
        --mode train
done
```

**Learning Rate Search:**
```bash
for lr in 0.0001 0.001 0.01; do
    # Create temporary config with modified learning rate
    sed "s/lr: .*/lr: $lr/" configs/train.yaml > /tmp/config_lr_${lr}.yaml
    
    python training/train_dti.py \
        --config /tmp/config_lr_${lr}.yaml \
        --experiment_name lr_search_${lr} \
        --mode train
done
```

## Advanced Features

### 1. Resume Training
```bash
python training/train_dti.py \
    --config configs/train.yaml \
    --resume_from_checkpoint outputs/my_experiment/latest_checkpoint.pt
```

### 2. Weights & Biases Integration
```yaml
# In config file
use_wandb: true
wandb_project: "dti-research"
experiment_name: "cross_fusion_experiment"
```

### 3. Custom Loss Functions
```yaml
# Focal loss for imbalanced data
criterion: "focal"
focal_alpha: 1
focal_gamma: 2
```

### 4. Advanced Scheduling
```yaml
# Cosine annealing with warm restarts
scheduler:
  name: "cosine_warm_restart"
  T_0: 10
  T_mult: 2
  eta_min: 0
```

## Output Structure

After training, the following structure is created:

```
outputs/
└── experiment_name/
    ├── best_model.pt              # Best model checkpoint
    ├── latest_checkpoint.pt       # Latest checkpoint
    ├── evaluation_results.json    # Detailed evaluation metrics
    ├── confusion_matrix.png       # Confusion matrix plot
    ├── roc_curve.png             # ROC curve plot
    ├── precision_recall_curve.png # PR curve plot
    └── probability_distribution.png # Probability histogram
```

## Monitoring Training

### 1. Log Files
Training logs are saved to `logs/experiment_name/training.log` with detailed information about:
- Training progress and metrics
- Model architecture and parameters
- Configuration settings
- Checkpoint saving events

### 2. Weights & Biases
If enabled, metrics are automatically logged to W&B including:
- Training/validation loss and accuracy
- Learning rate schedules
- Model architecture
- Hyperparameters
- System metrics

### 3. Checkpoints
- **Best Model**: Saved when validation metric improves
- **Latest Checkpoint**: Saved every N epochs (configurable)
- **Resume State**: Complete training state for resuming

## Tips for Best Results

1. **Start with Quick Config**: Use `train_quick.yaml` for initial experiments
2. **Use Production Config**: Switch to `train_production.yaml` for final models
3. **Monitor Overfitting**: Watch validation metrics and use early stopping
4. **Experiment with Fusion**: Try different fusion methods for your data
5. **Tune Learning Rate**: Use learning rate scheduling for better convergence
6. **Use Focal Loss**: For imbalanced datasets, try focal loss
7. **Enable W&B**: Use Weights & Biases for experiment tracking
8. **Save Checkpoints**: Regular checkpointing prevents loss of progress

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model dimensions
2. **Slow Training**: Increase batch size or use multiple GPUs
3. **Poor Convergence**: Try different learning rates or optimizers
4. **Overfitting**: Increase dropout, weight decay, or use early stopping
5. **Loss Not Decreasing**: 
   - Check if you're using the right loss function (BCEWithLogitsLoss for logits)
   - Verify model outputs raw logits, not probabilities
   - Try smaller learning rates (1e-4 or 1e-5)
   - Check data preprocessing and label encoding
6. **Validation Metrics Stuck**: 
   - Ensure model is in eval mode during validation
   - Check for data leakage between train/val sets
   - Verify batch normalization is working correctly

### Fixed Training Issues

**Major fixes applied in this version:**

1. **Fixed Double Sigmoid Problem**: Model now outputs raw logits, loss functions expect logits
2. **Proper Loss Functions**: Use BCEWithLogitsLoss or fixed focal loss with logits
3. **Better Initialization**: Xavier initialization for linear layers
4. **Batch Normalization**: Added to prevent internal covariate shift
5. **Conservative Learning Rates**: Default to 1e-4 for stability
6. **Proper Evaluation**: Apply sigmoid only during evaluation/prediction

### Debug Mode

For debugging training issues, use the debug configuration:
```bash
./scripts/run_training.sh train --config configs/train_debug.yaml --name debug_run
```

Add debug logging to any script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Configuration Recommendations

1. **For Debugging**: Use `configs/train_debug.yaml`
2. **For Quick Experiments**: Use `configs/train_fixed.yaml` 
3. **For Production**: Use `configs/train_production.yaml` (after debugging)

### Monitoring Training Health

Watch for these signs of healthy training:
- Training loss should decrease consistently
- Validation loss should follow training loss initially
- Learning rate should be adjusted based on plateau detection
- Accuracy should improve over epochs
- No NaN or exploding gradients

Signs of problematic training:
- Loss stays constant for many epochs
- Loss oscillates wildly
- Validation metrics don't change
- Very high loss values (>10 for binary classification)

For more details, see the individual script documentation and configuration comments.
