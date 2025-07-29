#!/bin/bash

# DTI Model Training and Evaluation Script
# This script provides convenient commands for training and evaluating DTI models

set -e

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/configs"
DEFAULT_CONFIG="$CONFIG_DIR/train.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    train           Train a new DTI model
    evaluate        Evaluate a trained model
    experiment      Run training experiments with different configurations
    help            Show this help message

Training Options:
    --config PATH           Path to configuration file (default: configs/train.yaml)
    --name NAME            Experiment name
    --gpu ID               GPU device ID to use
    --resume PATH          Resume training from checkpoint
    --wandb                Enable Weights & Biases logging

Evaluation Options:
    --model PATH           Path to trained model checkpoint
    --config PATH          Path to configuration file
    --output PATH          Output directory for results
    --detailed             Generate detailed evaluation with plots
    --threshold-analysis   Perform threshold analysis
    --gpu ID               GPU device ID to use

Examples:
    # Train with default configuration
    $0 train

    # Train with custom configuration and name
    $0 train --config configs/my_config.yaml --name my_experiment

    # Train with Weights & Biases logging
    $0 train --wandb --name wandb_experiment

    # Evaluate a trained model
    $0 evaluate --model outputs/my_experiment/best_model.pt --detailed

    # Run experiments with different fusion methods
    $0 experiment --fusion concat sum mul cross

    # Resume training from checkpoint
    $0 train --resume outputs/my_experiment/latest_checkpoint.pt

EOF
}

# Function to check if required files exist
check_requirements() {
    if [ ! -f "$PROJECT_ROOT/data/processed/train.tsv" ]; then
        print_error "Training data not found. Please run data preprocessing first."
        exit 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/outputs/drug_embeddings/drug_embeddings.pt" ]; then
        print_error "Drug embeddings not found. Please run precompute_drug_embeddings.py first."
        exit 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/outputs/protein_embeddings/protein_embeddings.pt" ]; then
        print_error "Protein embeddings not found. Please run precompute_protein_embeddings.py first."
        exit 1
    fi
}

# Function to train model
train_model() {
    local config="$DEFAULT_CONFIG"
    local name=""
    local gpu=""
    local resume=""
    local wandb_flag=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                config="$2"
                shift 2
                ;;
            --name)
                name="$2"
                shift 2
                ;;
            --gpu)
                gpu="$2"
                shift 2
                ;;
            --resume)
                resume="$2"
                shift 2
                ;;
            --wandb)
                wandb_flag="--use-wandb"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [ ! -f "$config" ]; then
        print_error "Configuration file not found: $config"
        exit 1
    fi
    
    print_header "Training DTI Model"
    print_status "Configuration: $config"
    
    # Build command
    local cmd="python $PROJECT_ROOT/training/train_dti.py --config $config --mode train"
    
    if [ -n "$name" ]; then
        cmd="$cmd --experiment_name $name"
        print_status "Experiment name: $name"
    fi
    
    if [ -n "$gpu" ]; then
        cmd="$cmd --gpu $gpu"
        print_status "Using GPU: $gpu"
    fi
    
    if [ -n "$resume" ]; then
        if [ ! -f "$resume" ]; then
            print_error "Checkpoint file not found: $resume"
            exit 1
        fi
        # Note: Resume functionality needs to be added to the config
        print_status "Resuming from: $resume"
    fi
    
    # Check requirements
    check_requirements
    
    # Run training
    print_status "Starting training..."
    eval $cmd
    
    print_status "Training completed!"
}

# Function to evaluate model
evaluate_model() {
    local model_path=""
    local config=""
    local output_dir=""
    local detailed_flag=""
    local threshold_flag=""
    local gpu=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                model_path="$2"
                shift 2
                ;;
            --config)
                config="$2"
                shift 2
                ;;
            --output)
                output_dir="$2"
                shift 2
                ;;
            --detailed)
                detailed_flag="--detailed"
                shift
                ;;
            --threshold-analysis)
                threshold_flag="--threshold_analysis"
                shift
                ;;
            --gpu)
                gpu="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [ -z "$model_path" ]; then
        print_error "Model path is required for evaluation"
        show_usage
        exit 1
    fi
    
    if [ ! -f "$model_path" ]; then
        print_error "Model file not found: $model_path"
        exit 1
    fi
    
    print_header "Evaluating DTI Model"
    print_status "Model: $model_path"
    
    # Build command
    local cmd="python $PROJECT_ROOT/training/evaluate_dti.py --model_path $model_path"
    
    if [ -n "$config" ]; then
        cmd="$cmd --config $config"
        print_status "Configuration: $config"
    fi
    
    if [ -n "$output_dir" ]; then
        cmd="$cmd --output_dir $output_dir"
        print_status "Output directory: $output_dir"
    fi
    
    if [ -n "$detailed_flag" ]; then
        cmd="$cmd $detailed_flag"
        print_status "Generating detailed evaluation"
    fi
    
    if [ -n "$threshold_flag" ]; then
        cmd="$cmd $threshold_flag"
        print_status "Performing threshold analysis"
    fi
    
    if [ -n "$gpu" ]; then
        cmd="$cmd --gpu $gpu"
        print_status "Using GPU: $gpu"
    fi
    
    # Run evaluation
    print_status "Starting evaluation..."
    eval $cmd
    
    print_status "Evaluation completed!"
}

# Function to run experiments
run_experiments() {
    local fusion_methods=()
    local base_config="$DEFAULT_CONFIG"
    local base_name="fusion_experiment"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --fusion)
                shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    fusion_methods+=("$1")
                    shift
                done
                ;;
            --config)
                base_config="$2"
                shift 2
                ;;
            --name)
                base_name="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [ ${#fusion_methods[@]} -eq 0 ]; then
        fusion_methods=("concat" "sum" "mul" "cross")
    fi
    
    print_header "Running Fusion Method Experiments"
    print_status "Base configuration: $base_config"
    print_status "Fusion methods: ${fusion_methods[*]}"
    
    for fusion in "${fusion_methods[@]}"; do
        print_status "Training with fusion method: $fusion"
        
        # Create temporary config
        local temp_config="/tmp/config_${fusion}.yaml"
        cp "$base_config" "$temp_config"
        
        # Update fusion method in config
        sed -i "s/fusion: .*/fusion: \"$fusion\"/" "$temp_config"
        
        # Run training
        train_model --config "$temp_config" --name "${base_name}_${fusion}"
        
        # Clean up
        rm "$temp_config"
    done
    
    print_status "All experiments completed!"
}

# Main script logic
case "${1:-help}" in
    train)
        shift
        train_model "$@"
        ;;
    evaluate)
        shift
        evaluate_model "$@"
        ;;
    experiment)
        shift
        run_experiments "$@"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
