#!/bin/bash

# Orpheus Arabic Voice Cloning Training Script
# This script provides an easy way to start training

set -e  # Exit on error

echo "=================================================="
echo "Orpheus Arabic Voice Cloning Training"
echo "=================================================="
echo ""

# Default values
CONFIG_PATH="training/config-orpheus.yaml"
PAIRED_DATASET_PATH="dataset_cache/paired_dataset/"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --dataset)
            PAIRED_DATASET_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH       Path to config YAML file (default: training/config-orpheus.yaml)"
            echo "  --dataset PATH      Path to paired dataset (default: dataset_cache/paired_dataset/)"
            echo "  --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --config my_config.yaml --dataset my_dataset/"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Config file not found at $CONFIG_PATH"
    echo "Please create a config file or specify a different path with --config"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$PAIRED_DATASET_PATH" ]; then
    echo "❌ Error: Paired dataset not found at $PAIRED_DATASET_PATH"
    echo ""
    echo "Please create a paired dataset first using:"
    echo "  python training/create_pairs.py --segments_path=... --output_path=$PAIRED_DATASET_PATH"
    echo ""
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import unsloth; import snac; import phonemizer; import wandb" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Missing dependencies"
    echo ""
    echo "Please install required packages:"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "✅ All dependencies found"
echo ""

# Display configuration
echo "Configuration:"
echo "  Config file: $CONFIG_PATH"
echo "  Dataset: $PAIRED_DATASET_PATH"
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
else
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Confirm before starting
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "=================================================="
echo "Starting Training..."
echo "=================================================="
echo ""

# Run training
python training/finetune-orpheus.py --config_fpath="$CONFIG_PATH"

echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="

