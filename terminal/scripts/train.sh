#!/bin/bash
# Terminal Agent Training Launch Script
# ======================================

set -e

# Load .env file if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Default values
ENV_TYPE="swe-synth"
MODEL="glowsenior/affine-senior-5GZQjTBnzFnNnKNbVry2hS29CzNa4EkovYAAsdvg3cDA7ssN"
OFFLINE_DATA=""
ONLINE_STEPS=2000
OFFLINE_STEPS=500
BATCH_SIZE=4
LR="5e-7"
WANDB_PROJECT=""
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-type)
            ENV_TYPE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --offline-data)
            OFFLINE_DATA="$2"
            shift 2
            ;;
        --online-steps)
            ONLINE_STEPS="$2"
            shift 2
            ;;
        --offline-steps)
            OFFLINE_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --wandb)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --env-type TYPE      Environment type: swe-synth or liveweb (default: swe-synth)"
            echo "  --model MODEL        Base model (default: glowsenior/affine-senior-5GZQjTBnzFnNnKNbVry2hS29CzNa4EkovYAAsdvg3cDA7ssN)"
            echo "  --offline-data PATH  Path to offline data for warm-start"
            echo "  --online-steps N     Number of online training steps (default: 2000)"
            echo "  --offline-steps N    Number of offline training steps (default: 500)"
            echo "  --batch-size N       Batch size per device (default: 4)"
            echo "  --lr RATE            Learning rate (default: 5e-7)"
            echo "  --wandb PROJECT      WandB project name"
            echo "  --seed N             Random seed (default: 42)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check environment variables
if [ -z "$CHUTES_API_KEY" ]; then
    echo "Error: CHUTES_API_KEY environment variable not set"
    exit 1
fi

# Print configuration
echo "=============================================="
echo "Terminal Agent Training"
echo "=============================================="
echo "Environment:    $ENV_TYPE"
echo "Model:          $MODEL"
echo "Online Steps:   $ONLINE_STEPS"
echo "Offline Steps:  $OFFLINE_STEPS"
echo "Batch Size:     $BATCH_SIZE"
echo "Learning Rate:  $LR"
echo "Seed:           $SEED"
if [ -n "$OFFLINE_DATA" ]; then
    echo "Offline Data:   $OFFLINE_DATA"
fi
if [ -n "$WANDB_PROJECT" ]; then
    echo "WandB Project:  $WANDB_PROJECT"
fi
echo "=============================================="
echo ""

# Build command
CMD="python train.py \
    --env-type $ENV_TYPE \
    --model-name $MODEL \
    --online-steps $ONLINE_STEPS \
    --offline-steps $OFFLINE_STEPS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LR \
    --seed $SEED"

if [ -n "$OFFLINE_DATA" ]; then
    CMD="$CMD --offline-data $OFFLINE_DATA"
fi

if [ -n "$WANDB_PROJECT" ]; then
    CMD="$CMD --wandb-project $WANDB_PROJECT"
fi

# Run training
echo "Starting training..."
echo ""
eval "$CMD"
