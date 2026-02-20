# Terminal Agent: ART Training for Qwen3-4B

Train Qwen3-4B as a multi-turn terminal agent for SWE-SYNTH and LIVEWEB tasks using Agent Reinforcement Trainer (ART) with process rewards.

## Features

- **Hybrid Training**: Offline warm-start + Online GRPO
- **Process Rewards**: Fine-grained reward signals at each step
- **ART Integration**: OpenPipe's Agent Reinforcement Trainer
- **Multi-Environment**: SWE-SYNTH and LIVEWEB support
- **Single GPU Optimized**: Fits on H200 (80GB VRAM)

## Quick Start

### 1. Install Dependencies

```bash
cd terminal_agent
pip install -r requirements.txt
pip install openpipe-art
```

### 2. Set Environment Variables

```bash
export CHUTES_API_KEY="your-api-key"
export TAOSTATS_API_KEY="your-taostats-key"  # For LIVEWEB
```

### 3. Run Training

```bash
# Train on SWE-SYNTH
python train.py --env-type swe-synth --online-steps 2000

# Train with offline warm-start
python train.py --env-type swe-synth --offline-data ./samples.json --online-steps 2000

# Train on LIVEWEB
python train.py --env-type liveweb --online-steps 1000
```

### 4. Evaluate Trained Model

```bash
python scripts/evaluate.py --model-path ./checkpoints/best_checkpoint --env-type swe-synth
```

## Project Structure

```
terminal_agent/
├── train.py                 # Main training script
├── requirements.txt         # Python dependencies
│
├── configs/
│   └── training_config.yaml # Training configuration
│
├── envs/
│   ├── __init__.py
│   ├── art_wrapper.py       # ART-compatible environment wrapper
│   ├── swe_synth_wrapper.py # SWE-SYNTH environment
│   └── liveweb_wrapper.py   # LIVEWEB environment
│
├── rewards/
│   ├── __init__.py
│   └── process_rewards.py   # Process reward calculator
│
├── data/
│   ├── __init__.py
│   └── convert_samples.py   # Offline sample converter
│
├── scripts/
│   └── evaluate.py          # Evaluation harness
│
└── checkpoints/             # Saved models
```

## Training Configuration

Key hyperparameters in `configs/training_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.rank` | 64 | LoRA rank for PEFT |
| `learning_rate` | 5e-7 | Learning rate |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |
| `num_generations` | 16 | GRPO generations per prompt |
| `temperature` | 0.7 | Sampling temperature |
| `max_new_tokens` | 4096 | Max tokens per response |

## Process Rewards

The reward calculator provides fine-grained signals:

| Event | Reward |
|-------|--------|
| Valid bash command | +0.01 |
| Command succeeds | +0.02 |
| Code modified | +0.03 |
| Tests discovered | +0.02 |
| Invalid format | -0.05 |
| Timeout | -0.02 |
| **All tests pass** | +1.0 |
| **Partial pass** | +0.5 × pass_rate |

## Hybrid Training Flow

```
┌─────────────────────────────────────────────────────────┐
│                  Phase 1: Offline                        │
│  Load pre-generated samples → SFT warm-start            │
│  Duration: ~500 steps                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Phase 2: Online                         │
│  Generate trajectories → GRPO update                    │
│  Duration: ~2000 steps                                   │
│  Evaluation every 100 steps                              │
└─────────────────────────────────────────────────────────┘
```

## Evaluation

Run evaluation on held-out tasks:

```bash
python scripts/evaluate.py \
  --model-path ./checkpoints/best_checkpoint \
  --env-type swe-synth \
  --task-ids 101-110 \
  --save-trajectories
```

Compare multiple models:

```python
from scripts.evaluate import compare_models

compare_models(
    model_paths=["./checkpoints/model_v1", "./checkpoints/model_v2"],
    env_type="swe-synth",
    task_ids=list(range(101, 111)),
)
```

## Sample Data Format

Offline samples should follow this format:

```json
{
  "task_id": 1034,
  "score": 0.0,
  "extra": {
    "all_passed": false,
    "all_result": "234/236",
    "bug_types": ["logic-inversion"],
    "conversation": [
      {"role": "system", "content": "You are a helpful assistant..."},
      {"role": "user", "content": "<pr_description>..."},
      {"role": "assistant", "content": "THOUGHT: ...\n```bash\nls -la\n```"},
      ...
    ],
    "problem_statement": "Reset identity button shows...",
    "fix_patch": "diff --git a/..."
  }
}
```

## Hardware Requirements

| Model Size | GPU Memory | Recommended |
|------------|------------|-------------|
| Qwen3-4B | ~20GB | H200/A100 |
| Qwen3-8B | ~40GB | H200/A100 |
| Qwen3-14B | ~60GB | H200 (80GB) |

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size`
- Reduce `vllm_gpu_memory_utilization`
- Enable `gradient_checkpointing`

### Slow Generation
- Increase `vllm_gpu_memory_utilization`
- Use `temperature=0` for faster sampling

### Environment Errors
- Ensure Docker is running
- Check API keys are set
- Verify affinetes is installed

## License

MIT
