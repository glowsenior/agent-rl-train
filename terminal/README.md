# Terminal Agent RL Training

Train an LLM as a multi-turn terminal agent using hybrid reinforcement learning (offline SFT warm-start + online GRPO) with fine-grained process rewards.

The agent learns to interact with bash terminals to solve software engineering tasks (SWE-SYNTH) or web automation tasks (LIVEWEB) by issuing commands, reading outputs, and iterating until the task is solved.

## Architecture

```
terminal/
├── train.py                      # Main training orchestrator
├── requirements.txt              # Python dependencies
├── .env                          # API keys (git-ignored)
├── .env.example                  # Template for .env
├── configs/
│   └── training_config.yaml      # Reference hyperparameters
├── envs/
│   ├── __init__.py
│   ├── art_wrapper.py            # Unified env interface + base class
│   ├── swe_synth_wrapper.py      # SWE-SYNTH (code bug fixing)
│   └── liveweb_wrapper.py        # LIVEWEB (web automation)
├── rewards/
│   ├── __init__.py
│   └── process_rewards.py        # Step-level reward calculator
├── data/
│   ├── __init__.py
│   └── convert_samples.py        # Offline trajectory converter
└── scripts/
    ├── train.sh                  # Bash launch wrapper
    ├── evaluate.py               # Evaluation + model comparison
    └── test_setup.py             # Installation verification
```

## How It Works

### Training Pipeline

```
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Offline SFT Warm-Start (~500 steps)             │
│                                                          │
│  Load pre-generated trajectory JSON/JSONL                │
│  → Convert to (prompt, completion) pairs                 │
│  → Supervised fine-tune with LoRA                        │
│  → Build initial model priors                            │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│ Phase 2: Online GRPO (~2000 steps)                       │
│                                                          │
│  For each step:                                          │
│    1. Sample a random task from train set (1-100)        │
│    2. Reset environment → get initial observation        │
│    3. Loop: generate action → step env → collect reward  │
│    4. Compute process rewards for each step              │
│    5. Update model via GRPO                              │
│    6. Evaluate on held-out tasks every 100 steps         │
│    7. Save best checkpoint                               │
└──────────────────────────────────────────────────────────┘
```

### Model Backend

The trainer tries to use [OpenPipe ART](https://github.com/OpenPipe/ART) (Agent Reinforcement Trainer) first. If ART is not installed, it falls back to [TRL](https://github.com/huggingface/trl) with LoRA + SFTTrainer for offline and basic GRPO for online.

### Environment Integration

Environments run as Docker containers managed by `affinetes` (the sibling project at `../affinetes/`). The training loop communicates with them over HTTP:

```
train.py → TerminalAgentEnv → SWESynthEnv → affinetes SynthActor → Docker container
                                                                       ↕ HTTP
                                                                    env.py (Actor)
```

### Process Rewards

Instead of only rewarding at episode end (sparse), the reward calculator assigns small signals at every step:

| Signal | Reward | Description |
|--------|--------|-------------|
| Valid bash/json command | +0.01 | Action has correct format |
| Command succeeds (rc=0) | +0.02 | Returncode zero |
| Code modified | +0.03 | Diff/patch detected in output |
| Tests discovered | +0.02 | Test-related output seen |
| Error context | +0.01 | Error info (learning opportunity) |
| Invalid format | -0.05 | No code block or multiple blocks |
| Timeout | -0.02 | Command timed out |
| No-op action | -0.01 | Empty or useless output |

Step rewards are clamped to [-0.1, +0.1]. Final rewards are added unclamped:

| Final Outcome | Reward |
|---------------|--------|
| All tests pass | **+1.0** |
| Partial pass | +0.5 x pass_rate |
| All tests fail | 0.0 |

---

## Ubuntu VPS Setup Guide

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| GPU | 24GB VRAM (RTX 4090) | H200 140GB / H100 80GB |
| RAM | 32GB | 64GB+ |
| Disk | 100GB free | 200GB+ (for Docker images + checkpoints) |
| Docker | 24.0+ | Latest |
| Python | 3.10+ | 3.11 |
| NVIDIA Driver | 535+ | 550+ |
| CUDA | 12.1+ | 12.4+ |

### Step 1: System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essentials
sudo apt install -y git python3-pip python3-venv curl wget
```

### Step 2: NVIDIA Driver + CUDA

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If not installed:
sudo apt install -y nvidia-driver-550
sudo reboot

# Verify after reboot
nvidia-smi
```

### Step 3: Docker + NVIDIA Container Toolkit

```bash
# Install Docker (skip if already installed)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

> **If the GPG key file already exists**, the command will prompt to overwrite. Type `y` to confirm.

> **If you get `unsupported shim version (3)` error**, your containerd is too old.
> Fix it:
> ```bash
> # Option A: Update containerd
> sudo apt install -y containerd.io
> sudo systemctl restart containerd
> sudo systemctl restart docker
>
> # Option B: Force shim v2 in Docker config
> sudo sed -i 's/io.containerd.runc.v3/io.containerd.runc.v2/g' /etc/docker/daemon.json
> sudo systemctl restart docker
> ```
> Then retry the `docker run --gpus all` command.

### Step 4: Clone and Install

```bash
# Clone the repo
git clone https://github.com/<your-username>/agent-rl-train.git ~/agent-rl-train
cd ~/agent-rl-train

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install affinetes first (required dependency)
cd affinetes
pip install -e .
cd ..

# Install terminal training dependencies
cd terminal
pip install -r requirements.txt
```

> **Note on ART**: If `openpipe-art` fails to install (it may need specific PyTorch/CUDA builds), the trainer falls back to TRL automatically. You can skip it:
> ```bash
> pip install -r requirements.txt 2>&1 | grep -v "openpipe-art" || true
> pip install "openpipe-art[backend]" || echo "ART not available, will use TRL fallback"
> ```

### Step 5: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your actual keys:

```bash
# Required
CHUTES_API_KEY=your-chutes-api-key

# Optional (for LIVEWEB environment)
TAO_API_KEY=your-tao-api-key
```

Then export them:

```bash
set -a; source .env; set +a
```

### Step 6: Build the SWE-SYNTH Docker Image

```bash
cd ~/agent-rl-train/affinetes
afs build environments/SWE-SYNTH --tag swe-synth:latest
```

Or if `afs` isn't in PATH:

```bash
python -m affinetes.cli.main build environments/SWE-SYNTH --tag swe-synth:latest
```

### Step 7: Verify Setup

```bash
cd ~/agent-rl-train/terminal
python scripts/test_setup.py
```

Expected output:

```
============================================================
Terminal Agent Setup Verification
============================================================
Testing imports...
  ✓ envs module
  ✓ rewards module
  ✓ data module

Testing reward calculator...
  Step reward: 0.1000
  Final reward: 0.1000
  ✓ Reward calculator working

Testing sample converter...
  Converted trajectory with 3 messages
  Reward: 1.0
  Task ID: 1234
  ✓ Sample converter working

Testing environment...
  ✓ Environment initialized

Testing model loading...
  PyTorch version: 2.x.x
  CUDA available: True
  GPU: NVIDIA H200
  ✓ Tokenizer loaded

============================================================
✓ All tests passed!
============================================================
```

---

## Running Training

### Online Only (No Pre-Existing Data)

```bash
cd ~/agent-rl-train/terminal

python train.py \
  --env-type swe-synth \
  --online-steps 2000
```

### With Offline Warm-Start

If you have pre-generated trajectory samples (JSON/JSONL):

```bash
python train.py \
  --env-type swe-synth \
  --offline-data ./samples.json \
  --offline-steps 500 \
  --online-steps 2000
```

### Using the Launch Script

```bash
chmod +x scripts/train.sh

./scripts/train.sh \
  --env-type swe-synth \
  --online-steps 2000 \
  --wandb my-project
```

The launch script auto-sources `.env` and passes all args to `train.py`.

### With WandB Logging

```bash
pip install wandb
wandb login

python train.py \
  --env-type swe-synth \
  --online-steps 2000 \
  --wandb-project terminal-agent
```

### All CLI Options

```
python train.py --help

Options:
  --model-name MODEL         Base model HF ID (default: RepoMax/Affine-18g-...)
  --output-dir DIR           Checkpoint directory (default: ./checkpoints)
  --env-type {swe-synth,liveweb}
  --lora-rank N              LoRA rank (default: 64)
  --learning-rate LR         Learning rate (default: 5e-7)
  --batch-size N             Per-device batch size (default: 4)
  --gradient-accumulation N  Gradient accumulation steps (default: 4)
  --offline-data PATH        Path to offline samples JSON/JSONL
  --offline-steps N          Offline training steps (default: 500)
  --online-steps N           Online training steps (default: 2000)
  --max-model-len N          vLLM KV cache max seq len (default: 32768)
  --gpu-memory-utilization F vLLM GPU memory fraction (default: 0.9)
  --wandb-project NAME       Enable WandB logging
  --seed N                   Random seed (default: 42)
```

---

## Evaluation

### Evaluate a Trained Model

```bash
python scripts/evaluate.py \
  --model-path ./checkpoints/<timestamp>/best_checkpoint \
  --env-type swe-synth \
  --task-ids 101-110 \
  --save-trajectories
```

### Compare Multiple Models

```python
from scripts.evaluate import compare_models

compare_models(
    model_paths=[
        "./checkpoints/run1/best_checkpoint",
        "./checkpoints/run2/best_checkpoint",
    ],
    env_type="swe-synth",
    task_ids=list(range(101, 111)),
)
```

### Evaluation Output

```
eval_results/
└── eval_<timestamp>/
    ├── results.json          # Summary + per-task metrics
    └── trajectories/         # Per-task action logs
        ├── task_101.json
        ├── task_102.json
        └── ...
```

---

## Offline Data Format

### SWE-SYNTH Samples

Each sample is a JSON object:

```json
{
  "task_id": 1034,
  "score": 1.0,
  "extra": {
    "all_passed": true,
    "all_result": "236/236",
    "bug_types": ["logic-inversion"],
    "conversation": [
      {"role": "system", "content": "You are a helpful assistant..."},
      {"role": "user", "content": "<pr_description>..."},
      {"role": "assistant", "content": "THOUGHT: ...\n```bash\nls -la\n```"},
      {"role": "user", "content": "<output>...</output>"},
      {"role": "assistant", "content": "THOUGHT: ...\n```bash\nsed -i ...\n```"}
    ],
    "problem_statement": "...",
    "swe_instance_id": "repo-123",
    "fix_patch": "diff --git ..."
  }
}
```

Files can be `.json` (array of samples) or `.jsonl` (one sample per line).

### Converting Samples

```python
from data import SampleConverter

converter = SampleConverter(min_reward_threshold=0.5)  # Only successful trajectories
batch = converter.load_and_convert("./raw_samples.json", env_type="swe-synth")
batch.to_jsonl("./converted.jsonl")

print(f"Converted: {batch.stats['converted_samples']}/{batch.stats['total_samples']}")
```

---

## Training Output

```
checkpoints/
└── <YYYYMMDD_HHMMSS>/
    ├── training.log              # Full training log
    ├── offline_data.jsonl        # Converted offline data (if used)
    ├── .art/                     # ART backend state (if using ART)
    ├── best_checkpoint/          # Best model by eval reward
    │   ├── adapter_model.safetensors
    │   ├── adapter_config.json
    │   └── meta.json
    ├── checkpoint_500/           # Periodic checkpoints
    ├── checkpoint_1000/
    └── events.out.tfevents.*     # TensorBoard logs
```

---

## Hyperparameters

Key hyperparameters (see `configs/training_config.yaml` for full reference):

| Parameter | Default | Notes |
|-----------|---------|-------|
| Model | RepoMax/Affine-18g-... | HuggingFace model ID |
| LoRA rank | 64 | Higher = more capacity, more VRAM |
| LoRA alpha | 64 | Usually same as rank |
| Learning rate | 5e-7 | Low LR for stable RL |
| Batch size | 4 | Per-device |
| Gradient accumulation | 4 | Effective batch = 16 |
| GRPO generations | 16 | Samples per prompt |
| Temperature | 0.7 | Sampling diversity |
| Max tokens | 4096 | Per response |
| vLLM GPU utilization | 0.9 | For inference during rollouts |
| Max model len | 32768 | vLLM KV cache context window |
| Step limit | 100 | Max steps per episode |
| Command timeout | 300s | Per-command timeout |

---

## Hardware Requirements

| Model | Training VRAM | Inference VRAM | Total (approx) |
|-------|--------------|----------------|-----------------|
| Qwen3-4B | ~16GB | ~8GB | ~24GB (RTX 4090) |
| Qwen3-8B | ~32GB | ~16GB | ~48GB (A100 80GB) |
| Qwen3-14B | ~48GB | ~24GB | ~72GB (H100 80GB) |

Single GPU is sufficient. Recommended: H200 (140GB) or H100 (80GB) for comfortable headroom.

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch-size 2 --gradient-accumulation 8

# Or reduce vLLM context / memory
python train.py --max-model-len 8192 --gpu-memory-utilization 0.6
```

> On smaller GPUs (RTX 4090 24GB), you may need `--max-model-len 8192` and `--gpu-memory-utilization 0.6`.
> On H200/H100, the defaults (32768 / 0.9) work fine.

### ImportError: No module named 'affinetes'

Make sure affinetes is installed:

```bash
cd ~/agent-rl-train/affinetes
pip install -e .
```

### Docker Permission Denied

```bash
sudo usermod -aG docker $USER
newgrp docker
# Or: sudo chmod 666 /var/run/docker.sock
```

### Docker: `unsupported shim version (3)`

Common on cloud/rental VPS with older containerd. Fix:

```bash
# Update containerd
sudo apt install -y containerd.io
sudo systemctl restart containerd && sudo systemctl restart docker

# Or force shim v2
sudo sed -i 's/io.containerd.runc.v3/io.containerd.runc.v2/g' /etc/docker/daemon.json
sudo systemctl restart docker
```

### SWE-SYNTH Environment Not Found

The SWE-SYNTH directory uses a dash (`SWE-SYNTH`), not underscore. The wrapper handles this automatically. Ensure the affinetes project structure is intact:

```bash
ls ~/agent-rl-train/affinetes/environments/SWE-SYNTH/env.py
```

### ART Not Available

If `openpipe-art` fails to install, the trainer falls back to TRL automatically. You'll see:

```
ART not available, using TRL fallback...
Loading with TRL...
```

This is fine for initial experiments. ART provides better RL training but TRL works as a baseline.

### Slow Training / Hanging

- Ensure Docker is running: `sudo systemctl start docker`
- Check GPU utilization: `nvidia-smi -l 1`
- Check container logs: `docker ps` then `docker logs <container_id>`
- Verify API key is valid: `echo $CHUTES_API_KEY`

---

## Task Distribution

| Set | Task IDs | Purpose |
|-----|----------|---------|
| Training | 1-100 | Online GRPO rollouts |
| Evaluation | 101-110 | Held-out eval (every 100 steps) |

---

## License

MIT
