"""
Main training script for Terminal Agent using ART (Agent Reinforcement Trainer).

Implements hybrid training:
1. Offline phase: Warm-start from pre-generated samples
2. Online phase: GRPO with process rewards
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    model_name: str = "RepoMax/Affine-18g-5Dr639TubpvhrbJGSKnCzKakCqHPr9gHze5sSWcgh66AaYGj"
    output_dir: str = "./checkpoints"
    
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    num_generations: int = 16
    temperature: float = 0.7
    max_new_tokens: int = 4096
    
    learning_rate: float = 5e-7
    num_train_epochs: int = 3
    max_steps: int = -1
    
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.6
    
    env_type: str = "swe-synth"
    step_limit: int = 100
    command_timeout: int = 300
    
    offline_data_path: Optional[str] = None
    offline_steps: int = 500
    
    online_steps: int = 2000
    eval_interval: int = 100
    
    log_interval: int = 10
    save_interval: int = 500
    
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    
    api_key: Optional[str] = None
    seed: int = 42


class TerminalAgentTrainer:
    """
    Main trainer class for terminal agent using ART.
    
    Combines:
    - Offline warm-start from pre-generated samples
    - Online GRPO training with process rewards
    - Integration with affinetes environments
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.tokenizer = None
        self.env = None
        self.reward_calculator = None
        
        self.global_step = 0
        self.best_reward = 0.0
        
    def setup(self):
        """Initialize model, environment, and training components."""
        print("=" * 60)
        print("Terminal Agent ART Training")
        print("=" * 60)
        print(f"\nModel: {self.config.model_name}")
        print(f"Device: {self.device}")
        print(f"Environment: {self.config.env_type}")
        
        self._setup_directories()
        self._setup_model()
        self._setup_environment()
        self._setup_reward_calculator()
        self._setup_logging()
        
        print("\n✓ Setup complete")
    
    def _setup_directories(self):
        """Create output directories."""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.run_dir}")
    
    def _setup_model(self):
        """Initialize model and tokenizer."""
        print(f"\nLoading model: {self.config.model_name}")

        try:
            from art import TrainableModel, TrainConfig
            from art.local import LocalBackend

            self.art_model = TrainableModel(
                project="terminal-agent",
                name=f"qwen3-4b-{self.config.env_type}",
                base_model=self.config.model_name,
            )

            art_dir = str(self.run_dir / ".art")
            backend = LocalBackend(in_process=False, path=art_dir)
            asyncio.get_event_loop().run_until_complete(
                self.art_model.register(backend)
            )

            self.art_config = TrainConfig(
                learning_rate=self.config.learning_rate,
            )

            # Store an OpenAI-compatible client for generation
            self._art_client = self.art_model.openai_client()
            self._art_inference_name = self.art_model.get_inference_name()

            print("✓ ART model initialized")

        except ImportError:
            print("ART not available, using TRL fallback...")
            self._setup_trl_model()
    
    def _setup_trl_model(self):
        """Fallback to TRL if ART is not available."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        
        print("Loading with TRL...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✓ TRL model initialized")
    
    def _setup_environment(self):
        """Initialize environment wrapper."""
        print(f"\nInitializing {self.config.env_type} environment...")
        
        from envs import TerminalAgentEnv
        
        self.env = TerminalAgentEnv(
            env_type=self.config.env_type,
            api_key=self.config.api_key or os.getenv("CHUTES_API_KEY"),
        )
        
        print("✓ Environment initialized")
    
    def _setup_reward_calculator(self):
        """Initialize process reward calculator."""
        from rewards import ProcessRewardCalculator, RewardConfig
        
        reward_config = RewardConfig()
        self.reward_calculator = ProcessRewardCalculator(reward_config)
        
        print("✓ Reward calculator initialized")
    
    def _setup_logging(self):
        """Setup logging and wandb."""
        import logging
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.run_dir / "training.log"),
                logging.StreamHandler(),
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        if self.config.wandb_project:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_name or f"terminal-agent-{self.config.env_type}",
                config=self.config.__dict__,
            )
            self.logger.info(f"WandB initialized: {self.config.wandb_project}")
    
    def train_offline(self):
        """Offline training phase from pre-generated samples."""
        if not self.config.offline_data_path:
            self.logger.info("No offline data path specified, skipping offline phase")
            return
        
        print("\n" + "=" * 60)
        print("Phase 1: Offline Training (Warm-start)")
        print("=" * 60)
        
        from data import SampleConverter
        
        converter = SampleConverter()
        batch = converter.load_and_convert(
            self.config.offline_data_path,
            self.config.env_type,
        )
        
        self.logger.info(f"Loaded {len(batch.trajectories)} trajectories")
        self.logger.info(f"Stats: {batch.stats}")
        
        output_path = self.run_dir / "offline_data.jsonl"
        batch.to_jsonl(str(output_path))
        
        if hasattr(self, 'art_model'):
            self.logger.info("Training with ART...")
            self._train_offline_art(batch)
        else:
            self.logger.info("Training with TRL...")
            self._train_offline_trl(batch)
    
    def _train_offline_art(self, batch):
        """Train offline phase with ART."""
        trajectories = [t.to_dict() for t in batch.trajectories]

        for epoch in range(self.config.num_train_epochs):
            self.logger.info(f"Offline epoch {epoch + 1}/{self.config.num_train_epochs}")

            for i in range(0, len(trajectories), self.config.per_device_train_batch_size):
                batch_trajectories = trajectories[i:i + self.config.per_device_train_batch_size]

                asyncio.get_event_loop().run_until_complete(
                    self.art_model.train(batch_trajectories, config=self.art_config)
                )

                self.global_step += 1

                if self.global_step % self.config.log_interval == 0:
                    self.logger.info(f"Offline step {self.global_step}")

                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint()
    
    def _train_offline_trl(self, batch):
        """Train offline phase with TRL."""
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        
        data = []
        for traj in batch.trajectories:
            for i, msg in enumerate(traj.messages):
                if msg["role"] == "user" and i + 1 < len(traj.messages):
                    assistant_msg = traj.messages[i + 1]
                    if assistant_msg["role"] == "assistant":
                        data.append({
                            "prompt": traj.messages[0]["content"] + "\n\n" + msg["content"],
                            "completion": assistant_msg["content"],
                        })
        
        dataset = Dataset.from_list(data)
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=str(self.run_dir),
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                max_steps=self.config.offline_steps,
                logging_steps=self.config.log_interval,
                save_steps=self.config.save_interval,
            ),
        )
        
        trainer.train()
    
    async def train_online(self):
        """Online training phase with GRPO."""
        print("\n" + "=" * 60)
        print("Phase 2: Online Training (GRPO)")
        print("=" * 60)
        
        self.logger.info(f"Starting online training for {self.config.online_steps} steps")
        
        task_ids = self._get_task_ids()
        
        for step in range(self.config.online_steps):
            self.global_step += 1
            
            trajectories = await self._collect_trajectories(task_ids)
            
            if hasattr(self, 'art_model'):
                self._train_step_art(trajectories)
            else:
                self._train_step_trl(trajectories)
            
            if self.global_step % self.config.eval_interval == 0:
                await self._evaluate()
            
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint()
        
        self.logger.info("Online training complete")
    
    def _get_task_ids(self) -> List[int]:
        """Get list of task IDs for training."""
        return list(range(1, 101))
    
    async def _collect_trajectories(
        self,
        task_ids: List[int],
        num_samples: int = 4,
    ) -> List[Dict[str, Any]]:
        """Collect trajectories from environment."""
        import random
        
        trajectories = []
        
        for _ in range(num_samples):
            task_id = random.choice(task_ids)
            
            try:
                observation, info = await self.env.reset(
                    task_id=task_id,
                    step_limit=self.config.step_limit,
                )
                
                trajectory = {
                    "messages": [{"role": "system", "content": self.env.get_system_prompt()}],
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "info": info,
                }
                
                trajectory["messages"].append({"role": "user", "content": observation})
                
                done = False
                while not done:
                    action = await self._generate_action(trajectory["messages"])
                    
                    trajectory["actions"].append(action)
                    trajectory["messages"].append({"role": "assistant", "content": action})
                    
                    result = await self.env.step(action, info.get("episode_id"))
                    
                    step_reward = self.reward_calculator.compute_step_reward(
                        result.observation,
                        action,
                        result.info,
                        result.done,
                    )
                    
                    trajectory["observations"].append(result.observation)
                    trajectory["rewards"].append(step_reward)
                    trajectory["messages"].append({"role": "user", "content": result.observation})
                    
                    done = result.done or result.truncated
                    
                    if result.done:
                        trajectory["final_reward"] = result.reward
                
                trajectories.append(trajectory)
                
                await self.env.stop(info.get("episode_id"))
                
            except Exception as e:
                self.logger.error(f"Error collecting trajectory: {e}")
                continue
        
        return trajectories
    
    async def _generate_action(self, messages: List[Dict[str, str]]) -> str:
        """Generate action from model."""
        if hasattr(self, '_art_client'):
            response = await self._art_client.chat.completions.create(
                model=self._art_inference_name,
                messages=messages,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
            )
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response
    
    def _train_step_art(self, trajectories: List[Dict[str, Any]]):
        """Train step with ART."""
        asyncio.get_event_loop().run_until_complete(
            self.art_model.train(trajectories, config=self.art_config)
        )
        self.logger.info(f"Trained on {len(trajectories)} trajectories")
    
    def _train_step_trl(self, trajectories: List[Dict[str, Any]]):
        """Train step with TRL GRPO."""
        self.logger.info(f"Would train on {len(trajectories)} trajectories (TRL GRPO)")
    
    async def _evaluate(self):
        """Run evaluation on held-out tasks."""
        self.logger.info("Running evaluation...")
        
        eval_task_ids = list(range(101, 111))
        
        total_reward = 0.0
        num_success = 0
        
        for task_id in eval_task_ids:
            try:
                observation, info = await self.env.reset(task_id=task_id)
                
                messages = [
                    {"role": "system", "content": self.env.get_system_prompt()},
                    {"role": "user", "content": observation},
                ]
                
                done = False
                while not done:
                    action = await self._generate_action(messages)
                    messages.append({"role": "assistant", "content": action})
                    
                    result = await self.env.step(action, info.get("episode_id"))
                    messages.append({"role": "user", "content": result.observation})
                    
                    done = result.done or result.truncated
                    
                    if result.done:
                        total_reward += result.reward
                        if result.reward > 0:
                            num_success += 1
                
                await self.env.stop(info.get("episode_id"))
                
            except Exception as e:
                self.logger.error(f"Evaluation error on task {task_id}: {e}")
        
        avg_reward = total_reward / len(eval_task_ids) if eval_task_ids else 0.0
        success_rate = num_success / len(eval_task_ids) if eval_task_ids else 0.0
        
        self.logger.info(f"Evaluation results: avg_reward={avg_reward:.4f}, success_rate={success_rate:.2%}")
        
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self._save_checkpoint(best=True)
    
    def _save_checkpoint(self, best: bool = False):
        """Save model checkpoint."""
        checkpoint_name = "best_checkpoint" if best else f"checkpoint_{self.global_step}"
        checkpoint_dir = self.run_dir / checkpoint_name

        if hasattr(self, 'art_model'):
            # ART manages checkpoints via the backend; export adapter weights
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            step = self.art_model.get_step()
            meta = {"global_step": self.global_step, "art_step": step, "best": best}
            with open(checkpoint_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            self.logger.info(f"ART checkpoint meta saved to {checkpoint_dir} (art_step={step})")
        else:
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            self.logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.env:
            asyncio.run(self.env.cleanup())
        
        if self.config.wandb_project:
            import wandb
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Terminal Agent with ART")
    
    parser.add_argument("--model-name", type=str, default="RepoMax/Affine-18g-5Dr639TubpvhrbJGSKnCzKakCqHPr9gHze5sSWcgh66AaYGj")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--env-type", type=str, default="swe-synth", choices=["swe-synth", "liveweb"])
    
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    
    parser.add_argument("--offline-data", type=str, default=None)
    parser.add_argument("--offline-steps", type=int, default=500)
    parser.add_argument("--online-steps", type=int, default=2000)
    
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        env_type=args.env_type,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        offline_data_path=args.offline_data,
        offline_steps=args.offline_steps,
        online_steps=args.online_steps,
        wandb_project=args.wandb_project,
        seed=args.seed,
    )
    
    trainer = TerminalAgentTrainer(config)
    
    try:
        trainer.setup()
        
        trainer.train_offline()
        
        asyncio.run(trainer.train_online())
        
    finally:
        trainer.cleanup()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
