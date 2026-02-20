"""
Evaluation harness for trained terminal agents.

Provides evaluation metrics, visualization, and comparison utilities.
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
class EvalConfig:
    """Evaluation configuration."""
    
    model_path: str
    env_type: str = "swe-synth"
    task_ids: List[int] = field(default_factory=lambda: list(range(101, 111)))
    
    step_limit: int = 100
    command_timeout: int = 300
    
    max_new_tokens: int = 4096
    temperature: float = 0.0
    
    output_dir: str = "./eval_results"
    save_trajectories: bool = True
    
    api_key: Optional[str] = None


@dataclass
class EvalResult:
    """Single evaluation result."""
    
    task_id: int
    success: bool
    reward: float
    steps: int
    time_taken: float
    
    test_result: str = ""
    all_passed: bool = False
    
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "reward": self.reward,
            "steps": self.steps,
            "time_taken": self.time_taken,
            "test_result": self.test_result,
            "all_passed": self.all_passed,
            "error": self.error,
        }


class AgentEvaluator:
    """
    Evaluator for trained terminal agents.
    
    Features:
    - Run evaluation on held-out tasks
    - Compare multiple models
    - Generate detailed reports
    - Track performance metrics
    """
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.tokenizer = None
        self.env = None
        
    def setup(self):
        """Initialize model and environment."""
        print("=" * 60)
        print("Terminal Agent Evaluation")
        print("=" * 60)
        print(f"\nModel: {self.config.model_path}")
        print(f"Environment: {self.config.env_type}")
        print(f"Tasks: {len(self.config.task_ids)}")
        
        self._setup_model()
        self._setup_environment()
        self._setup_output_dir()
        
        print("\n✓ Setup complete")
    
    def _setup_model(self):
        """Load model for evaluation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        print(f"\nLoading model from {self.config.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print("✓ Loaded full model")
        except:
            base_model = "Qwen/Qwen3-4B-Instruct"
            print(f"Loading base model {base_model} and adapter...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(self.model, self.config.model_path)
            print("✓ Loaded model with LoRA adapter")
        
        self.model.eval()
    
    def _setup_environment(self):
        """Initialize environment."""
        from envs import TerminalAgentEnv
        
        self.env = TerminalAgentEnv(
            env_type=self.config.env_type,
            api_key=self.config.api_key or os.getenv("CHUTES_API_KEY"),
        )
    
    def _setup_output_dir(self):
        """Create output directory."""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_dir = self.output_dir / f"eval_{timestamp}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
    
    async def evaluate(self) -> List[EvalResult]:
        """Run evaluation on all tasks."""
        results = []
        
        for task_id in self.config.task_ids:
            print(f"\nEvaluating task {task_id}...", end=" ", flush=True)
            
            try:
                result = await self._evaluate_task(task_id)
                results.append(result)
                
                status = "✓" if result.success else "✗"
                print(f"{status} (reward={result.reward:.2f}, steps={result.steps})")
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append(EvalResult(
                    task_id=task_id,
                    success=False,
                    reward=0.0,
                    steps=0,
                    time_taken=0.0,
                    error=str(e),
                ))
        
        return results
    
    async def _evaluate_task(self, task_id: int) -> EvalResult:
        """Evaluate a single task."""
        start_time = time.time()
        
        observation, info = await self.env.reset(
            task_id=task_id,
            step_limit=self.config.step_limit,
            command_timeout=self.config.command_timeout,
        )
        
        trajectory = []
        messages = [
            {"role": "system", "content": self.env.get_system_prompt()},
            {"role": "user", "content": observation},
        ]
        
        step = 0
        done = False
        final_reward = 0.0
        final_info = {}
        
        while not done and step < self.config.step_limit:
            action = self._generate_action(messages)
            
            trajectory.append({
                "step": step,
                "observation": observation[:500],
                "action": action,
            })
            
            messages.append({"role": "assistant", "content": action})
            
            result = await self.env.step(action, info.get("episode_id"))
            
            messages.append({"role": "user", "content": result.observation})
            observation = result.observation
            
            done = result.done or result.truncated
            final_reward = result.reward
            final_info = result.info
            step += 1
        
        await self.env.stop(info.get("episode_id"))
        
        time_taken = time.time() - start_time
        
        return EvalResult(
            task_id=task_id,
            success=final_reward > 0,
            reward=final_reward,
            steps=step,
            time_taken=time_taken,
            test_result=final_info.get("all_result", ""),
            all_passed=final_info.get("all_passed", False),
            trajectory=trajectory if self.config.save_trajectories else [],
        )
    
    def _generate_action(self, messages: List[Dict[str, str]]) -> str:
        """Generate action from model."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response
    
    def save_results(self, results: List[EvalResult]):
        """Save evaluation results."""
        results_file = self.run_output_dir / "results.json"
        
        data = {
            "config": self.config.__dict__,
            "results": [r.to_dict() for r in results],
            "summary": self._compute_summary(results),
        }
        
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        if self.config.save_trajectories:
            trajectories_dir = self.run_output_dir / "trajectories"
            trajectories_dir.mkdir(exist_ok=True)
            
            for result in results:
                if result.trajectory:
                    traj_file = trajectories_dir / f"task_{result.task_id}.json"
                    with open(traj_file, "w") as f:
                        json.dump(result.trajectory, f, indent=2)
    
    def _compute_summary(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Compute summary statistics."""
        successful = [r for r in results if r.success]
        
        return {
            "total_tasks": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results) if results else 0.0,
            "avg_reward": sum(r.reward for r in results) / len(results) if results else 0.0,
            "avg_steps": sum(r.steps for r in results) / len(results) if results else 0.0,
            "avg_time": sum(r.time_taken for r in results) / len(results) if results else 0.0,
            "total_errors": sum(1 for r in results if r.error),
        }
    
    def print_summary(self, results: List[EvalResult]):
        """Print evaluation summary."""
        summary = self._compute_summary(results)
        
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print(f"Total tasks:     {summary['total_tasks']}")
        print(f"Successful:      {summary['successful']}")
        print(f"Success rate:    {summary['success_rate']:.1%}")
        print(f"Average reward:  {summary['avg_reward']:.4f}")
        print(f"Average steps:   {summary['avg_steps']:.1f}")
        print(f"Average time:    {summary['avg_time']:.1f}s")
        print(f"Errors:          {summary['total_errors']}")
        print("=" * 60)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.env:
            asyncio.run(self.env.cleanup())


def compare_models(
    model_paths: List[str],
    env_type: str,
    task_ids: List[int],
    output_dir: str = "./eval_results",
):
    """Compare multiple models."""
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)
    
    all_results = {}
    
    for model_path in model_paths:
        print(f"\nEvaluating {model_path}...")
        
        config = EvalConfig(
            model_path=model_path,
            env_type=env_type,
            task_ids=task_ids,
            output_dir=output_dir,
        )
        
        evaluator = AgentEvaluator(config)
        evaluator.setup()
        
        results = asyncio.run(evaluator.evaluate())
        evaluator.save_results(results)
        evaluator.print_summary(results)
        
        all_results[model_path] = results
        
        evaluator.cleanup()
    
    comparison_file = Path(output_dir) / "comparison.json"
    with open(comparison_file, "w") as f:
        json.dump({
            "models": {
                path: {
                    "summary": evaluator._compute_summary(results)
                    for path, results in all_results.items()
                }
            }
        }, f, indent=2)
    
    print(f"\nComparison saved to {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Terminal Agent")
    
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--env-type", type=str, default="swe-synth")
    parser.add_argument("--task-ids", type=str, default="101-110", help="Range or list (e.g., '101-110' or '1,5,10')")
    parser.add_argument("--step-limit", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-trajectories", action="store_true")
    
    args = parser.parse_args()
    
    if "-" in args.task_ids:
        start, end = map(int, args.task_ids.split("-"))
        task_ids = list(range(start, end + 1))
    else:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    
    config = EvalConfig(
        model_path=args.model_path,
        env_type=args.env_type,
        task_ids=task_ids,
        step_limit=args.step_limit,
        output_dir=args.output_dir,
        temperature=args.temperature,
        save_trajectories=args.save_trajectories,
    )
    
    evaluator = AgentEvaluator(config)
    
    try:
        evaluator.setup()
        results = asyncio.run(evaluator.evaluate())
        evaluator.save_results(results)
        evaluator.print_summary(results)
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
