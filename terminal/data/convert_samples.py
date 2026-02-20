"""
Offline Sample Converter for ART Training.

Converts existing trajectory samples (like the provided JSON format)
to ART-compatible format for warm-start training.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ARTTrajectory:
    """Single trajectory in ART format."""
    
    messages: List[Dict[str, str]]
    reward: float
    task_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": self.messages,
            "reward": self.reward,
            "task_id": self.task_id,
            "metadata": self.metadata,
        }


@dataclass
class ARTBatch:
    """Batch of trajectories for training."""
    
    trajectories: List[ARTTrajectory]
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_jsonl(self, path: str):
        """Save batch as JSONL file."""
        with open(path, "w") as f:
            for traj in self.trajectories:
                f.write(json.dumps(traj.to_dict()) + "\n")


class SampleConverter:
    """
    Converts various sample formats to ART-compatible training data.
    
    Supports:
    - SWE-SYNTH sample format (provided example)
    - LIVEWEB trajectory format
    - Custom JSON formats
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""
    
    def __init__(
        self,
        output_dir: str = "./data/converted",
        min_reward_threshold: float = 0.0,
    ):
        """
        Initialize converter.
        
        Args:
            output_dir: Directory for converted output
            min_reward_threshold: Minimum reward to include sample
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_reward_threshold = min_reward_threshold
    
    def convert_swe_synth_sample(
        self,
        sample: Dict[str, Any],
    ) -> Optional[ARTTrajectory]:
        """
        Convert a single SWE-SYNTH sample to ART trajectory.
        
        Args:
            sample: Sample dict with 'extra' containing 'conversation'
            
        Returns:
            ARTTrajectory or None if conversion fails
        """
        extra = sample.get("extra", {})
        conversation = extra.get("conversation", [])
        
        if not conversation:
            return None
        
        score = sample.get("score", 0.0)
        if score < self.min_reward_threshold:
            return None
        
        messages = []
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT,
        })
        
        task_id = sample.get("task_id", 0)
        problem_statement = extra.get("problem_statement", "")
        swe_instance_id = extra.get("swe_instance_id", "")
        
        for i, turn in enumerate(conversation):
            role = turn.get("role", "")
            content = turn.get("content", "")
            
            if role == "user":
                if i == 1 and problem_statement:
                    content = self._format_problem_statement(
                        problem_statement,
                        swe_instance_id,
                        content,
                    )
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
        
        all_passed = extra.get("all_passed", False)
        all_result = extra.get("all_result", "0/0")
        
        metadata = {
            "swe_instance_id": swe_instance_id,
            "bug_types": extra.get("bug_types", []),
            "all_passed": all_passed,
            "all_result": all_result,
            "model": sample.get("model", ""),
            "model_calls": extra.get("model_calls", 0),
            "latency_ms": sample.get("latency_ms", 0),
            "fix_patch": extra.get("fix_patch", ""),
        }
        
        return ARTTrajectory(
            messages=messages,
            reward=score,
            task_id=task_id,
            metadata=metadata,
        )
    
    def convert_liveweb_sample(
        self,
        sample: Dict[str, Any],
    ) -> Optional[ARTTrajectory]:
        """
        Convert a LIVEWEB sample to ART trajectory.
        
        Args:
            sample: LIVEWEB trajectory sample
            
        Returns:
            ARTTrajectory or None if conversion fails
        """
        trajectory = sample.get("trajectory", [])
        if not trajectory:
            return None
        
        score = sample.get("score", 0.0)
        if score < self.min_reward_threshold:
            return None
        
        messages = [{
            "role": "system",
            "content": self._get_liveweb_system_prompt(),
        }]
        
        for step in trajectory:
            obs = step.get("observation", "")
            action = step.get("action", "")
            
            if obs:
                messages.append({"role": "user", "content": obs})
            if action:
                messages.append({"role": "assistant", "content": action})
        
        return ARTTrajectory(
            messages=messages,
            reward=score,
            task_id=sample.get("task_id", 0),
            metadata={
                "task_name": sample.get("task_name", ""),
                "time_taken": sample.get("time_taken", 0),
            },
        )
    
    def convert_batch(
        self,
        samples: List[Dict[str, Any]],
        env_type: str = "swe-synth",
    ) -> ARTBatch:
        """
        Convert a batch of samples.
        
        Args:
            samples: List of sample dicts
            env_type: "swe-synth" or "liveweb"
            
        Returns:
            ARTBatch with converted trajectories
        """
        trajectories = []
        
        convert_fn = (
            self.convert_swe_synth_sample
            if env_type == "swe-synth"
            else self.convert_liveweb_sample
        )
        
        for sample in samples:
            traj = convert_fn(sample)
            if traj is not None:
                trajectories.append(traj)
        
        total_reward = sum(t.reward for t in trajectories)
        avg_reward = total_reward / len(trajectories) if trajectories else 0.0
        
        stats = {
            "total_samples": len(samples),
            "converted_samples": len(trajectories),
            "avg_reward": avg_reward,
            "total_reward": total_reward,
            "conversion_rate": len(trajectories) / len(samples) if samples else 0.0,
        }
        
        return ARTBatch(trajectories=trajectories, stats=stats)
    
    def load_and_convert(
        self,
        input_path: str,
        env_type: str = "swe-synth",
    ) -> ARTBatch:
        """
        Load samples from file and convert.
        
        Args:
            input_path: Path to JSON/JSONL file
            env_type: Environment type
            
        Returns:
            ARTBatch with converted trajectories
        """
        path = Path(input_path)
        
        if path.suffix == ".jsonl":
            samples = []
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        else:
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]
        
        return self.convert_batch(samples, env_type)
    
    def _format_problem_statement(
        self,
        problem_statement: str,
        instance_id: str,
        original_content: str,
    ) -> str:
        """Format problem statement for training."""
        if "<pr_description>" in original_content:
            return original_content
        
        return f"""<pr_description>
Consider the following PR description:
"{problem_statement}"
</pr_description>

<instructions>
# Task Instructions

## Overview
You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description.

IMPORTANT: This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.

## Command Execution Rules
You are operating in an environment where:
1. You write a single command
2. The system executes that command in a subshell
3. You see the result
4. You write your next command
</instructions>"""
    
    def _get_liveweb_system_prompt(self) -> str:
        """Get LIVEWEB system prompt."""
        return """You are a helpful assistant that can interact with a web browser to complete tasks.
Your response must contain exactly ONE action in the specified format.

Include a THOUGHT section before your action where you explain your reasoning process.

<format_example>
THOUGHT: Your reasoning and analysis here

```json
{
  "action": "click",
  "selector": "#submit-button",
  "value": null
}
```
</format_example>

Available actions:
- click: Click on an element
- type: Type text into an input
- navigate: Go to a URL
- scroll: Scroll the page
- wait: Wait for specified time
- extract: Extract content from page"""


def load_sample_files(
    input_paths: List[str],
    env_type: str = "swe-synth",
    output_path: Optional[str] = None,
) -> ARTBatch:
    """
    Load and convert multiple sample files.
    
    Args:
        input_paths: List of input file paths
        env_type: Environment type
        output_path: Optional output path for converted data
        
    Returns:
        ARTBatch with all converted trajectories
    """
    converter = SampleConverter()
    
    all_trajectories: List[ARTTrajectory] = []
    total_stats: Dict[str, Any] = {
        "total_samples": 0,
        "converted_samples": 0,
        "files_processed": 0,
        "avg_reward": 0.0,
    }
    
    for path in input_paths:
        batch = converter.load_and_convert(path, env_type)
        all_trajectories.extend(batch.trajectories)
        total_stats["total_samples"] += batch.stats["total_samples"]
        total_stats["converted_samples"] += batch.stats["converted_samples"]
        total_stats["files_processed"] += 1
    
    if all_trajectories:
        total_stats["avg_reward"] = sum(t.reward for t in all_trajectories) / len(all_trajectories)
    else:
        total_stats["avg_reward"] = 0.0
    
    batch = ARTBatch(trajectories=all_trajectories, stats=total_stats)
    
    if output_path:
        batch.to_jsonl(output_path)
    
    return batch
