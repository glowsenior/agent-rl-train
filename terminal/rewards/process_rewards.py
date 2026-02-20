"""
Process Reward Calculator for Terminal Agent RL Training.

Implements fine-grained reward signals at each step of the agent's
trajectory, enabling faster learning compared to sparse final rewards.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RewardConfig:
    """Configuration for process rewards."""
    
    valid_command: float = 0.01
    command_success: float = 0.02
    code_modified: float = 0.03
    test_discovered: float = 0.02
    error_found: float = 0.01
    invalid_format: float = -0.05
    timeout_penalty: float = -0.02
    no_op_penalty: float = -0.01
    
    final_success: float = 1.0
    final_partial_multiplier: float = 0.5
    
    max_step_reward: float = 0.1
    min_step_reward: float = -0.1


class ProcessRewardCalculator:
    """
    Calculates process rewards for terminal agent trajectories.
    
    Reward Components:
    1. Command execution rewards
    2. Code change detection
    3. Test discovery rewards
    4. Format validation penalties
    5. Final outcome rewards
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize with optional custom config."""
        self.config = config or RewardConfig()
        
        self._code_patterns = [
            r"diff --git",
            r"modified:",
            r"new file:",
            r"deleted:",
            r"@@.*@@",
            r"\+[^+]",
            r"-[^-]",
        ]
        
        self._test_patterns = [
            r"test",
            r"pytest",
            r"jest",
            r"unittest",
            r"PASS",
            r"FAIL",
            r"passed",
            r"failed",
            r"assertion",
        ]
        
        self._error_patterns = [
            r"error",
            r"exception",
            r"traceback",
            r"failed",
            r"invalid",
            r"not found",
            r"permission denied",
        ]
    
    def compute_step_reward(
        self,
        observation: str,
        action: str,
        info: Dict[str, Any],
        done: bool = False,
    ) -> float:
        """
        Compute reward for a single step.
        
        Args:
            observation: Environment observation after action
            action: Agent's action (LLM response)
            info: Step info dict
            done: Whether episode is finished
            
        Returns:
            Reward value for this step
        """
        reward = 0.0
        
        if self._is_invalid_format(action):
            reward += self.config.invalid_format
            return max(min(reward, self.config.max_step_reward), self.config.min_step_reward)
        
        if info.get("timeout"):
            reward += self.config.timeout_penalty
        
        returncode = info.get("returncode", info.get("output", {}).get("returncode"))
        if returncode == 0:
            reward += self.config.command_success
        elif returncode is not None and returncode != 0:
            pass
        
        if self._has_code_changes(observation):
            reward += self.config.code_modified
        
        if self._has_test_discovery(observation):
            reward += self.config.test_discovered
        
        if self._has_error_context(observation):
            reward += self.config.error_found
        
        if self._is_no_op(action, observation):
            reward += self.config.no_op_penalty

        # Clamp step-level rewards only; final reward is added unclamped
        reward = max(min(reward, self.config.max_step_reward), self.config.min_step_reward)

        if done:
            reward += self._compute_final_reward(info)

        return reward
    
    def compute_trajectory_reward(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute total reward and statistics for a trajectory.
        
        Args:
            trajectory: List of (observation, action, info, done) tuples
            
        Returns:
            Tuple of (total_reward, stats_dict)
        """
        total_reward = 0.0
        step_rewards = []
        
        for step in trajectory:
            obs = step.get("observation", "")
            action = step.get("action", "")
            info = step.get("info", {})
            done = step.get("done", False)
            
            step_reward = self.compute_step_reward(obs, action, info, done)
            step_rewards.append(step_reward)
            total_reward += step_reward
        
        final_info = trajectory[-1].get("info", {}) if trajectory else {}
        
        stats = {
            "total_reward": total_reward,
            "step_rewards": step_rewards,
            "num_steps": len(trajectory),
            "avg_step_reward": total_reward / len(trajectory) if trajectory else 0.0,
            "final_success": final_info.get("all_passed", False),
            "pass_rate": self._extract_pass_rate(final_info),
        }
        
        return total_reward, stats
    
    def _is_invalid_format(self, action: str) -> bool:
        """Check if action has invalid format (bash or json code block)."""
        if not action or not action.strip():
            return True

        bash_matches = re.findall(r"```bash\s*\n(.*?)\n```", action, re.DOTALL)
        json_matches = re.findall(r"```json\s*\n(.*?)\n```", action, re.DOTALL)

        total = len(bash_matches) + len(json_matches)

        # Must have exactly one code block (either bash or json)
        if total == 0:
            return True
        if total > 1:
            return True

        return False
    
    def _has_code_changes(self, observation: str) -> bool:
        """Detect if code was modified."""
        for pattern in self._code_patterns:
            if re.search(pattern, observation, re.IGNORECASE):
                return True
        return False
    
    def _has_test_discovery(self, observation: str) -> bool:
        """Detect if tests were discovered/run."""
        obs_lower = observation.lower()
        for pattern in self._test_patterns:
            if pattern.lower() in obs_lower:
                return True
        return False
    
    def _has_error_context(self, observation: str) -> bool:
        """Detect if error context is present (learning opportunity)."""
        obs_lower = observation.lower()
        for pattern in self._error_patterns:
            if pattern.lower() in obs_lower:
                return True
        return False
    
    def _is_no_op(self, action: str, observation: str) -> bool:
        """Detect if action was essentially a no-op."""
        obs_lower = observation.lower().strip()
        
        no_op_outputs = [
            "",
            "\n",
            "no output",
            "command not found",
        ]
        
        for no_op in no_op_outputs:
            if obs_lower == no_op.lower():
                return True
        
        return False
    
    def _compute_final_reward(self, info: Dict[str, Any]) -> float:
        """Compute final reward based on task outcome."""
        if info.get("all_passed"):
            return self.config.final_success
        
        pass_rate = self._extract_pass_rate(info)
        if pass_rate > 0:
            return self.config.final_partial_multiplier * pass_rate * self.config.final_success
        
        return 0.0
    
    def _extract_pass_rate(self, info: Dict[str, Any]) -> float:
        """Extract test pass rate from info."""
        all_result = info.get("all_result", "")
        if "/" in all_result:
            try:
                passed, total = all_result.split("/")
                return float(passed) / float(total)
            except (ValueError, ZeroDivisionError):
                pass
        
        target_result = info.get("target_result", "")
        if "/" in target_result:
            try:
                passed, total = target_result.split("/")
                return float(passed) / float(total)
            except (ValueError, ZeroDivisionError):
                pass
        
        if "pass_rate" in info:
            return info["pass_rate"]
        
        return 0.0


def compute_process_reward(
    observation: str,
    action: str,
    info: Dict[str, Any],
    done: bool = False,
    config: Optional[RewardConfig] = None,
) -> float:
    """
    Convenience function to compute process reward.
    
    Args:
        observation: Environment observation
        action: Agent's action
        info: Step info dict
        done: Whether episode is finished
        config: Optional reward config
        
    Returns:
        Reward value
    """
    calculator = ProcessRewardCalculator(config)
    return calculator.compute_step_reward(observation, action, info, done)
