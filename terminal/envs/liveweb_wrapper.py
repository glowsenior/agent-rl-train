"""
LIVEWEB environment wrapper for ART training.

Wraps the affinetes LIVEWEB interface for use with
Agent Reinforcement Trainer (ART).
"""

import os
import sys
import time
import json
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.art_wrapper import BaseTerminalEnv, StepResult

LIVEWEB_SYSTEM_PROMPT = """You are a helpful assistant that can interact with a web browser to complete tasks.
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
- click: Click on an element (requires selector)
- type: Type text into an input (requires selector and value)
- navigate: Go to a URL (requires value as URL)
- scroll: Scroll the page (requires value as "up" or "down")
- wait: Wait for a specified time in seconds (requires value)
- extract: Extract content from page (requires selector)

Failure to follow these rules will cause your response to be rejected."""


@dataclass
class LiveWebEpisodeState:
    episode_id: str
    task_id: int
    seed: int
    step_count: int = 0
    done: bool = False
    truncated: bool = False
    messages: List[Dict[str, Any]] = field(default_factory=list)
    task_info: Optional[Dict[str, Any]] = None
    start_time: float = field(default_factory=time.time)


class LiveWebEnv(BaseTerminalEnv):
    """
    LIVEWEB environment wrapper for ART.
    
    Provides reset/step/stop interface for web-based tasks.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://llm.chutes.ai/v1",
        **kwargs
    ):
        """
        Initialize LIVEWEB environment.
        
        Args:
            api_key: API key for services
            base_url: LLM API base URL
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY") or os.getenv("API_KEY")
        self.base_url = base_url
        self.taostats_api_key = os.getenv("TAOSTATS_API_KEY", "")
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY", "")
        
        self._env = None
        self._episodes: Dict[str, LiveWebEpisodeState] = {}
        
    async def _init_env(self):
        """Lazy initialization of the LIVEWEB environment."""
        if self._env is None:
            import affinetes as af
            env_vars = {
                "API_KEY": self.api_key,
                "LIVEWEB_VERBOSE": "true",
            }
            if self.taostats_api_key:
                env_vars["TAOSTATS_API_KEY"] = self.taostats_api_key
            if self.coingecko_api_key:
                env_vars["COINGECKO_API_KEY"] = self.coingecko_api_key
            
            self._env = af.load_env(
                image="affinefoundation/liveweb-arena:latest",
                mode="docker",
                env_vars=env_vars,
                pull=True,
            )
        return self._env
    
    async def reset(
        self,
        task_id: int,
        seed: Optional[int] = None,
        step_limit: int = 50,
        command_timeout: int = 300,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Reset and start a new episode.
        
        Returns:
            Tuple of (initial_observation, info_dict with episode_id)
        """
        env = await self._init_env()
        resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        episode_id = uuid.uuid4().hex
        
        result = await env.reset(
            task_id=task_id,
            seed=resolved_seed,
        )
        
        state = LiveWebEpisodeState(
            episode_id=episode_id,
            task_id=task_id,
            seed=resolved_seed,
        )
        
        initial_obs = result.get("observation", "")
        if isinstance(result, dict):
            observation = initial_obs
            task_info = result.get("info", {})
        else:
            observation = str(result)
            task_info = {}
        
        state.messages = [
            {"role": "system", "content": LIVEWEB_SYSTEM_PROMPT, "timestamp": time.time()},
            {"role": "user", "content": observation, "timestamp": time.time()},
        ]
        state.task_info = task_info
        
        self._episodes[episode_id] = state
        
        info = {
            "episode_id": episode_id,
            "task_id": task_id,
            "seed": resolved_seed,
            "task_name": task_info.get("task_name"),
        }
        
        return observation, info
    
    async def step(self, action: str, episode_id: Optional[str] = None) -> StepResult:
        """
        Execute action in the episode.
        
        Args:
            action: LLM response with action specification
            episode_id: Episode identifier
            
        Returns:
            StepResult with observation, reward, done, info
        """
        if not episode_id:
            return StepResult(
                observation="Error: No episode_id provided.",
                reward=0.0,
                done=True,
                truncated=True,
                info={"error": "no_episode_id"},
            )
        
        state = self._episodes.get(episode_id)
        if not state:
            return StepResult(
                observation=f"Error: Episode {episode_id} not found.",
                reward=0.0,
                done=True,
                truncated=True,
                info={"error": "episode_not_found"},
            )
        
        if state.done:
            return StepResult(
                observation="Episode already finished.",
                reward=0.0,
                done=True,
                info={"error": "episode_done"},
            )
        
        env = await self._init_env()
        
        state.messages.append({"role": "assistant", "content": action, "timestamp": time.time()})
        state.step_count += 1
        
        result = await env.step(action, episode_id=episode_id)
        
        if isinstance(result, dict):
            observation = result.get("observation", "")
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            truncated = result.get("truncated", False)
            info = result.get("info", {})
        else:
            observation = str(result)
            reward = 0.0
            done = False
            truncated = False
            info = {}
        
        state.messages.append({"role": "user", "content": observation, "timestamp": time.time()})
        
        if done:
            state.done = True
        if truncated:
            state.truncated = True
        
        info["step_count"] = state.step_count
        info["episode_id"] = episode_id
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )
    
    async def stop(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Stop and cleanup episode."""
        if not episode_id:
            return {"status": "ok", "stopped": False, "message": "No episode_id provided"}
        
        state = self._episodes.pop(episode_id, None)
        if not state:
            return {"status": "ok", "stopped": False, "message": f"Episode {episode_id} not found"}
        
        return {
            "status": "ok",
            "stopped": True,
            "episode_id": episode_id,
            "step_count": state.step_count,
            "done": state.done,
        }
    
    async def cleanup(self):
        """Cleanup environment resources."""
        if self._env is not None:
            await self._env.cleanup()
            self._env = None
        self._episodes.clear()
    
    def get_system_prompt(self) -> str:
        """Get system prompt for LIVEWEB."""
        return LIVEWEB_SYSTEM_PROMPT
