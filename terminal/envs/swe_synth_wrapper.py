"""
SWE-SYNTH environment wrapper for ART training.

Wraps the affinetes SWE-SYNTH OpenEnv interface for use with
Agent Reinforcement Trainer (ART).
"""

import os
import sys
import time
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from affinetes.core.openenv import OpenEnvResponse
except ImportError:
    OpenEnvResponse = None

from envs.art_wrapper import BaseTerminalEnv, StepResult

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


@dataclass
class EpisodeState:
    episode_id: str
    task_id: int
    seed: int
    step_count: int = 0
    done: bool = False
    truncated: bool = False
    messages: List[Dict[str, Any]] = field(default_factory=list)
    bug_instance: Optional[Dict[str, Any]] = None
    start_time: float = field(default_factory=time.time)


class SWESynthEnv(BaseTerminalEnv):
    """
    SWE-SYNTH environment wrapper for ART.
    
    Provides reset/step/stop interface compatible with ART's
    trajectory generation loop.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "/tmp/swe-synth-cache",
        dockerhub_username: str = "jefzda",
        **kwargs
    ):
        """
        Initialize SWE-SYNTH environment.
        
        Args:
            api_key: API key for LLM service
            cache_dir: Local cache directory
            dockerhub_username: Docker Hub username for images
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.cache_dir = cache_dir
        self.dockerhub_username = dockerhub_username
        
        self._actor = None
        self._episodes: Dict[str, EpisodeState] = {}
        
    def _get_actor(self):
        """Lazy initialization of the SynthActor."""
        if self._actor is None:
            import importlib
            affinetes_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            swe_synth_dir = os.path.join(affinetes_root, "affinetes", "environments", "SWE-SYNTH")
            if os.path.isdir(swe_synth_dir):
                sys.path.insert(0, swe_synth_dir)
                from env import SynthActor
            else:
                # Fallback: try underscore variant or package import
                try:
                    mod = importlib.import_module("affinetes.environments.SWE_SYNTH")
                    SynthActor = mod.SynthActor
                except ImportError:
                    raise ImportError(
                        f"Cannot find SWE-SYNTH environment. "
                        f"Searched: {swe_synth_dir} and affinetes.environments.SWE_SYNTH"
                    )
            self._actor = SynthActor(
                api_key=self.api_key,
                cache_dir=self.cache_dir,
                dockerhub_username=self.dockerhub_username,
            )
        return self._actor
    
    async def reset(
        self,
        task_id: int,
        seed: Optional[int] = None,
        step_limit: int = 100,
        command_timeout: int = 300,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Reset and start a new episode.
        
        Returns:
            Tuple of (initial_observation, info_dict with episode_id)
        """
        import random
        import uuid
        
        actor = self._get_actor()
        resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        result = await actor.reset(
            task_id=task_id,
            seed=resolved_seed,
            step_limit=step_limit,
            command_timeout=command_timeout,
        )

        # Use the actor's episode_id, not a self-generated one
        episode_id = result.episode_id or uuid.uuid4().hex

        state = EpisodeState(
            episode_id=episode_id,
            task_id=task_id,
            seed=resolved_seed,
        )
        state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT, "timestamp": time.time()},
            {"role": "user", "content": result.observation, "timestamp": time.time()},
        ]
        
        if result.info.get("bug_instance"):
            state.bug_instance = result.info["bug_instance"]
        
        self._episodes[episode_id] = state
        
        info = {
            "episode_id": episode_id,
            "task_id": task_id,
            "seed": resolved_seed,
            "swe_instance_id": result.info.get("swe_instance_id"),
            "bug_types": result.info.get("bug_types", []),
        }
        
        if result.info.get("error"):
            info["error"] = result.info["error"]

        return result.observation, info
    
    async def step(self, action: str, episode_id: Optional[str] = None) -> StepResult:
        """
        Execute action in the episode.
        
        Args:
            action: LLM response with THOUGHT + ```bash ... ```
            episode_id: Episode identifier
            
        Returns:
            StepResult with observation, reward, done, info
        """
        if not episode_id:
            return StepResult(
                observation="Error: No episode_id provided. Call reset() first.",
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
        
        actor = self._get_actor()
        
        state.messages.append({"role": "assistant", "content": action, "timestamp": time.time()})
        state.step_count += 1
        
        result = await actor.step(action, episode_id)
        
        state.messages.append({"role": "user", "content": result.observation, "timestamp": time.time()})
        
        if result.done:
            state.done = True
            
        if result.truncated:
            state.truncated = True
        
        info = result.info.copy()
        info["step_count"] = state.step_count
        info["episode_id"] = episode_id
        
        reward = result.reward
        
        return StepResult(
            observation=result.observation,
            reward=reward,
            done=result.done,
            truncated=result.truncated,
            info=info,
        )
    
    async def stop(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Stop and cleanup episode."""
        if not episode_id:
            return {"status": "ok", "stopped": False, "message": "No episode_id provided"}
        
        state = self._episodes.pop(episode_id, None)
        if not state:
            return {"status": "ok", "stopped": False, "message": f"Episode {episode_id} not found"}
        
        actor = self._get_actor()
        result = await actor.stop(episode_id)
        
        return {
            "status": "ok",
            "stopped": True,
            "episode_id": episode_id,
            "step_count": state.step_count,
            "done": state.done,
            **result,
        }
    
    def get_system_prompt(self) -> str:
        """Get system prompt for SWE-SYNTH."""
        return SYSTEM_PROMPT
