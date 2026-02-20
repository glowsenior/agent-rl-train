"""
ART-compatible environment wrapper for terminal agent training.

Provides a unified interface for SWE-SYNTH and LIVEWEB environments
that integrates with OpenPipe's ART (Agent Reinforcement Trainer).
"""

import asyncio
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from affinetes.core.openenv import OpenEnvResponse


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class BaseTerminalEnv(ABC):
    """Base class for terminal agent environments."""
    
    @abstractmethod
    async def reset(self, task_id: int, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and return initial observation."""
        pass
    
    @abstractmethod
    async def step(self, action: str, episode_id: Optional[str] = None) -> StepResult:
        """Execute action and return result."""
        pass
    
    @abstractmethod
    async def stop(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Stop and cleanup episode."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for the environment."""
        pass


class TerminalAgentEnv:
    """
    ART-compatible wrapper that unifies SWE-SYNTH and LIVEWEB environments.
    
    This class provides:
    - Unified interface for multiple environment types
    - Episode management for concurrent rollouts
    - Integration with ART's trajectory generation
    """
    
    def __init__(
        self,
        env_type: str = "swe-synth",
        api_key: Optional[str] = None,
        max_episodes: int = 10,
        **kwargs
    ):
        """
        Initialize terminal agent environment.
        
        Args:
            env_type: "swe-synth" or "liveweb"
            api_key: API key for LLM calls
            max_episodes: Maximum concurrent episodes
            **kwargs: Additional environment-specific arguments
        """
        self.env_type = env_type
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.max_episodes = max_episodes
        self.kwargs = kwargs
        
        self._env: Optional[BaseTerminalEnv] = None
        self._episodes: Dict[str, Dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_episodes)
        
    async def _init_env(self):
        """Lazy initialization of the underlying environment."""
        if self._env is not None:
            return
            
        if self.env_type == "swe-synth":
            from .swe_synth_wrapper import SWESynthEnv
            self._env = SWESynthEnv(api_key=self.api_key, **self.kwargs)
        elif self.env_type == "liveweb":
            from .liveweb_wrapper import LiveWebEnv
            self._env = LiveWebEnv(api_key=self.api_key, **self.kwargs)
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")
    
    async def reset(
        self,
        task_id: int,
        seed: Optional[int] = None,
        step_limit: int = 100,
        command_timeout: int = 300,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Reset environment for a new episode.
        
        Args:
            task_id: Task identifier
            seed: Random seed
            step_limit: Maximum steps per episode
            command_timeout: Timeout for each command
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        await self._init_env()
        
        result = await self._env.reset(
            task_id=task_id,
            seed=seed,
            step_limit=step_limit,
            command_timeout=command_timeout,
        )
        
        return result
    
    async def step(
        self,
        action: str,
        episode_id: Optional[str] = None,
    ) -> StepResult:
        """
        Execute action in the environment.
        
        Args:
            action: LLM's response containing THOUGHT + ```bash ... ```
            episode_id: Episode identifier (required for multi-episode)
            
        Returns:
            StepResult with observation, reward, done, info
        """
        await self._init_env()
        
        result = await self._env.step(action, episode_id)
        return result
    
    async def stop(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Stop and cleanup episode."""
        if self._env is None:
            return {"status": "ok", "stopped": False, "message": "Environment not initialized"}
        return await self._env.stop(episode_id)
    
    def get_system_prompt(self) -> str:
        """Get system prompt for this environment type."""
        if self._env is None:
            return ""
        return self._env.get_system_prompt()
    
    async def cleanup(self):
        """Cleanup all resources."""
        for episode_id in list(self._episodes.keys()):
            try:
                await self.stop(episode_id)
            except Exception:
                pass
        self._episodes.clear()
        self._executor.shutdown(wait=False)


def create_env_pool(
    env_type: str = "swe-synth",
    pool_size: int = 4,
    api_key: Optional[str] = None,
    **kwargs
) -> List[TerminalAgentEnv]:
    """
    Create a pool of environments for parallel rollout generation.
    
    Args:
        env_type: Environment type
        pool_size: Number of environments
        api_key: API key
        **kwargs: Additional arguments
        
    Returns:
        List of TerminalAgentEnv instances
    """
    return [
        TerminalAgentEnv(env_type=env_type, api_key=api_key, **kwargs)
        for _ in range(pool_size)
    ]
