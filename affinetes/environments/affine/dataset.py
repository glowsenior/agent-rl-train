import logging
import random
from typing import Any, Optional
from datasets import load_dataset

logger = logging.getLogger("affine")


class HFDataset:
    """
    HuggingFace dataset wrapper with local caching and get_by_id support.
    
    Features:
    - Automatic caching to disk (managed by HuggingFace datasets library)
    - Random sampling with optional seed
    - Deterministic access via get_by_id
    - Compatible with R2Dataset interface
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        seed: Optional[int] = None,
        preload: bool = False,
    ):
        """
        Initialize HuggingFace dataset.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub (e.g., "satpalsr/rl-python")
            split: Dataset split to use (default: "train")
            seed: Random seed for reproducibility (optional)
            preload: If True, load entire dataset into memory. If False, use streaming/lazy loading
        """
        self.dataset_name = dataset_name
        self.split = split
        self._rng = random.Random(seed)
        
        logger.info(f"Loading HuggingFace dataset '{dataset_name}' (split='{split}')")
        
        # Load dataset from HuggingFace Hub
        # This will automatically cache to ~/.cache/huggingface/datasets
        self._dataset = load_dataset(
            dataset_name,
            split=split
        )
        
        self.total_size = len(self._dataset)
        logger.info(f"Dataset '{dataset_name}' loaded: {self.total_size} samples")
    
    async def get(self) -> Any:
        """
        Get a random sample from the dataset.
        
        Returns:
            A randomly selected sample from the dataset
        """
        if self.total_size == 0:
            raise RuntimeError(f"Dataset '{self.dataset_name}' is empty")
        
        # Randomly select an index
        idx = self._rng.randint(0, self.total_size - 1)
        
        # Get sample by index
        sample = self._dataset[idx]
        
        logger.debug(f"Retrieved random sample from dataset '{self.dataset_name}' (index={idx})")
        return sample
    
    def __aiter__(self):
        """Support async iteration"""
        return self
    
    async def __anext__(self) -> Any:
        """Get next random sample"""
        return await self.get()
    
    async def get_by_id(self, task_id: int) -> Any:
        """
        Get a specific sample by task ID (deterministic).
        
        Args:
            task_id: Task identifier (0-based index into dataset)
            
        Returns:
            Sample at the specified index
            
        Raises:
            ValueError: If task_id is out of range
        """
        # Validate task_id range
        if task_id < 0:
            raise ValueError(f"task_id must be non-negative, got {task_id}")
        
        if task_id >= self.total_size:
            raise ValueError(
                f"task_id {task_id} out of range [0, {self.total_size-1}]. "
                f"Dataset '{self.dataset_name}' contains {self.total_size} samples."
            )
        
        # Get sample by index
        sample = self._dataset[task_id]
        
        logger.debug(f"Retrieved task_id {task_id} from dataset '{self.dataset_name}'")
        return sample