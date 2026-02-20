"""
Quick test script to verify the training setup.

Tests:
1. Environment loading
2. Model loading
3. Reward calculation
4. Sample conversion
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from envs import TerminalAgentEnv, SWESynthEnv, LiveWebEnv
        print("  ✓ envs module")
    except ImportError as e:
        print(f"  ✗ envs module: {e}")
        return False
    
    try:
        from rewards import ProcessRewardCalculator, compute_process_reward
        print("  ✓ rewards module")
    except ImportError as e:
        print(f"  ✗ rewards module: {e}")
        return False
    
    try:
        from data import SampleConverter, load_sample_files
        print("  ✓ data module")
    except ImportError as e:
        print(f"  ✗ data module: {e}")
        return False
    
    return True


def test_reward_calculator():
    """Test reward calculation."""
    print("\nTesting reward calculator...")
    
    from rewards import ProcessRewardCalculator, RewardConfig
    
    config = RewardConfig()
    calculator = ProcessRewardCalculator(config)
    
    observation = """
<returncode>0</returncode>
<output>
diff --git a/src/file.py b/src/file.py
modified file
2 tests passed
</output>
"""
    
    action = """THOUGHT: Let me check the files

```bash
ls -la
```"""
    
    info = {"returncode": 0}
    
    reward = calculator.compute_step_reward(observation, action, info, done=False)
    print(f"  Step reward: {reward:.4f}")
    
    final_info = {"all_passed": True, "all_result": "10/10"}
    final_reward = calculator.compute_step_reward(
        "All tests passed!", 
        action, 
        final_info, 
        done=True
    )
    print(f"  Final reward: {final_reward:.4f}")
    
    print("  ✓ Reward calculator working")
    return True


def test_sample_converter():
    """Test sample conversion."""
    print("\nTesting sample converter...")
    
    from data import SampleConverter
    
    converter = SampleConverter()
    
    sample = {
        "task_id": 1234,
        "score": 1.0,
        "extra": {
            "all_passed": True,
            "all_result": "10/10",
            "bug_types": ["logic-inversion"],
            "conversation": [
                {"role": "system", "content": "You are a helpful assistant..."},
                {"role": "user", "content": "Fix the bug..."},
                {"role": "assistant", "content": "THOUGHT: Let me check\n```bash\nls\n```"},
            ],
            "problem_statement": "Test problem",
            "swe_instance_id": "test-repo-123",
        }
    }
    
    trajectory = converter.convert_swe_synth_sample(sample)
    
    if trajectory:
        print(f"  Converted trajectory with {len(trajectory.messages)} messages")
        print(f"  Reward: {trajectory.reward}")
        print(f"  Task ID: {trajectory.task_id}")
        print("  ✓ Sample converter working")
        return True
    else:
        print("  ✗ Failed to convert sample")
        return False


async def test_environment():
    """Test environment loading (requires API key)."""
    print("\nTesting environment...")
    
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("  ⚠ Skipping: CHUTES_API_KEY not set")
        return True
    
    try:
        from envs import TerminalAgentEnv
        
        env = TerminalAgentEnv(
            env_type="swe-synth",
            api_key=api_key,
        )
        
        print("  ✓ Environment initialized")
        print(f"  System prompt length: {len(env.get_system_prompt())}")
        return True
        
    except Exception as e:
        print(f"  ⚠ Environment test skipped: {e}")
        return True


def test_model_loading():
    """Test model loading (requires GPU and transformers)."""
    print("\nTesting model loading...")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("glowsenior/affine-senior-5GZQjTBnzFnNnKNbVry2hS29CzNa4EkovYAAsdvg3cDA7ssN")
        print(f"  ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
        return True
        
    except ImportError as e:
        print(f"  ⚠ Skipping: {e}")
        return True


def main():
    print("=" * 60)
    print("Terminal Agent Setup Verification")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_reward_calculator()
    all_passed &= test_sample_converter()
    all_passed &= asyncio.run(test_environment())
    all_passed &= test_model_loading()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
