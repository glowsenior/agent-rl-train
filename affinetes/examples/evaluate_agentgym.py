import asyncio
import os
import sys
import affinetes as af_env

async def main():
    agentgym_type = "textcraft-v2"
    env = af_env.load_env(
        image=f"bignickeye/agentgym:{agentgym_type}",
        pull=True,
        cleanup=False,
    )

    print("\nRunning evaluation...")
    result = await env.evaluate(
        model="openai/gpt-oss-20b",
        base_url="http://172.17.0.2:30000",
        temperature=0.7,
        task_id=10,
        max_round=10,
        timeout=600
    )

    if result.get('error'):
        print(f"\nError occurred:")
        print(f"  {result['error']}")
        return

    print(f"\nResults:")
    print(f"Task: {result['task_name']}")
    print(f"Reward: {result['score']:.3f}")
    print(f"Success: {result['success']}")
    print(f"Time: {result['time_taken']:.1f}s")

    if result.get('extra'):
        print(f"\nextra:")
        print(f"  {result['extra']}")

if __name__ == "__main__":
    asyncio.run(main())