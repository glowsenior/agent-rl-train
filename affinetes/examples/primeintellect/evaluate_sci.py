#!/usr/bin/env python3

import asyncio
import json
import sys
import os
import affinetes as af
from dotenv import load_dotenv

load_dotenv(override=True)

async def main():
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("Error: CHUTES_API_KEY not set")
        sys.exit(1)

    image_tag = af.build_image_from_env(
        env_path="environments/primeintellect/sci",
        image_tag="science:latest"
    )
    
    env = af.load_env(
        image=image_tag,
        mode="docker",
        env_vars={"CHUTES_API_KEY": api_key}
    )

    result = await env.evaluate(
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        task_id=50,
        judge_model="deepseek-ai/DeepSeek-V3.2-Speciale",
        judge_base_url="https://llm.chutes.ai/v1",
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())