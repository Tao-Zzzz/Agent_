import asyncio
import os
from dotenv import load_dotenv
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer."""
    return a * b


async def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    print(token)
    if not token:
        raise RuntimeError("Missing HF_TOKEN in .env")
    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=token,
    )
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    agent = AgentWorkflow.from_tools_or_functions(
        [multiply_tool],
        llm=llm,
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )
    response = await agent.run("What is 2 times 2?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
