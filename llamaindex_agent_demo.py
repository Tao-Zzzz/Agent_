# -*- coding: utf-8 -*-
"""Offline LlamaIndex Agent demo.

This version does not call Hugging Face or OpenAI. It uses a tiny fake LLM so
you can see the AgentWorkflow -> FunctionTool -> tool result flow locally.

Run:
    python .\llamaindex_agent_demo.py
"""

import asyncio
from typing import Any, Generator

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM
from llama_index.core.tools import FunctionTool


def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer."""
    print(f"[tool called] multiply(a={a}, b={b})")
    return a * b


class OfflineReActLLM(CustomLLM):
    """A tiny deterministic LLM that emits one ReAct tool call, then an answer."""

    step: int = 0

    def __init__(self) -> None:
        super().__init__(callback_manager=CallbackManager([]))

    @classmethod
    def class_name(cls) -> str:
        return "OfflineReActLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_chat_model=True, num_output=256)

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.step += 1

        if self.step == 1:
            text = (
                "Thought: I need to multiply 2 and 2 using the tool.\n"
                "Action: multiply\n"
                'Action Input: {"a": 2, "b": 2}'
            )
        else:
            text = "Thought: I have the tool result.\nAnswer: 2 times 2 is 4."

        return CompletionResponse(text=text)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        response = self.complete(prompt, formatted=formatted, **kwargs)

        def gen() -> Generator[CompletionResponse, None, None]:
            yield CompletionResponse(text=response.text, delta=response.text)

        return gen()


async def main() -> None:
    llm = OfflineReActLLM()
    multiply_tool = FunctionTool.from_defaults(fn=multiply)

    agent = AgentWorkflow.from_tools_or_functions(
        [multiply_tool],
        llm=llm,
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    response = await agent.run("What is 2 times 2?")
    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
