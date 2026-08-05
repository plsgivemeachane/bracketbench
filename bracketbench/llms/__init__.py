"""LLM interface for BracketBench.

The focused JSON-repair path depends only on :class:`~bracketbench.llms.base.LLMInterface`
-- the minimal contract a repair model must satisfy (``generate(prompt) -> str`` emitting an
edit script per ADR-0005/0006). Concrete provider implementations (OpenAI, OpenRouter) were
part of the legacy multi-provider registry stripped per ADR-0001; bring your own model by
subclassing ``LLMInterface``.
"""

from .base import LLMInterface

__all__ = ["LLMInterface"]
