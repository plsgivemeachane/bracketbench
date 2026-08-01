"""
LLM management module for BracketBench.

This module provides classes and functions for managing different Large Language Models,
including their initialization, configuration, and interaction.
"""

from .base import LLMInterface
from .openai_llms import OpenAILLM, GPT35LLM, GPT4LLM, GPT4TurboLLM
from .manager import LLMManager, ModelRegistry

__all__ = [
    # Base interface
    "LLMInterface",
    
    # OpenAI implementations
    "OpenAILLM",
    "GPT35LLM",
    "GPT4LLM",
    "GPT4TurboLLM",
    
    # Manager and registry
    "LLMManager",
    "ModelRegistry",
]