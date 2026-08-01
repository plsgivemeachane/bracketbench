"""
BracketBench - A comprehensive LLM benchmarking system.

This package provides tools and utilities for benchmarking Large Language Models
across various tasks and metrics.
"""

__version__ = "0.1.0"
__author__ = "BracketBench Team"
__email__ = "contact@bracketbench.org"

# Import key classes and functions for easier access
from . import config
from .llms import LLMManager
from .benchmarking import BenchmarkRunner

__all__ = [
    "config",
    "LLMManager",
    "BenchmarkRunner",
]