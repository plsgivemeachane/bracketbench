"""
Data models for BracketBench benchmarking.

This module provides data classes used throughout the benchmarking system.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results for a single test case."""
    test_case_id: str
    input_prompt: str
    expected_output: Optional[str] = None
    actual_output: Optional[str] = None
    execution_time: float = 0.0
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)