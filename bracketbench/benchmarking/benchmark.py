"""
Core Benchmarking class for BracketBench.

This module provides the main simplified Benchmarking class for running benchmarks on LLMs,
including test case execution and scoring functionality.
"""

import time
from typing import Dict, Any, List, Optional, Callable, Union

from .models import BenchmarkResult
from .test_cases import TestCase
from .scoring import ScoringEngine
from .results import ResultsHandler


class Benchmarking:
    """
    Main simplified Benchmarking class for running benchmarks on LLMs.
    
    This class provides functionality to run benchmarks on LLM models,
    including test case execution, scoring, and result aggregation.
    """
    
    def __init__(
        self, 
        generate_function: Callable[[str], str],
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the Benchmarking class.
        
        Args:
            generate_function: Function that takes a prompt and returns generated text
            name: Name of the benchmark (optional)
            config: Configuration dictionary for the benchmark
        """
        self.name = name or "Benchmark"
        self.config = config or {}
        self.generate_function = generate_function
        
        # Initialize components
        self.scoring_engine = ScoringEngine()
        self.results_handler = ResultsHandler()
        
        # Benchmark state
        self.test_cases: List[TestCase] = []
        self.results: List[BenchmarkResult] = []
    
    def add_test_case(self, test_case: TestCase) -> None:
        """
        Add a test case to the benchmark.
        
        Args:
            test_case: TestCase object to add
            
        Raises:
            TypeError: If test_case is not a valid TestCase object
        """
        if not isinstance(test_case, TestCase):
            raise TypeError(f"Expected TestCase object, got {type(test_case)}")
        
        self.test_cases.append(test_case)
    
    def add_test_cases(self, test_cases: List[TestCase]) -> None:
        """
        Add multiple test cases to the benchmark.
        
        Args:
            test_cases: List of TestCase objects to add
            
        Raises:
            TypeError: If any item in test_cases is not a valid TestCase object
        """
        for test_case in test_cases:
            self.add_test_case(test_case)
    
    def run_benchmark(
        self, 
        test_cases: Optional[List[TestCase]] = None,
        track_time: bool = True,
        **kwargs
    ) -> List[BenchmarkResult]:
        """
        Run the benchmark on the specified test cases.
        
        Args:
            test_cases: List of test cases to run (uses all test cases if None)
            track_time: Whether to track execution time
            **kwargs: Additional parameters for generation
            
        Returns:
            List of BenchmarkResult objects
            
        Raises:
            ValueError: If no test cases are available
        """
        # Use provided test cases or all available test cases
        test_cases_to_run = test_cases or self.test_cases
        if not test_cases_to_run:
            raise ValueError("No test cases available for benchmarking")
        
        # Clear previous results
        self.results = []
        
        # Run each test case
        for test_case in test_cases_to_run:
            result = self._run_single_test_case(
                test_case, 
                track_time=track_time,
                **kwargs
            )
            
            self.results.append(result)
        
        return self.results
    
    def _run_single_test_case(
        self, 
        test_case: TestCase,
        track_time: bool = True,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a single test case and return the result.
        
        Args:
            test_case: TestCase object to run
            track_time: Whether to track execution time
            **kwargs: Additional parameters for generation
            
        Returns:
            BenchmarkResult object
        """
        # Create result object
        result = BenchmarkResult(
            test_case_id=test_case.id,
            input_prompt=test_case.input_prompt,
            expected_output=test_case.expected_output
        )
        
        try:
            # Generate output
            start_time = time.time() if track_time else None
            
            # Call the generate function
            actual_output = self.generate_function(
                test_case.input_prompt,
                **kwargs
            )
            
            execution_time = time.time() - start_time if track_time and start_time else 0.0
            
            # Update result
            result.actual_output = actual_output
            result.execution_time = execution_time
            
            # Score the output
            result.score = self.scoring_engine.score_output(
                actual_output, 
                test_case.expected_output,
                test_case.input_prompt
            )
            
            # Add metadata
            result.metadata = {
                "test_case_metadata": test_case.metadata,
                "generation_params": kwargs
            }
            
        except Exception as e:
            # Store error information in metadata
            result.metadata["error"] = str(e)
        
        return result
    
    def get_results(self) -> List[BenchmarkResult]:
        """
        Get all benchmark results.
        
        Returns:
            List of BenchmarkResult objects
        """
        return self.results.copy()
    
    def get_result_by_test_case_id(self, test_case_id: str) -> Optional[BenchmarkResult]:
        """
        Get the result for a specific test case.
        
        Args:
            test_case_id: ID of the test case
            
        Returns:
            BenchmarkResult object if found, None otherwise
        """
        for result in self.results:
            if result.test_case_id == test_case_id:
                return result
        return None
    
    def save_results(self, filepath: str) -> None:
        """
        Save benchmark results to a file.
        
        Args:
            filepath: Path to save the results
        """
        self.results_handler.save_results(self.results, filepath)
    
    def clear_results(self) -> None:
        """Clear all benchmark results."""
        self.results = []
    
    def clear_test_cases(self) -> None:
        """Clear all test cases."""
        self.test_cases = []
    
    def reset(self) -> None:
        """Reset the benchmark to its initial state."""
        self.clear_results()
        self.clear_test_cases()
    
    def __str__(self) -> str:
        """String representation of the Benchmarking object."""
        return f"Benchmarking(name='{self.name}', test_cases={len(self.test_cases)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the Benchmarking object."""
        return f"Benchmarking(name='{self.name}', test_cases={len(self.test_cases)}, results={len(self.results)})"