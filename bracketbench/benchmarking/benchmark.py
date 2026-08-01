"""
Core Benchmarking class for BracketBench.

This module provides the main simplified Benchmarking class for running benchmarks on LLMs,
including test case execution and scoring functionality.
"""

import os
import json
import time
import logging
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


# Built-in standard test cases used when the CLI requests the "standard" set.
_STANDARD_TEST_CASES: List[TestCase] = [
    TestCase(
        id="standard-json-repair",
        input_prompt="Repair the following JSON and return ONLY valid JSON: {name: \"Alice\" age: 30}",
        expected_output='{"name": "Alice", "age": 30}',
        metadata={"category": "json_repair", "difficulty": "easy"},
    ),
    TestCase(
        id="standard-bracket-balance",
        input_prompt="Balance the brackets and return ONLY the corrected string: ([a,b]c)",
        expected_output="([a,b]c)",
        metadata={"category": "bracket_balance", "difficulty": "easy"},
    ),
    TestCase(
        id="standard-follow-instructions",
        input_prompt="Return ONLY the word 'pong' and nothing else.",
        expected_output="pong",
        metadata={"category": "instruction_following", "difficulty": "easy"},
    ),
]


class BenchmarkRunner:
    """
    High-level orchestrator that runs benchmarks across multiple LLMs.

    This class bridges the CLI (``main.py``) to the lower-level
    :class:`Benchmarking` engine. Instead of requiring the caller to supply a
    single ``generate_function`` (as :class:`Benchmarking` does), it accepts an
    :class:`~bracketbench.llms.manager.LLMManager` containing one or more
    already-configured models, runs the configured test cases against every
    model, aggregates the results, and writes them to disk.
    """

    def __init__(
        self,
        output_dir: str = "results",
        iterations: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the benchmark runner.

        Args:
            output_dir: Directory where result files are written.
            iterations: Number of times to run the full test-case set per model.
            config: Optional configuration dictionary (reserved for future use).
        """
        self.output_dir = output_dir
        self.iterations = max(1, int(iterations))
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.BenchmarkRunner")

    def _build_test_cases(self, test_cases: Optional[List[str]] = None) -> List[TestCase]:
        """
        Build the list of TestCase objects for this run.

        Currently only the built-in ``"standard"`` set is supported. Unknown
        names fall back to the standard set so the run never fails silently on
        an empty test-case list.
        """
        names = [n.strip() for n in test_cases if n and n.strip()] if test_cases else []
        if not names or any(n.lower() == "standard" for n in names):
            return list(_STANDARD_TEST_CASES)
        # Unknown set names: fall back to standard rather than raising, so the
        # CLI keeps working. Each requested name is recorded as metadata.
        cases = []
        for name in names:
            base = _STANDARD_TEST_CASES[0]
            cases.append(
                TestCase(
                    id=f"{name}-{base.id}",
                    input_prompt=base.input_prompt,
                    expected_output=base.expected_output,
                    metadata={**base.metadata, "requested_set": name},
                )
            )
        return cases

    def run(
        self,
        llm_manager: Any,
        test_cases: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Run the benchmark suite against every model in ``llm_manager``.

        Args:
            llm_manager: An :class:`~bracketbench.llms.manager.LLMManager` with
                one or more models already added.
            test_cases: Names of test-case sets to run (e.g. ``["standard"]``).
            metrics: Optional list of metric names to collect. Currently
                informational; scoring is handled by the :class:`ScoringEngine`.

        Returns:
            Mapping of model name -> list of :class:`BenchmarkResult` objects
            aggregated across all iterations.
        """
        cases = self._build_test_cases(test_cases)
        model_names = llm_manager.list_models()
        if not model_names:
            raise ValueError("No models registered in the LLM manager.")

        all_results: Dict[str, List[BenchmarkResult]] = {}

        for model_name in model_names:
            model = llm_manager.get_model(model_name)

            def generate_function(prompt: str, _model=model, **kwargs) -> str:
                return _model.generate(prompt, **kwargs)

            benchmark = Benchmarking(
                generate_function=generate_function,
                name=model_name,
                config={"metrics": metrics} if metrics else None,
            )
            benchmark.add_test_cases(cases)

            model_results: List[BenchmarkResult] = []
            for iteration in range(self.iterations):
                benchmark.run_benchmark()
                for result in benchmark.get_results():
                    # Tag every result with the originating model + iteration so
                    # downstream analysis and saved files stay attributable.
                    result.metadata["model_name"] = model_name
                    result.metadata["iteration"] = iteration + 1
                    model_results.append(result)

            all_results[model_name] = model_results
            self.logger.info(
                "Completed %d iteration(s) for model '%s' (%d results)",
                self.iterations,
                model_name,
                len(model_results),
            )

        return all_results

    def analyze_results(
        self, results: Dict[str, List[BenchmarkResult]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute summary statistics for each model's results.

        Args:
            results: Mapping returned by :meth:`run`.

        Returns:
            Mapping of model name -> summary statistics dictionary.
        """
        handler = ResultsHandler()
        analysis: Dict[str, Dict[str, Any]] = {}
        for model_name, model_results in results.items():
            analysis[model_name] = handler._calculate_summary(model_results)
        return analysis

    def save_results(
        self,
        results: Dict[str, List[BenchmarkResult]],
        analysis: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Save per-model results and the combined analysis to ``output_dir``.

        Writes ``<output_dir>/<model_name>.json`` for each model and a single
        ``<output_dir>/analysis.json`` containing all summaries.

        Args:
            results: Mapping returned by :meth:`run`.
            analysis: Mapping returned by :meth:`analyze_results`.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        handler = ResultsHandler()

        for model_name, model_results in results.items():
            # Sanitize the model name into a safe filename component.
            safe_name = model_name.replace("/", "_").replace(":", "_")
            filepath = os.path.join(self.output_dir, f"{safe_name}.json")
            handler.save_results(model_results, filepath)

        analysis_path = os.path.join(self.output_dir, "analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        self.logger.info("Saved results for %d model(s) to %s", len(results), self.output_dir)