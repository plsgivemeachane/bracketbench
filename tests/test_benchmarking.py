"""
Unit tests for the benchmarking functionality in BracketBench.

This module contains comprehensive unit tests for the core benchmarking classes,
including Benchmarking, TestCase, ScoringEngine, and ResultsHandler.
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from bracketbench.benchmarking import (
    Benchmarking, 
    TestCase, 
    ScoringEngine, 
    ResultsHandler, 
    BenchmarkResult
)


class TestTestCase(unittest.TestCase):
    """Test cases for the TestCase class."""
    
    def test_test_case_creation(self) -> None:
        """Test creating a TestCase with all parameters."""
        test_case = TestCase(
            id="test-1",
            input_prompt="What is AI?",
            expected_output="Artificial Intelligence",
            metadata={"category": "definition", "difficulty": "easy"}
        )
        
        self.assertEqual(test_case.id, "test-1")
        self.assertEqual(test_case.input_prompt, "What is AI?")
        self.assertEqual(test_case.expected_output, "Artificial Intelligence")
        self.assertEqual(test_case.metadata, {"category": "definition", "difficulty": "easy"})
    
    def test_test_case_auto_id_generation(self) -> None:
        """Test that TestCase generates ID automatically if not provided."""
        test_case = TestCase(
            id="",
            input_prompt="What is AI?"
        )
        
        # ID should be generated and not empty
        self.assertTrue(test_case.id)
        self.assertNotEqual(test_case.id, "")
    
    def test_test_case_validation(self) -> None:
        """Test TestCase validation."""
        # Valid test case
        test_case = TestCase(
            id="test-1",
            input_prompt="What is AI?",
            expected_output="Artificial Intelligence"
        )
        self.assertTrue(test_case.validate())
        
        # Missing ID
        with self.assertRaises(ValueError):
            TestCase(id="", input_prompt="What is AI?").validate()
        
        # Missing input prompt
        with self.assertRaises(ValueError):
            TestCase(id="test-1", input_prompt="").validate()
        
        # Invalid input prompt type
        with self.assertRaises(ValueError):
            TestCase(id="test-1", input_prompt=123).validate()
        
        # Invalid expected output type
        with self.assertRaises(ValueError):
            TestCase(id="test-1", input_prompt="What is AI?", expected_output=123).validate()
        
        # Invalid metadata type
        with self.assertRaises(ValueError):
            TestCase(id="test-1", input_prompt="What is AI?", metadata="invalid").validate()
    
    def test_test_case_to_dict(self) -> None:
        """Test converting TestCase to dictionary."""
        test_case = TestCase(
            id="test-1",
            input_prompt="What is AI?",
            expected_output="Artificial Intelligence",
            metadata={"category": "definition"}
        )
        
        result = test_case.to_dict()
        expected = {
            "id": "test-1",
            "input_prompt": "What is AI?",
            "expected_output": "Artificial Intelligence",
            "metadata": {"category": "definition"}
        }
        
        self.assertEqual(result, expected)
    
    def test_test_case_from_dict(self) -> None:
        """Test creating TestCase from dictionary."""
        data = {
            "id": "test-1",
            "input_prompt": "What is AI?",
            "expected_output": "Artificial Intelligence",
            "metadata": {"category": "definition"}
        }
        
        test_case = TestCase.from_dict(data)
        self.assertEqual(test_case.id, "test-1")
        self.assertEqual(test_case.input_prompt, "What is AI?")
        self.assertEqual(test_case.expected_output, "Artificial Intelligence")
        self.assertEqual(test_case.metadata, {"category": "definition"})
    
    def test_test_case_metadata_operations(self) -> None:
        """Test metadata operations on TestCase."""
        test_case = TestCase(
            id="test-1",
            input_prompt="What is AI?",
            metadata={"category": "definition"}
        )
        
        # Update metadata
        test_case.update_metadata("difficulty", "easy")
        self.assertEqual(test_case.metadata, {"category": "definition", "difficulty": "easy"})
        
        # Remove metadata
        result = test_case.remove_metadata("category")
        self.assertTrue(result)
        self.assertEqual(test_case.metadata, {"difficulty": "easy"})
        
        # Remove non-existent metadata
        result = test_case.remove_metadata("nonexistent")
        self.assertFalse(result)
        self.assertEqual(test_case.metadata, {"difficulty": "easy"})


class TestScoringEngine(unittest.TestCase):
    """Test cases for the ScoringEngine class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.scoring_engine = ScoringEngine()
    
    def test_initialization(self) -> None:
        """Test ScoringEngine initialization."""
        self.assertIn("exact_match", self.scoring_engine.scoring_methods)
        self.assertIn("similarity", self.scoring_engine.scoring_methods)
        self.assertIn("length_ratio", self.scoring_engine.scoring_methods)
    
    def test_exact_match_score(self) -> None:
        """Test exact match scoring."""
        # Exact match
        score = self.scoring_engine.score_output(
            "Hello world", 
            "Hello world", 
            method="exact_match"
        )
        self.assertEqual(score, 1.0)
        
        # No match
        score = self.scoring_engine.score_output(
            "Hello world", 
            "Goodbye world", 
            method="exact_match"
        )
        self.assertEqual(score, 0.0)
        
        # Whitespace difference
        score = self.scoring_engine.score_output(
            "Hello world", 
            "Hello world  ", 
            method="exact_match"
        )
        self.assertEqual(score, 1.0)
    
    def test_similarity_score(self) -> None:
        """Test similarity scoring."""
        # Identical strings
        score = self.scoring_engine.score_output(
            "Hello world", 
            "Hello world", 
            method="similarity"
        )
        self.assertEqual(score, 1.0)
        
        # Similar strings
        score = self.scoring_engine.score_output(
            "Hello world", 
            "Hello there", 
            method="similarity"
        )
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)
        
        # Different strings
        score = self.scoring_engine.score_output(
            "Hello world", 
            "Goodbye moon", 
            method="similarity"
        )
        self.assertLess(score, 0.5)
        
        # No expected output
        score = self.scoring_engine.score_output(
            "Hello world", 
            None, 
            method="similarity"
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_length_ratio_score(self) -> None:
        """Test length ratio scoring."""
        # Same length
        score = self.scoring_engine.score_output(
            "Hello world", 
            "Hello world", 
            method="length_ratio"
        )
        self.assertEqual(score, 1.0)
        
        # Different lengths
        score = self.scoring_engine.score_output(
            "Hello", 
            "Hello world", 
            method="length_ratio"
        )
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)
        
        # No expected output
        score = self.scoring_engine.score_output(
            "Hello world", 
            None, 
            method="length_ratio"
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_invalid_scoring_method(self) -> None:
        """Test using invalid scoring method."""
        with self.assertRaises(ValueError):
            self.scoring_engine.score_output(
                "Hello world", 
                "Hello world", 
                method="invalid_method"
            )
    
    def test_register_custom_scoring_method(self) -> None:
        """Test registering a custom scoring method."""
        def custom_scorer(actual: str, expected: Optional[str] = None, prompt: Optional[str] = None) -> float:
            return 0.5
        
        self.scoring_engine.register_scoring_method("custom", custom_scorer)
        
        # Should be available now
        score = self.scoring_engine.score_output(
            "Hello world", 
            "Hello world", 
            method="custom"
        )
        self.assertEqual(score, 0.5)
        
        # Should be in available methods
        self.assertIn("custom", self.scoring_engine.get_available_methods())
    
    def test_get_available_methods(self) -> None:
        """Test getting available scoring methods."""
        methods = self.scoring_engine.get_available_methods()
        self.assertIsInstance(methods, list)
        self.assertIn("exact_match", methods)
        self.assertIn("similarity", methods)
        self.assertIn("length_ratio", methods)


class TestResultsHandler(unittest.TestCase):
    """Test cases for the ResultsHandler class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.results_handler = ResultsHandler()
        self.sample_results = [
            BenchmarkResult(
                test_case_id="test-1",
                input_prompt="What is AI?",
                expected_output="Artificial Intelligence",
                actual_output="AI is artificial intelligence",
                execution_time=0.5,
                score=0.8,
                metadata={"category": "definition"}
            ),
            BenchmarkResult(
                test_case_id="test-2",
                input_prompt="Write code",
                expected_output=None,
                actual_output="def hello(): pass",
                execution_time=0.3,
                score=0.9,
                metadata={"category": "code"}
            )
        ]
    
    def test_save_and_load_json(self) -> None:
        """Test saving and loading results as JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Save results
            self.results_handler.save_results(self.sample_results, filepath)
            
            # Load results
            loaded_results = self.results_handler.load_results(filepath)
            
            # Verify results
            self.assertEqual(len(loaded_results), len(self.sample_results))
            
            for i, (original, loaded) in enumerate(zip(self.sample_results, loaded_results)):
                self.assertEqual(original.test_case_id, loaded.test_case_id)
                self.assertEqual(original.input_prompt, loaded.input_prompt)
                self.assertEqual(original.expected_output, loaded.expected_output)
                self.assertEqual(original.actual_output, loaded.actual_output)
                self.assertEqual(original.execution_time, loaded.execution_time)
                self.assertEqual(original.score, loaded.score)
                self.assertEqual(original.metadata, loaded.metadata)
                
        finally:
            os.unlink(filepath)
    
    def test_save_and_load_csv(self) -> None:
        """Test saving and loading results as CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            # Save results
            self.results_handler.save_results(self.sample_results, filepath)
            
            # Load results
            loaded_results = self.results_handler.load_results(filepath)
            
            # Verify results
            self.assertEqual(len(loaded_results), len(self.sample_results))
            
            for i, (original, loaded) in enumerate(zip(self.sample_results, loaded_results)):
                self.assertEqual(original.test_case_id, loaded.test_case_id)
                self.assertEqual(original.input_prompt, loaded.input_prompt)
                self.assertEqual(original.expected_output, loaded.expected_output)
                self.assertEqual(original.actual_output, loaded.actual_output)
                self.assertEqual(original.execution_time, loaded.execution_time)
                self.assertEqual(original.score, loaded.score)
                self.assertEqual(original.metadata, loaded.metadata)
                
        finally:
            os.unlink(filepath)
    
    def test_calculate_summary(self) -> None:
        """Test summary calculation."""
        summary = self.results_handler.get_summary(self.sample_results)
        
        self.assertEqual(summary["total_tests"], 2)
        self.assertEqual(summary["total_score"], 1.7)  # 0.8 + 0.9
        self.assertEqual(summary["average_score"], 0.85)
        self.assertEqual(summary["total_execution_time"], 0.8)  # 0.5 + 0.3
        self.assertEqual(summary["average_execution_time"], 0.4)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["success_rate"], 1.0)
    
    def test_save_empty_results(self) -> None:
        """Test saving empty results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.results_handler.save_results([], filepath)
        finally:
            os.unlink(filepath)
    
    def test_load_invalid_file(self) -> None:
        """Test loading from invalid file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
            f.write("invalid json content")
        
        try:
            with self.assertRaises(Exception):
                self.results_handler.load_results(filepath)
        finally:
            os.unlink(filepath)


class TestBenchmarking(unittest.TestCase):
    """Test cases for the Benchmarking class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_generate_function = Mock(return_value="Mock response")
        self.benchmark = Benchmarking(
            generate_function=self.mock_generate_function,
            name="Test Benchmark"
        )
    
    def test_initialization(self) -> None:
        """Test Benchmarking initialization."""
        self.assertEqual(self.benchmark.name, "Test Benchmark")
        self.assertEqual(self.benchmark.generate_function, self.mock_generate_function)
        self.assertIsInstance(self.benchmark.scoring_engine, ScoringEngine)
        self.assertIsInstance(self.benchmark.results_handler, ResultsHandler)
        self.assertEqual(len(self.benchmark.test_cases), 0)
        self.assertEqual(len(self.benchmark.results), 0)
    
    def test_add_test_case(self) -> None:
        """Test adding a test case."""
        test_case = TestCase(
            id="test-1",
            input_prompt="What is AI?"
        )
        
        self.benchmark.add_test_case(test_case)
        self.assertEqual(len(self.benchmark.test_cases), 1)
        self.assertEqual(self.benchmark.test_cases[0], test_case)
    
    def test_add_invalid_test_case(self) -> None:
        """Test adding invalid test case."""
        with self.assertRaises(TypeError):
            self.benchmark.add_test_case("not a test case")
    
    def test_add_test_cases(self) -> None:
        """Test adding multiple test cases."""
        test_cases = [
            TestCase(id="test-1", input_prompt="What is AI?"),
            TestCase(id="test-2", input_prompt="Write code")
        ]
        
        self.benchmark.add_test_cases(test_cases)
        self.assertEqual(len(self.benchmark.test_cases), 2)
    
    def test_run_benchmark_no_test_cases(self) -> None:
        """Test running benchmark with no test cases."""
        with self.assertRaises(ValueError):
            self.benchmark.run_benchmark()
    
    def test_run_benchmark_success(self) -> None:
        """Test successful benchmark run."""
        # Add test cases
        test_cases = [
            TestCase(
                id="test-1",
                input_prompt="What is AI?",
                expected_output="Artificial Intelligence"
            ),
            TestCase(
                id="test-2",
                input_prompt="Write code",
                expected_output=None
            )
        ]
        
        self.benchmark.add_test_cases(test_cases)
        
        # Run benchmark
        results = self.benchmark.run_benchmark()
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(len(self.benchmark.results), 2)
        
        # Verify mock function was called
        self.assertEqual(self.mock_generate_function.call_count, 2)
        
        # Verify result structure
        for result in results:
            self.assertIsInstance(result, BenchmarkResult)
            self.assertIn(result.test_case_id, ["test-1", "test-2"])
            self.assertIsNotNone(result.actual_output)
            self.assertGreater(result.execution_time, 0)
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)
    
    def test_run_benchmark_with_error(self) -> None:
        """Test benchmark run with generation error."""
        # Make mock function raise an exception
        self.mock_generate_function.side_effect = RuntimeError("Generation failed")
        
        # Add test case
        test_case = TestCase(
            id="test-1",
            input_prompt="What is AI?",
            expected_output="Artificial Intelligence"
        )
        
        self.benchmark.add_test_case(test_case)
        
        # Run benchmark
        results = self.benchmark.run_benchmark()
        
        # Should still return a result, but with error in metadata
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIn("error", result.metadata)
        self.assertEqual(result.metadata["error"], "Generation failed")
    
    def test_get_results(self) -> None:
        """Test getting results."""
        # Add test case and run benchmark
        test_case = TestCase(id="test-1", input_prompt="What is AI?")
        self.benchmark.add_test_case(test_case)
        self.benchmark.run_benchmark()
        
        # Get results
        results = self.benchmark.get_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].test_case_id, "test-1")
    
    def test_get_result_by_test_case_id(self) -> None:
        """Test getting result by test case ID."""
        # Add test case and run benchmark
        test_case = TestCase(id="test-1", input_prompt="What is AI?")
        self.benchmark.add_test_case(test_case)
        self.benchmark.run_benchmark()
        
        # Get result by ID
        result = self.benchmark.get_result_by_test_case_id("test-1")
        self.assertIsNotNone(result)
        self.assertEqual(result.test_case_id, "test-1")
        
        # Get non-existent result
        result = self.benchmark.get_result_by_test_case_id("non-existent")
        self.assertIsNone(result)
    
    def test_save_results(self) -> None:
        """Test saving results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Add test case and run benchmark
            test_case = TestCase(id="test-1", input_prompt="What is AI?")
            self.benchmark.add_test_case(test_case)
            self.benchmark.run_benchmark()
            
            # Save results
            self.benchmark.save_results(filepath)
            
            # Verify file exists and contains data
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.assertIn("results", data)
                self.assertEqual(len(data["results"]), 1)
                
        finally:
            os.unlink(filepath)
    
    def test_clear_results(self) -> None:
        """Test clearing results."""
        # Add test case and run benchmark
        test_case = TestCase(id="test-1", input_prompt="What is AI?")
        self.benchmark.add_test_case(test_case)
        self.benchmark.run_benchmark()
        
        # Verify results exist
        self.assertEqual(len(self.benchmark.results), 1)
        
        # Clear results
        self.benchmark.clear_results()
        self.assertEqual(len(self.benchmark.results), 0)
    
    def test_clear_test_cases(self) -> None:
        """Test clearing test cases."""
        # Add test case
        test_case = TestCase(id="test-1", input_prompt="What is AI?")
        self.benchmark.add_test_case(test_case)
        
        # Verify test case exists
        self.assertEqual(len(self.benchmark.test_cases), 1)
        
        # Clear test cases
        self.benchmark.clear_test_cases()
        self.assertEqual(len(self.benchmark.test_cases), 0)
    
    def test_reset(self) -> None:
        """Test resetting benchmark."""
        # Add test case and run benchmark
        test_case = TestCase(id="test-1", input_prompt="What is AI?")
        self.benchmark.add_test_case(test_case)
        self.benchmark.run_benchmark()
        
        # Verify data exists
        self.assertEqual(len(self.benchmark.test_cases), 1)
        self.assertEqual(len(self.benchmark.results), 1)
        
        # Reset
        self.benchmark.reset()
        
        # Verify data is cleared
        self.assertEqual(len(self.benchmark.test_cases), 0)
        self.assertEqual(len(self.benchmark.results), 0)
    
    def test_string_representation(self) -> None:
        """Test string representation."""
        # Add test case
        test_case = TestCase(id="test-1", input_prompt="What is AI?")
        self.benchmark.add_test_case(test_case)
        
        str_repr = str(self.benchmark)
        self.assertIn("Test Benchmark", str_repr)
        self.assertIn("1", str_repr)  # Number of test cases
        
        repr_str = repr(self.benchmark)
        self.assertIn("Test Benchmark", repr_str)
        self.assertIn("1", repr_str)  # Number of test cases


if __name__ == "__main__":
    unittest.main()