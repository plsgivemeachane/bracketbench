"""
Test runner script for BracketBench.

This script provides a convenient way to run all tests for the BracketBench system,
with options for running specific test modules, generating reports, and more.
"""

import unittest
import sys
import os
import time
import argparse
from typing import List, Dict, Any, Optional
import json


class TestResult:
    """Simple class to store test results."""
    
    def __init__(self, module_name: str, total_tests: int, failures: int, errors: int, 
                 skipped: int, execution_time: float) -> None:
        self.module_name = module_name
        self.total_tests = total_tests
        self.failures = failures
        self.errors = errors
        self.skipped = skipped
        self.execution_time = execution_time
        self.success = (failures == 0 and errors == 0)
    
    @property
    def passed_tests(self) -> int:
        """Get number of passed tests."""
        return self.total_tests - self.failures - self.errors - self.skipped
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "module_name": self.module_name,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failures": self.failures,
            "errors": self.errors,
            "skipped": self.skipped,
            "success": self.success,
            "execution_time": self.execution_time,
            "success_rate": self.passed_tests / self.total_tests if self.total_tests > 0 else 0
        }


class BracketBenchTestRunner:
    """Test runner for BracketBench tests."""
    
    def __init__(self) -> None:
        """Initialize the test runner."""
        self.test_modules = [
            "test_benchmarking",
            "test_llms"
        ]
        self.results: List[TestResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def discover_tests(self, test_dir: str = "tests") -> List[str]:
        """
        Discover test modules in the specified directory.
        
        Args:
            test_dir: Directory to search for test modules
            
        Returns:
            List of test module names
        """
        test_modules = []
        
        if not os.path.exists(test_dir):
            print(f"Warning: Test directory '{test_dir}' not found")
            return test_modules
        
        for file in os.listdir(test_dir):
            if file.startswith("test_") and file.endswith(".py"):
                module_name = file[:-3]  # Remove .py extension
                test_modules.append(module_name)
        
        return sorted(test_modules)
    
    def run_single_module(self, module_name: str, verbose: bool = False) -> TestResult:
        """
        Run tests for a single module.
        
        Args:
            module_name: Name of the test module to run
            verbose: Whether to show verbose output
            
        Returns:
            TestResult object with test statistics
        """
        start_time = time.time()
        
        # Import the test module
        try:
            module = __import__(module_name)
        except ImportError as e:
            print(f"Error importing test module '{module_name}': {e}")
            return TestResult(module_name, 0, 0, 1, 0, 0.0)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = runner.run(suite)
        
        execution_time = time.time() - start_time
        
        return TestResult(
            module_name=module_name,
            total_tests=result.testsRun,
            failures=len(result.failures),
            errors=len(result.errors),
            skipped=len(result.skipped) if hasattr(result, 'skipped') else 0,
            execution_time=execution_time
        )
    
    def run_all_tests(self, verbose: bool = False) -> None:
        """
        Run all test modules.
        
        Args:
            verbose: Whether to show verbose output
        """
        self.start_time = time.time()
        self.results = []
        
        print("=" * 70)
        print("Running BracketBench Test Suite")
        print("=" * 70)
        
        for module_name in self.test_modules:
            print(f"\nRunning {module_name}...")
            print("-" * 50)
            
            result = self.run_single_module(module_name, verbose)
            self.results.append(result)
            
            # Print summary for this module
            status = "✓ PASS" if result.success else "✗ FAIL"
            print(f"\n{status}: {module_name} "
                  f"({result.passed_tests}/{result.total_tests} passed, "
                  f"{result.execution_time:.2f}s)")
            
            if result.failures > 0:
                print(f"  Failures: {result.failures}")
            if result.errors > 0:
                print(f"  Errors: {result.errors}")
            if result.skipped > 0:
                print(f"  Skipped: {result.skipped}")
        
        self.end_time = time.time()
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a test report.
        
        Args:
            output_file: Optional file to save the report to
            
        Returns:
            Report dictionary
        """
        if not self.results:
            return {"error": "No test results available"}
        
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed_tests for r in self.results)
        total_failures = sum(r.failures for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_time = sum(r.execution_time for r in self.results)
        
        overall_success = total_failures == 0 and total_errors == 0
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_execution_time": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "failed_tests": total_failures,
                "errors": total_errors,
                "skipped_tests": total_skipped,
                "success_rate": total_passed / total_tests if total_tests > 0 else 0,
                "overall_success": overall_success,
                "total_time": total_time
            },
            "module_results": [result.to_dict() for result in self.results]
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nTest report saved to: {output_file}")
        
        return report
    
    def print_summary(self) -> None:
        """Print a summary of test results."""
        if not self.results:
            print("No test results available")
            return
        
        print("\n" + "=" * 70)
        print("Test Suite Summary")
        print("=" * 70)
        
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed_tests for r in self.results)
        total_failures = sum(r.failures for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_time = sum(r.execution_time for r in self.results)
        
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        overall_success = total_failures == 0 and total_errors == 0
        
        # Overall status
        status = "✓ ALL TESTS PASSED" if overall_success else "✗ SOME TESTS FAILED"
        print(f"\n{status}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Skipped: {total_skipped}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Total Time: {total_time:.2f}s")
        
        # Module breakdown
        print("\nModule Breakdown:")
        print("-" * 50)
        for result in self.results:
            status = "✓" if result.success else "✗"
            print(f"{status} {result.module_name}: "
                  f"{result.passed_tests}/{result.total_tests} "
                  f"({result.execution_time:.2f}s)")
        
        print("\n" + "=" * 70)


def main():
    """Main function to run tests from command line."""
    parser = argparse.ArgumentParser(description="Run BracketBench tests")
    parser.add_argument(
        "--module", "-m", 
        help="Specific test module to run (e.g., test_benchmarking)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Show verbose test output"
    )
    parser.add_argument(
        "--report", "-r",
        help="Save test report to specified JSON file"
    )
    parser.add_argument(
        "--discover", "-d",
        action="store_true",
        help="Discover and list available test modules"
    )
    parser.add_argument(
        "--test-dir", "-t",
        default="tests",
        help="Directory containing test modules (default: tests)"
    )
    
    args = parser.parse_args()
    
    # Add test directory to Python path
    if args.test_dir not in sys.path:
        sys.path.insert(0, args.test_dir)
    
    # Create test runner
    runner = BracketBenchTestRunner()
    
    # Discover test modules if requested
    if args.discover:
        modules = runner.discover_tests(args.test_dir)
        print(f"Discovered test modules in '{args.test_dir}':")
        for module in modules:
            print(f"  - {module}")
        return
    
    # Update test modules if custom directory is specified
    if args.test_dir != "tests":
        runner.test_modules = runner.discover_tests(args.test_dir)
    
    # Run specific module if requested
    if args.module:
        if args.module not in runner.test_modules:
            print(f"Error: Test module '{args.module}' not found")
            print(f"Available modules: {runner.test_modules}")
            return
        
        print(f"Running single test module: {args.module}")
        result = runner.run_single_module(args.module, args.verbose)
        runner.results = [result]
        
        # Print single module summary
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"\n{status}: {args.module} "
              f"({result.passed_tests}/{result.total_tests} passed, "
              f"{result.execution_time:.2f}s)")
    else:
        # Run all tests
        runner.run_all_tests(args.verbose)
    
    # Print summary
    runner.print_summary()
    
    # Generate report if requested
    if args.report:
        runner.generate_report(args.report)


if __name__ == "__main__":
    main()