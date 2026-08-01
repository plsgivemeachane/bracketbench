"""
Basic usage example for BracketBench.

This example demonstrates how to set up and use the benchmarking system
with a simple mock LLM, create test cases, run benchmarks, and get results.
"""

import time
from typing import Dict, Any, Optional

from bracketbench.benchmarking import Benchmarking, TestCase
from bracketbench.llms.base import LLMInterface


class MockLLM(LLMInterface):
    """
    A simple mock LLM for demonstration purposes.
    
    This mock LLM generates responses based on simple rules
    without requiring actual API calls or credentials.
    """
    
    def __init__(self, model_name: str = "mock-model", config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the mock LLM."""
        super().__init__(model_name, config)
        self.response_delay = config.get("response_delay", 0.1) if config else 0.1
    
    def initialize(self) -> None:
        """Initialize the mock LLM."""
        self._is_initialized = True
        print(f"Mock LLM '{self.model_name}' initialized")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a mock response based on the prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens (ignored in mock)
            temperature: Temperature (ignored in mock)
            **kwargs: Additional parameters (ignored in mock)
            
        Returns:
            Generated response
        """
        if not self.is_initialized():
            self.initialize()
        
        # Simulate processing delay
        time.sleep(self.response_delay)
        
        # Simple response generation based on prompt content
        prompt_lower = prompt.lower()
        
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! I'm a mock language model. How can I help you today?"
        elif "what is" in prompt_lower or "define" in prompt_lower:
            return "This is a mock definition. In a real LLM, this would be a comprehensive answer."
        elif "translate" in prompt_lower:
            return "Translated text: [This would be the translated text in a real LLM]"
        elif "summarize" in prompt_lower:
            return "Summary: This is a brief summary of the content that would have been provided."
        elif "code" in prompt_lower or "python" in prompt_lower:
            return """def mock_function():
    # This is a mock code example
    return "Hello from mock code!"

print(mock_function())"""
        else:
            return f"This is a mock response to: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'"
    
    def generate_batch(
        self, 
        prompts: list, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> list:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens (ignored in mock)
            temperature: Temperature (ignored in mock)
            **kwargs: Additional parameters (ignored in mock)
            
        Returns:
            List of generated responses
        """
        return [self.generate(prompt, max_tokens, temperature, **kwargs) for prompt in prompts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the mock model.
        
        Returns:
            Model information dictionary
        """
        return {
            "name": self.model_name,
            "provider": "MockProvider",
            "version": "1.0.0",
            "capabilities": ["text-generation", "chat", "code-generation"],
            "parameters": {
                "response_delay": self.response_delay
            }
        }


def main():
    """
    Main function demonstrating basic usage of BracketBench.
    """
    print("=== BracketBench Basic Usage Example ===\n")
    
    # Step 1: Create a mock LLM instance
    print("Step 1: Creating a mock LLM instance...")
    mock_llm = MockLLM("mock-model-v1", {"response_delay": 0.05})
    mock_llm.initialize()
    print(f"LLM created: {mock_llm.get_model_info()['name']}\n")
    
    # Step 2: Create a benchmarking instance
    print("Step 2: Creating a benchmarking instance...")
    benchmark = Benchmarking(
        generate_function=mock_llm.generate,
        name="Basic Usage Example"
    )
    print(f"Benchmark created: {benchmark}\n")
    
    # Step 3: Create test cases
    print("Step 3: Creating test cases...")
    test_cases = [
        TestCase(
            id="greeting",
            input_prompt="Hello, how are you?",
            expected_output="Hello! I'm a mock language model. How can I help you today?"
        ),
        TestCase(
            id="question",
            input_prompt="What is artificial intelligence?",
            expected_output="This is a mock definition. In a real LLM, this would be a comprehensive answer."
        ),
        TestCase(
            id="code_request",
            input_prompt="Write a simple Python function that returns 'Hello World'",
            expected_output=None  # No expected output for this one
        ),
        TestCase(
            id="translation",
            input_prompt="Translate 'Hello world' to Spanish",
            expected_output="Translated text: [This would be the translated text in a real LLM]"
        )
    ]
    
    # Add test cases to the benchmark
    for test_case in test_cases:
        benchmark.add_test_case(test_case)
        print(f"Added test case: {test_case.id}")
    print(f"Total test cases: {len(benchmark.test_cases)}\n")
    
    # Step 4: Run the benchmark
    print("Step 4: Running the benchmark...")
    results = benchmark.run_benchmark()
    print(f"Benchmark completed with {len(results)} results\n")
    
    # Step 5: Display results
    print("Step 5: Results:")
    print("-" * 50)
    for result in results:
        print(f"Test Case: {result.test_case_id}")
        print(f"Input: {result.input_prompt}")
        print(f"Expected: {result.expected_output}")
        print(f"Actual: {result.actual_output}")
        print(f"Score: {result.score:.2f}")
        print(f"Execution Time: {result.execution_time:.3f}s")
        print("-" * 50)
    
    # Step 6: Save results
    print("\nStep 6: Saving results...")
    benchmark.save_results("basic_benchmark_results.json")
    print("Results saved to 'basic_benchmark_results.json'")
    
    # Step 7: Get summary statistics
    print("\nStep 7: Summary Statistics:")
    summary = benchmark.results_handler.get_summary(results)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()