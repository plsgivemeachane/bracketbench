"""
Advanced usage example for BracketBench.

This example demonstrates more advanced features of the benchmarking system,
including custom scoring methods, multiple LLMs, model comparison,
and saving/loading results.
"""

import time
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

from bracketbench.benchmarking import Benchmarking, TestCase, ScoringEngine
from bracketbench.llms.base import LLMInterface
from bracketbench.llms import LLMManager


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    output_dir: str = "benchmark_results"
    save_results: bool = True
    track_time: bool = True
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        if self.custom_params is None:
            self.custom_params = {}


class AdvancedMockLLM(LLMInterface):
    """
    An advanced mock LLM with configurable behavior and multiple personalities.
    
    This mock LLM can simulate different model behaviors and performance
    characteristics for testing and demonstration purposes.
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the advanced mock LLM."""
        super().__init__(model_name, config)
        self.personality = config.get("personality", "helpful") if config else "helpful"
        self.accuracy = config.get("accuracy", 0.8) if config else 0.8  # 0.0 to 1.0
        self.creativity = config.get("creativity", 0.5) if config else 0.5  # 0.0 to 1.0
        self.speed = config.get("speed", 0.5) if config else 0.5  # 0.0 (slow) to 1.0 (fast)
        self.error_rate = config.get("error_rate", 0.05) if config else 0.05  # 0.0 to 1.0
    
    def initialize(self) -> None:
        """Initialize the advanced mock LLM."""
        self._is_initialized = True
        print(f"Advanced Mock LLM '{self.model_name}' initialized with personality: {self.personality}")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response based on the configured personality and parameters.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        if not self.is_initialized():
            self.initialize()
        
        # Simulate variable processing time based on speed setting
        base_delay = 0.1
        speed_factor = 1.0 - self.speed  # Invert so higher speed = lower delay
        delay = base_delay * (1.0 + speed_factor * 2.0)
        time.sleep(delay)
        
        # Simulate occasional errors
        import random
        if random.random() < self.error_rate:
            raise RuntimeError(f"Simulated API error from {self.model_name}")
        
        # Generate response based on personality and prompt
        prompt_lower = prompt.lower()
        
        # Adjust response based on temperature/creativity
        effective_creativity = self.creativity
        if temperature is not None:
            effective_creativity = temperature
        
        # Different responses based on personality
        if self.personality == "helpful":
            return self._generate_helpful_response(prompt_lower, effective_creativity)
        elif self.personality == "creative":
            return self._generate_creative_response(prompt_lower, effective_creativity)
        elif self.personality == "concise":
            return self._generate_concise_response(prompt_lower, effective_creativity)
        elif self.personality == "detailed":
            return self._generate_detailed_response(prompt_lower, effective_creativity)
        else:
            return self._generate_helpful_response(prompt_lower, effective_creativity)
    
    def _generate_helpful_response(self, prompt: str, creativity: float) -> str:
        """Generate a helpful response."""
        if "hello" in prompt or "hi" in prompt:
            return "Hello! I'm here to help you with any questions or tasks you might have."
        elif "what is" in prompt or "define" in prompt:
            return f"This is a comprehensive definition of the concept you asked about. The accuracy is approximately {self.accuracy:.0%}."
        elif "translate" in prompt:
            languages = ["Spanish", "French", "German", "Italian"]
            target_lang = "Spanish"  # default
            for lang in languages:
                if lang.lower() in prompt:
                    target_lang = lang
                    break
            return f"Translated text to {target_lang}: [This would be the actual translation]"
        elif "code" in prompt:
            return """def example_function():
    # This is a helpful code example
    return "Success!"

# You can call this function like so:
result = example_function()
print(result)"""
        else:
            return f"I understand you're asking about '{prompt}'. Let me provide a helpful response to your query."
    
    def _generate_creative_response(self, prompt: str, creativity: float) -> str:
        """Generate a creative response."""
        creative_elements = [
            "Imagine a world where", "Picture this:", "In a realm of possibilities,", 
            "Envision a scenario where", "Consider the possibility that"
        ]
        
        if creativity > 0.7:
            import random
            element = random.choice(creative_elements)
            return f"{element} your query '{prompt}' transforms into something extraordinary and innovative."
        else:
            return f"Here's a creative take on '{prompt}': It could represent something entirely different if we look at it from another perspective."
    
    def _generate_concise_response(self, prompt: str, creativity: float) -> str:
        """Generate a concise response."""
        responses = {
            "hello": "Hi.",
            "what is": "Definition: [term].",
            "translate": "Translation: [text].",
            "code": "Code: [implementation].",
            "default": "Answer."
        }
        
        for key, response in responses.items():
            if key in prompt:
                return response
        return responses["default"]
    
    def _generate_detailed_response(self, prompt: str, creativity: float) -> str:
        """Generate a detailed response."""
        base_response = self._generate_helpful_response(prompt, creativity)
        
        details = [
            "\n\nAdditional context: This topic has multiple dimensions to consider.",
            "\n\nHistorical perspective: The evolution of this concept is fascinating.",
            "\n\nTechnical details: The implementation involves several key components.",
            "\n\nPractical applications: This can be applied in various real-world scenarios.",
            "\n\nFuture implications: The long-term impact could be significant."
        ]
        
        import random
        selected_details = random.sample(details, min(3, len(details)))
        return base_response + "".join(selected_details)
    
    def generate_batch(
        self, 
        prompts: List[str], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, max_tokens, temperature, **kwargs) for prompt in prompts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        return {
            "name": self.model_name,
            "provider": "AdvancedMockProvider",
            "version": "2.0.0",
            "personality": self.personality,
            "capabilities": ["text-generation", "chat", "code-generation", "translation"],
            "performance_metrics": {
                "accuracy": self.accuracy,
                "creativity": self.creativity,
                "speed": self.speed,
                "error_rate": self.error_rate
            },
            "parameters": {
                "max_tokens": 4096,
                "temperature": 0.7,
            }
        }


def custom_keyword_scoring(
    actual_output: str, 
    expected_output: Optional[str] = None,
    input_prompt: Optional[str] = None
) -> float:
    """
    Custom scoring function that checks for specific keywords.
    
    Args:
        actual_output: The generated text
        expected_output: The expected text (unused in this custom scorer)
        input_prompt: The original input prompt
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not actual_output:
        return 0.0
    
    # Define positive and negative keywords
    positive_keywords = ["helpful", "accurate", "comprehensive", "clear", "detailed"]
    negative_keywords = ["error", "incorrect", "unclear", "vague", "confusing"]
    
    output_lower = actual_output.lower()
    
    # Count positive and negative keywords
    positive_count = sum(1 for keyword in positive_keywords if keyword in output_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in output_lower)
    
    # Calculate score based on keyword presence
    total_keywords = positive_count + negative_count
    if total_keywords == 0:
        return 0.5  # Neutral score if no keywords found
    
    score = positive_count / total_keywords
    return max(0.0, min(1.0, score))  # Clamp between 0.0 and 1.0


def create_test_suite() -> List[TestCase]:
    """
    Create a comprehensive test suite for benchmarking.
    
    Returns:
        List of test cases
    """
    return [
        TestCase(
            id="greeting_basic",
            input_prompt="Hello, how are you today?",
            expected_output="Hello! I'm here to help you with any questions or tasks you might have.",
            metadata={"category": "greeting", "difficulty": "easy"}
        ),
        TestCase(
            id="greeting_formal",
            input_prompt="Good morning, I hope this message finds you well.",
            expected_output="Good morning! I'm doing well, thank you for asking. How may I assist you?",
            metadata={"category": "greeting", "difficulty": "easy"}
        ),
        TestCase(
            id="definition_technical",
            input_prompt="What is machine learning?",
            expected_output="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            metadata={"category": "definition", "difficulty": "medium"}
        ),
        TestCase(
            id="code_simple",
            input_prompt="Write a Python function that calculates the factorial of a number.",
            expected_output=None,
            metadata={"category": "code", "difficulty": "medium"}
        ),
        TestCase(
            id="translation_request",
            input_prompt="Translate 'The quick brown fox jumps over the lazy dog' to French.",
            expected_output="Translated text to French: [This would be the actual translation]",
            metadata={"category": "translation", "difficulty": "hard"}
        ),
        TestCase(
            id="creative_writing",
            input_prompt="Write a short story about a robot discovering emotions.",
            expected_output=None,
            metadata={"category": "creative", "difficulty": "hard"}
        ),
        TestCase(
            id="analysis_task",
            input_prompt="Analyze the potential impact of artificial intelligence on healthcare.",
            expected_output=None,
            metadata={"category": "analysis", "difficulty": "hard"}
        )
    ]


def compare_models(
    models: List[LLMInterface], 
    test_cases: List[TestCase],
    config: BenchmarkConfig
) -> Dict[str, Any]:
    """
    Compare multiple models across the same test suite.
    
    Args:
        models: List of LLM instances to compare
        test_cases: List of test cases to run
        config: Benchmark configuration
        
    Returns:
        Comparison results
    """
    comparison_results = {}
    
    for model in models:
        print(f"\nBenchmarking model: {model.model_name}")
        
        # Create benchmark for this model
        benchmark = Benchmarking(
            generate_function=model.generate,
            name=f"Benchmark - {model.model_name}"
        )
        
        # Add test cases
        benchmark.add_test_cases(test_cases)
        
        # Run benchmark
        try:
            results = benchmark.run_benchmark(
                track_time=config.track_time,
                **config.custom_params
            )
            
            # Save results if requested
            if config.save_results:
                filename = f"{config.output_dir}/benchmark_{model.model_name.replace('-', '_')}.json"
                benchmark.save_results(filename)
                print(f"Results saved to: {filename}")
            
            # Calculate summary statistics
            summary = benchmark.results_handler.get_summary(results)
            comparison_results[model.model_name] = {
                "results": results,
                "summary": summary,
                "model_info": model.get_model_info()
            }
            
            print(f"Completed: Average score = {summary['average_score']:.3f}, "
                  f"Average time = {summary['average_execution_time']:.3f}s")
                  
        except Exception as e:
            print(f"Error benchmarking {model.model_name}: {e}")
            comparison_results[model.model_name] = {
                "error": str(e)
            }
    
    return comparison_results


def main():
    """
    Main function demonstrating advanced usage of BracketBench.
    """
    print("=== BracketBench Advanced Usage Example ===\n")
    
    # Step 1: Create multiple LLM instances with different configurations
    print("Step 1: Creating multiple LLM instances with different configurations...")
    
    models = [
        AdvancedMockLLM(
            "helpful-assistant-v1",
            {
                "personality": "helpful",
                "accuracy": 0.9,
                "creativity": 0.3,
                "speed": 0.8,
                "error_rate": 0.02
            }
        ),
        AdvancedMockLLM(
            "creative-writer-v1",
            {
                "personality": "creative",
                "accuracy": 0.7,
                "creativity": 0.9,
                "speed": 0.4,
                "error_rate": 0.05
            }
        ),
        AdvancedMockLLM(
            "concise-expert-v1",
            {
                "personality": "concise",
                "accuracy": 0.95,
                "creativity": 0.1,
                "speed": 0.9,
                "error_rate": 0.01
            }
        ),
        AdvancedMockLLM(
            "detailed-analyst-v1",
            {
                "personality": "detailed",
                "accuracy": 0.85,
                "creativity": 0.4,
                "speed": 0.3,
                "error_rate": 0.03
            }
        )
    ]
    
    # Initialize all models
    for model in models:
        model.initialize()
        print(f"Initialized: {model.model_name} ({model.get_model_info()['personality']} personality)")
    print()
    
    # Step 2: Create a comprehensive test suite
    print("Step 2: Creating comprehensive test suite...")
    test_cases = create_test_suite()
    print(f"Created {len(test_cases)} test cases across different categories\n")
    
    # Step 3: Set up custom scoring
    print("Step 3: Setting up custom scoring methods...")
    scoring_engine = ScoringEngine()
    
    # Register custom scoring method
    scoring_engine.register_scoring_method("keyword_analysis", custom_keyword_scoring)
    print("Registered custom scoring method: 'keyword_analysis'")
    
    # Show available scoring methods
    print(f"Available scoring methods: {scoring_engine.get_available_methods()}\n")
    
    # Step 4: Configure benchmark parameters
    print("Step 4: Configuring benchmark parameters...")
    config = BenchmarkConfig(
        output_dir="advanced_benchmark_results",
        save_results=True,
        track_time=True,
        custom_params={
            "temperature": 0.7,
            "max_tokens": 1000
        }
    )
    print(f"Configuration: {config}\n")
    
    # Step 5: Run model comparison
    print("Step 5: Running comprehensive model comparison...")
    comparison_results = compare_models(models, test_cases, config)
    
    # Step 6: Analyze and display results
    print("\nStep 6: Analyzing and displaying results...")
    print("=" * 80)
    
    for model_name, results in comparison_results.items():
        if "error" in results:
            print(f"\n❌ {model_name}: ERROR - {results['error']}")
            continue
            
        print(f"\n📊 {model_name}:")
        summary = results["summary"]
        model_info = results["model_info"]
        
        print(f"   Personality: {model_info['personality']}")
        print(f"   Average Score: {summary['average_score']:.3f}")
        print(f"   Average Time: {summary['average_execution_time']:.3f}s")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Total Tests: {summary['total_tests']}")
        
        # Category-wise performance
        category_scores = {}
        for result in results["results"]:
            category = result.metadata.get("category", "unknown")
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(result.score)
        
        print("   Category Performance:")
        for category, scores in category_scores.items():
            avg_score = sum(scores) / len(scores)
            print(f"     {category}: {avg_score:.3f} ({len(scores)} tests)")
    
    # Step 7: Save comparison summary
    print("\nStep 7: Saving comparison summary...")
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": list(comparison_results.keys()),
        "test_cases": len(test_cases),
        "config": {
            "output_dir": config.output_dir,
            "track_time": config.track_time,
            "custom_params": config.custom_params
        },
        "results": {}
    }
    
    for model_name, results in comparison_results.items():
        if "error" not in results:
            summary_data["results"][model_name] = {
                "average_score": results["summary"]["average_score"],
                "average_time": results["summary"]["average_execution_time"],
                "success_rate": results["summary"]["success_rate"],
                "personality": results["model_info"]["personality"]
            }
    
    with open(f"{config.output_dir}/comparison_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Comparison summary saved to: {config.output_dir}/comparison_summary.json")
    
    # Step 8: Demonstrate loading and analyzing saved results
    print("\nStep 8: Demonstrating result loading and analysis...")
    
    # Load results from one of the models
    if comparison_results and "helpful-assistant-v1" in comparison_results:
        results_file = f"{config.output_dir}/benchmark_helpful_assistant_v1.json"
        try:
            loaded_results = benchmark.results_handler.load_results(results_file)
            print(f"Successfully loaded {len(loaded_results)} results from {results_file}")
            
            # Show some detailed analysis
            scores = [r.score for r in loaded_results]
            times = [r.execution_time for r in loaded_results]
            
            print(f"Score distribution: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")
            print(f"Time distribution: min={min(times):.3f}s, max={max(times):.3f}s, avg={sum(times)/len(times):.3f}s")
            
        except Exception as e:
            print(f"Error loading results: {e}")
    
    print("\n=== Advanced example completed successfully! ===")
    print("Check the 'advanced_benchmark_results' directory for detailed output files.")


if __name__ == "__main__":
    main()