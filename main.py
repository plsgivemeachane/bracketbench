#!/usr/bin/env python3
"""
Main entry point for BracketBench LLM benchmarking system.

This module provides the command-line interface and main execution
logic for running benchmarks on Large Language Models.
"""

import argparse
import sys
import logging
from pathlib import Path

from bracketbench import config
from bracketbench.llms import LLMManager
from bracketbench.benchmarking import BenchmarkRunner


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bracketbench.log')
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the application.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="BracketBench: LLM Benchmarking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config.yaml --test-cases standard
  python main.py --models gpt-3.5-turbo,claude-2 --metrics accuracy,latency
  python main.py --list-models
  python main.py --validate-config config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML format)'
    )
    
    parser.add_argument(
        '--models', '-m',
        type=str,
        help='Comma-separated list of models to benchmark'
    )
    
    parser.add_argument(
        '--test-cases', '-t',
        type=str,
        default='standard',
        help='Test cases to run (default: standard)'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        help='Comma-separated list of metrics to collect'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=1,
        help='Number of iterations to run (default: 1)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--validate-config',
        type=str,
        help='Validate configuration file and exit'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    return parser.parse_args()


def list_available_models() -> None:
    """List all available models from the model registry."""
    from bracketbench.llms import ModelRegistry
    
    registry = ModelRegistry()
    models = registry.get_available_models()
    
    print("Available models:")
    for model in models:
        print(f"  - {model['name']} ({model['provider']})")
        if model['description']:
            print(f"    Description: {model['description']}")


def validate_config_file(config_path: str) -> bool:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        config.load_config(config_path)
        print(f"Configuration file '{config_path}' is valid.")
        return True
    except Exception as e:
        print(f"Configuration file '{config_path}' is invalid: {e}")
        return False


def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Handle special commands
        if args.list_models:
            list_available_models()
            return 0
            
        if args.validate_config:
            return 0 if validate_config_file(args.validate_config) else 1
        
        # Load configuration
        if args.config:
            config.load_config(args.config)
        else:
            logger.info("No configuration file provided, using defaults")
        
        # Initialize LLM manager
        llm_manager = LLMManager()
        
        # Add models to benchmark
        if args.models:
            model_names = args.models.split(',')
            for model_name in model_names:
                model_name = model_name.strip()
                if model_name:
                    llm_manager.add_model(model_name)
        else:
            # Use models from configuration
            configured_models = config.get('models', [])
            for model_config in configured_models:
                llm_manager.add_model(
                    model_config['name'],
                    model_config['provider'],
                    model_config.get('config')
                )
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmark runner
        benchmark = BenchmarkRunner(
            output_dir=str(output_dir),
            iterations=args.iterations
        )
        
        # Parse test cases and metrics
        test_cases = args.test_cases.split(',') if args.test_cases else ['standard']
        metrics = args.metrics.split(',') if args.metrics else None
        
        # Run benchmark
        logger.info(f"Starting benchmark with {len(llm_manager.list_models())} models")
        results = benchmark.run(
            llm_manager=llm_manager,
            test_cases=test_cases,
            metrics=metrics
        )
        
        # Analyze and save results
        logger.info("Analyzing results")
        analysis = benchmark.analyze_results(results)
        benchmark.save_results(results, analysis)
        
        logger.info(f"Benchmark completed successfully. Results saved to {output_dir}")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())