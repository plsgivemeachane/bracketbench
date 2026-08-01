# BracketBench: LLM Benchmarking System

## Overview

BracketBench is a comprehensive benchmarking system designed to evaluate and compare Large Language Models (LLMs) across various tasks and metrics. The system provides a standardized framework for running benchmarks, collecting results, and analyzing model performance.

## Features

- **Multi-Model Support**: Evaluate different LLMs with a unified interface
- **Flexible Benchmarking**: Create custom test cases and evaluation metrics
- **Comprehensive Analysis**: Detailed performance metrics and statistical analysis
- **Extensible Architecture**: Easy to add new models, tests, and metrics
- **Reproducible Results**: Consistent testing environment and result tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/bracketbench/bracketbench.git
   cd bracketbench
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Production Environment

For production usage, install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `openai` - OpenAI Python library for GPT models
- `PyYAML` - YAML configuration file parsing
- `dataclasses-json` - JSON serialization for dataclasses

### Development Environment

For development, including testing and code quality tools:

```bash
pip install -r requirements-dev.txt
```

This will install all production dependencies plus:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `flake8` - Code style and error checking
- `black` - Code formatting
- `isort` - Import sorting
- `mypy` - Static type checking
- `sphinx` - Documentation generation
- `pre-commit` - Git hooks for code quality
- `tox` - Testing across multiple Python versions

### Virtual Environment (Recommended)

It's highly recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv bracketbench-env

# Activate virtual environment
# On Windows:
bracketbench-env\Scripts\activate
# On macOS/Linux:
source bracketbench-env/bin/activate

# Install dependencies (production or development)
pip install -r requirements.txt  # or requirements-dev.txt
```

### Environment Variables

Set up your API keys as environment variables:

```bash
# For OpenAI models
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Custom OpenAI base URL
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Quick Start

### Basic Usage

```python
from bracketbench import LLMManager, BenchmarkRunner

# Initialize LLM manager
llm_manager = LLMManager()

# Add models to benchmark
llm_manager.add_model("gpt-3.5-turbo", "openai")
llm_manager.add_model("claude-2", "anthropic")

# Create benchmark runner
benchmark = BenchmarkRunner()

# Run benchmark
results = benchmark.run(llm_manager, test_cases="standard")

# Analyze results
benchmark.analyze_results(results)
```

### Configuration

Create a configuration file `config.yaml`:

```yaml
models:
  - name: "gpt-3.5-turbo"
    provider: "openai"
    api_key: "your-api-key"
  
  - name: "claude-2"
    provider: "anthropic"
    api_key: "your-api-key"

benchmark:
  test_cases: "standard"
  metrics: ["accuracy", "latency", "cost"]
  iterations: 3
```

## Project Structure

```
bracketbench/
├── __init__.py          # Main package initialization
├── llms/               # LLM management module
│   ├── __init__.py
│   ├── llm_manager.py
│   ├── llm_interface.py
│   └── model_registry.py
├── benchmarking/       # Benchmarking functionality
│   ├── __init__.py
│   ├── benchmark_runner.py
│   ├── test_case.py
│   ├── result_analyzer.py
│   └── metrics.py
└── examples/           # Usage examples
    ├── __init__.py
    ├── basic_usage.py
    ├── advanced_usage.py
    └── custom_metrics.py
```

## API Reference

### LLMManager

The `LLMManager` class handles the registration and management of different LLMs.

```python
from bracketbench.llms import LLMManager

manager = LLMManager()
manager.add_model(model_name, provider, config=None)
manager.remove_model(model_name)
manager.list_models()
```

### BenchmarkRunner

The `BenchmarkRunner` class executes benchmarks and collects results.

```python
from bracketbench.benchmarking import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run(llm_manager, test_cases)
analysis = runner.analyze_results(results)
```

## Examples

See the `bracketbench/examples/` directory for detailed usage examples:

- `basic_usage.py`: Basic benchmarking workflow
- `advanced_usage.py`: Advanced configuration and custom test cases
- `custom_metrics.py`: Implementing custom evaluation metrics

## Contributing

We welcome contributions to BracketBench! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow the coding standards outlined in `docs/coding_standards.md`
- Write comprehensive tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions, please:

- Open an issue on [GitHub Issues](https://github.com/bracketbench/bracketbench/issues)
- Contact us at [contact@bracketbench.org](mailto:contact@bracketbench.org)
- Join our [Discord community](https://discord.gg/bracketbench)

## Acknowledgments

- Thanks to all contributors who have helped make BracketBench better
- Inspired by existing LLM evaluation frameworks
- Built with the Python scientific computing ecosystem