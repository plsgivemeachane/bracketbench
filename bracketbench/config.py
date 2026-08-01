"""
Configuration management module for BracketBench.

This module provides configuration loading, validation, and access
functionality for the BracketBench system.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigManager:
    """
    Manages configuration settings for BracketBench.
    
    This class handles loading configuration from files, environment variables,
    and provides access to configuration values with validation.
    """
    
    def __init__(self) -> None:
        """Initialize the configuration manager with default values."""
        self._config: Dict[str, Any] = {}
        self._load_defaults()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self._config = {
            'models': [],
            'benchmark': {
                'test_cases': ['standard'],
                'metrics': ['accuracy', 'latency', 'cost'],
                'iterations': 1,
                'timeout': 300,
            },
            'output': {
                'directory': 'results',
                'format': 'json',
                'include_raw_data': False,
            },
            'logging': {
                'level': 'INFO',
                'file': 'bracketbench.log',
            },
            'api': {
                'retry_attempts': 3,
                'retry_delay': 1.0,
                'timeout': 30,
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the configuration file is invalid
            ValueError: If the configuration is invalid
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._merge_config(file_config)
            
            self._validate_config()
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """
        Merge new configuration with existing configuration.
        
        Args:
            new_config: New configuration dictionary to merge
        """
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge dictionaries."""
            result = base.copy()
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        self._config = merge_dict(self._config, new_config)
    
    def _validate_config(self) -> None:
        """
        Validate the current configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # Validate models section
        if 'models' in self._config:
            if not isinstance(self._config['models'], list):
                raise ValueError("'models' must be a list")
            
            for i, model in enumerate(self._config['models']):
                if not isinstance(model, dict):
                    raise ValueError(f"Model at index {i} must be a dictionary")
                
                required_fields = ['name', 'provider']
                for field in required_fields:
                    if field not in model:
                        raise ValueError(f"Model at index {i} missing required field: {field}")
        
        # Validate benchmark section
        if 'benchmark' in self._config:
            benchmark = self._config['benchmark']
            if 'iterations' in benchmark and not isinstance(benchmark['iterations'], int):
                raise ValueError("'benchmark.iterations' must be an integer")
            if 'timeout' in benchmark and not isinstance(benchmark['timeout'], (int, float)):
                raise ValueError("'benchmark.timeout' must be a number")
        
        # Validate output section
        if 'output' in self._config:
            output = self._config['output']
            if 'format' in output and output['format'] not in ['json', 'csv', 'yaml']:
                raise ValueError("'output.format' must be one of: json, csv, yaml")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'benchmark.iterations')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config_dict = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        
        # Set the value
        config_dict[keys[-1]] = value
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get the list of configured models.
        
        Returns:
            List of model configurations
        """
        return self.get('models', [])
    
    def get_benchmark_config(self) -> Dict[str, Any]:
        """
        Get the benchmark configuration.
        
        Returns:
            Benchmark configuration dictionary
        """
        return self.get('benchmark', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get the output configuration.
        
        Returns:
            Output configuration dictionary
        """
        return self.get('output', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """
        Get the API configuration.
        
        Returns:
            API configuration dictionary
        """
        return self.get('api', {})
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'BRACKETBENCH_LOG_LEVEL': 'logging.level',
            'BRACKETBENCH_OUTPUT_DIR': 'output.directory',
            'BRACKETBENCH_OUTPUT_FORMAT': 'output.format',
            'BRACKETBENCH_API_TIMEOUT': 'api.timeout',
            'BRACKETBENCH_BENCHMARK_ITERATIONS': 'benchmark.iterations',
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if config_key.endswith('.iterations') or config_key.endswith('.timeout'):
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                
                self.set(config_key, value)
    
    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration file
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_from_env()
    return _config_manager


def load_config(config_path: str) -> None:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file
    """
    get_config_manager().load_config(config_path)


def get(key: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        key: Configuration key
        default: Default value
        
    Returns:
        Configuration value or default
    """
    return get_config_manager().get(key, default)


def set(key: str, value: Any) -> None:
    """
    Set a configuration value.
    
    Args:
        key: Configuration key
        value: Value to set
    """
    get_config_manager().set(key, value)


def save_config(config_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config_path: Path to save the configuration file
    """
    get_config_manager().save_config(config_path)