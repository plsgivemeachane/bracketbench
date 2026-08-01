"""
LLM manager and factory module for BracketBench.

This module provides the LLMManager class for creating and managing different LLM instances,
following the factory pattern for dynamic loading and configuration of LLM implementations.
"""

import os
import importlib
import inspect
from typing import Dict, Any, Optional, List, Type, Union
import logging

from .base import LLMInterface
from .openai_llms import OpenAILLM, GPT35LLM, GPT4LLM, GPT4TurboLLM
from .openrouter_llms import OpenRouterLLM, DeepSeekR1LLM


class LLMManager:
    """
    Manager class for creating and managing LLM instances.
    
    This class implements the factory pattern to create and manage different LLM
    implementations, providing a unified interface for working with various models.
    """
    
    def __init__(self) -> None:
        """Initialize the LLM manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._llm_instances: Dict[str, LLMInterface] = {}
        self._llm_classes: Dict[str, Type[LLMInterface]] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in LLM classes
        self._register_builtin_llms()
    
    def _register_builtin_llms(self) -> None:
        """Register built-in LLM classes."""
        self.register_llm_class("openai", OpenAILLM)
        self.register_llm_class("gpt-3.5-turbo", GPT35LLM)
        self.register_llm_class("gpt-4", GPT4LLM)
        self.register_llm_class("gpt-4-1106-preview", GPT4TurboLLM)
        self.register_llm_class("openrouter", OpenRouterLLM)
        self.register_llm_class("deepseek/deepseek-r1-0528:free", DeepSeekR1LLM)
    
    def register_llm_class(self, name: str, llm_class: Type[LLMInterface]) -> None:
        """
        Register an LLM class with the manager.
        
        Args:
            name: Name to register the class under
            llm_class: LLM class to register
            
        Raises:
            TypeError: If llm_class is not a subclass of LLMInterface
            ValueError: If name is already registered
        """
        if not inspect.isclass(llm_class) or not issubclass(llm_class, LLMInterface):
            raise TypeError(f"llm_class must be a subclass of LLMInterface, got {type(llm_class)}")
        
        if name in self._llm_classes:
            self.logger.warning(f"LLM class '{name}' already registered, overwriting")
        
        self._llm_classes[name] = llm_class
        self.logger.info(f"Registered LLM class: {name}")
    
    def register_llm_from_module(self, name: str, module_path: str, class_name: str) -> None:
        """
        Register an LLM class from a module dynamically.
        
        Args:
            name: Name to register the class under
            module_path: Python module path (e.g., "my_package.my_module")
            class_name: Name of the LLM class in the module
            
        Raises:
            ImportError: If the module cannot be imported
            AttributeError: If the class is not found in the module
            TypeError: If the class is not a subclass of LLMInterface
        """
        try:
            module = importlib.import_module(module_path)
            llm_class = getattr(module, class_name)
            
            if not inspect.isclass(llm_class) or not issubclass(llm_class, LLMInterface):
                raise TypeError(f"Class '{class_name}' is not a subclass of LLMInterface")
            
            self.register_llm_class(name, llm_class)
            
        except ImportError as e:
            raise ImportError(f"Failed to import module '{module_path}': {e}")
        except AttributeError as e:
            raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}")
    
    def create_llm(
        self, 
        model_name: str, 
        provider: Optional[str] = None, 
        config: Optional[Dict[str, Any]] = None
    ) -> LLMInterface:
        """
        Create an LLM instance using the factory pattern.
        
        Args:
            model_name: Name of the model to create
            provider: Provider name (optional, inferred from model_name if not provided)
            config: Configuration dictionary for the model
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If model_name or provider is invalid
            RuntimeError: If LLM creation fails
        """
        # Determine provider if not specified
        if provider is None:
            provider = self._infer_provider(model_name)
        
        # Get the LLM class
        if provider not in self._llm_classes:
            raise ValueError(f"Unknown LLM provider: {provider}. Available providers: {list(self._llm_classes.keys())}")
        
        llm_class = self._llm_classes[provider]
        
        try:
            # Create LLM instance
            llm_instance = llm_class(model_name, config)
            
            # Store model config for future reference
            self._model_configs[model_name] = config or {}
            
            self.logger.info(f"Created LLM instance: {model_name} (provider: {provider})")
            return llm_instance
            
        except Exception as e:
            raise RuntimeError(f"Failed to create LLM instance '{model_name}': {e}")
    
    def add_model(
        self, 
        model_name: str, 
        provider: Optional[str] = None, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an LLM model to the manager.
        
        Args:
            model_name: Name of the model to add
            provider: Provider name (optional, inferred from model_name if not provided)
            config: Configuration dictionary for the model
            
        Raises:
            ValueError: If model_name is invalid or already exists
            RuntimeError: If model creation fails
        """
        if model_name in self._llm_instances:
            self.logger.warning(f"Model '{model_name}' already exists, replacing")
            self.remove_model(model_name)
        
        # Create LLM instance
        llm_instance = self.create_llm(model_name, provider, config)
        
        # Store the instance
        self._llm_instances[model_name] = llm_instance
        
        self.logger.info(f"Added model: {model_name}")
    
    def remove_model(self, model_name: str) -> None:
        """
        Remove an LLM model from the manager.
        
        Args:
            model_name: Name of the model to remove
            
        Raises:
            ValueError: If model_name is not found
        """
        if model_name not in self._llm_instances:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Remove the instance
        del self._llm_instances[model_name]
        
        # Remove config if it exists
        if model_name in self._model_configs:
            del self._model_configs[model_name]
        
        self.logger.info(f"Removed model: {model_name}")
    
    def get_model(self, model_name: str) -> LLMInterface:
        """
        Get an LLM model instance.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If model_name is not found
        """
        if model_name not in self._llm_instances:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self._llm_instances[model_name]
    
    def list_models(self) -> List[str]:
        """
        Get a list of all registered model names.
        
        Returns:
            List of model names
        """
        return list(self._llm_instances.keys())
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
            
        Raises:
            ValueError: If model_name is not found
        """
        if model_name not in self._model_configs:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self._model_configs[model_name].copy()
    
    def update_model_config(self, model_name: str, config: Dict[str, Any]) -> None:
        """
        Update the configuration for a model.
        
        Args:
            model_name: Name of the model
            config: New configuration dictionary
            
        Raises:
            ValueError: If model_name is not found
        """
        if model_name not in self._llm_instances:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Update the model's configuration
        llm_instance = self._llm_instances[model_name]
        llm_instance.update_config(config)
        
        # Update stored config
        self._model_configs[model_name].update(config)
        
        self.logger.info(f"Updated configuration for model: {model_name}")
    
    def get_available_providers(self) -> List[str]:
        """
        Get a list of all available LLM providers.
        
        Returns:
            List of provider names
        """
        return list(self._llm_classes.keys())
    
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """
        Get information about a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider information dictionary
            
        Raises:
            ValueError: If provider is not found
        """
        if provider not in self._llm_classes:
            raise ValueError(f"Provider '{provider}' not found")
        
        llm_class = self._llm_classes[provider]
        
        return {
            "name": provider,
            "class": llm_class.__name__,
            "module": llm_class.__module__,
            "description": llm_class.__doc__ or "No description available",
        }
    
    def load_models_from_config(self, models_config: List[Dict[str, Any]]) -> None:
        """
        Load multiple models from a configuration list.
        
        Args:
            models_config: List of model configuration dictionaries
            
        Raises:
            ValueError: If any model configuration is invalid
            RuntimeError: If any model creation fails
        """
        for model_config in models_config:
            if not isinstance(model_config, dict):
                raise ValueError("Each model configuration must be a dictionary")
            
            name = model_config.get("name")
            provider = model_config.get("provider")
            config = model_config.get("config", {})
            
            if not name:
                raise ValueError("Model configuration missing 'name' field")
            
            try:
                self.add_model(name, provider, config)
            except Exception as e:
                raise RuntimeError(f"Failed to load model '{name}': {e}")
    
    def initialize_all_models(self) -> None:
        """
        Initialize all registered LLM models.
        
        Raises:
            RuntimeError: If any model initialization fails
        """
        failed_models = []
        
        for model_name, llm_instance in self._llm_instances.items():
            try:
                if not llm_instance.is_initialized():
                    llm_instance.initialize()
                    self.logger.info(f"Initialized model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize model '{model_name}': {e}")
                failed_models.append(model_name)
        
        if failed_models:
            raise RuntimeError(f"Failed to initialize models: {failed_models}")
    
    def _infer_provider(self, model_name: str) -> str:
        """
        Infer the provider from the model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Inferred provider name
            
        Raises:
            ValueError: If provider cannot be inferred
        """
        # Check if model_name matches a registered class name
        if model_name in self._llm_classes:
            return model_name
        
        # Try to infer from model name patterns
        if model_name.startswith(("gpt-", "text-davinci-", "text-curie-")):
            return "openai"
        if model_name.startswith(("deepseek/", "anthropic/", "meta/", "google/")):
            return "openrouter"
        
        # Default fallback
        raise ValueError(f"Cannot infer provider for model: {model_name}. Please specify provider explicitly.")


class ModelRegistry:
    """
    Registry for available LLM models and their metadata.
    
    This class maintains a registry of available models with their metadata,
    making it easy to discover and query available models.
    """
    
    def __init__(self) -> None:
        """Initialize the model registry."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._models: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in models
        self._register_builtin_models()
    
    def _register_builtin_models(self) -> None:
        """Register built-in models with their metadata."""
        self.register_model(
            name="gpt-3.5-turbo",
            provider="openai",
            description="GPT-3.5 Turbo - A capable and cost-effective model for most tasks",
            capabilities=["text-generation", "chat", "code-generation"],
            context_length=16385,
            parameters={
                "max_tokens": 4096,
                "temperature": 0.7,
            }
        )
        
        self.register_model(
            name="gpt-4",
            provider="openai",
            description="GPT-4 - The most capable model for complex tasks",
            capabilities=["text-generation", "chat", "code-generation", "reasoning"],
            context_length=8192,
            parameters={
                "max_tokens": 8192,
                "temperature": 0.7,
            }
        )
        
        self.register_model(
            name="gpt-4-1106-preview",
            provider="openai",
            description="GPT-4 Turbo - The latest GPT-4 model with improved capabilities",
            capabilities=["text-generation", "chat", "code-generation", "reasoning"],
            context_length=128000,
            parameters={
                "max_tokens": 4096,
                "temperature": 0.7,
            }
        )
        
        self.register_model(
            name="deepseek/deepseek-r1-0528:free",
            provider="openrouter",
            description="DeepSeek R1 - A powerful reasoning model available through OpenRouter",
            capabilities=["text-generation", "chat", "reasoning", "code-generation"],
            context_length=16384,
            parameters={
                "max_tokens": 2048,
                "temperature": 0.7,
            }
        )
    
    def register_model(
        self, 
        name: str, 
        provider: str, 
        description: str,
        capabilities: List[str],
        context_length: int,
        parameters: Dict[str, Any],
        **kwargs
    ) -> None:
        """
        Register a model with the registry.
        
        Args:
            name: Name of the model
            provider: Provider name
            description: Description of the model
            capabilities: List of model capabilities
            context_length: Context length of the model
            parameters: Default parameters for the model
            **kwargs: Additional metadata
        """
        model_info = {
            "name": name,
            "provider": provider,
            "description": description,
            "capabilities": capabilities,
            "context_length": context_length,
            "parameters": parameters,
            **kwargs
        }
        
        self._models[name] = model_info
        self.logger.info(f"Registered model: {name} (provider: {provider})")
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            name: Name of the model
            
        Returns:
            Model information dictionary
            
        Raises:
            ValueError: If model is not found
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")
        
        return self._models[name].copy()
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available models.
        
        Returns:
            List of model information dictionaries
        """
        return list(self._models.values())
    
    def get_models_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """
        Get models by provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of model information dictionaries for the provider
        """
        return [model for model in self._models.values() if model["provider"] == provider]
    
    def get_models_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """
        Get models by capability.
        
        Args:
            capability: Capability name
            
        Returns:
            List of model information dictionaries with the capability
        """
        return [
            model for model in self._models.values() 
            if capability in model.get("capabilities", [])
        ]
    
    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for models by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching model information dictionaries
        """
        query_lower = query.lower()
        results = []
        
        for model in self._models.values():
            if (query_lower in model["name"].lower() or 
                query_lower in model["description"].lower() or
                any(query_lower in capability.lower() for capability in model.get("capabilities", []))):
                results.append(model)
        
        return results