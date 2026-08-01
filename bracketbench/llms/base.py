"""
Base LLM interface module for BracketBench.

This module provides the abstract base class for all LLM implementations,
ensuring a consistent interface across different model providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import logging


class LLMInterface(ABC):
    """
    Abstract base class for all LLM implementations.
    
    This class defines the interface that all LLM implementations must follow,
    ensuring consistent behavior across different model providers.
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the LLM with a model name and configuration.
        
        Args:
            model_name: Name of the model to use
            config: Configuration dictionary for the model
        """
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the LLM model.
        
        This method should handle any setup required for the model,
        such as loading model weights, establishing connections, etc.
        
        Raises:
            RuntimeError: If initialization fails
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text based on the provided prompt.
        
        Args:
            prompt: The input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If text generation fails
            ValueError: If invalid parameters are provided
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self, 
        prompts: List[str], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate for each prompt
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of generated text responses
            
        Raises:
            RuntimeError: If batch text generation fails
            ValueError: If invalid parameters are provided
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information such as:
            - name: Model name
            - provider: Model provider
            - version: Model version
            - capabilities: List of model capabilities
            - parameters: Model-specific parameters
        """
        pass
    
    def is_initialized(self) -> bool:
        """
        Check if the LLM is initialized.
        
        Returns:
            True if the LLM is initialized, False otherwise
        """
        return self._is_initialized
    
    def validate_config(self) -> bool:
        """
        Validate the configuration for this LLM.
        
        Returns:
            True if configuration is valid, False otherwise
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Basic validation - can be overridden by subclasses
        if not self.model_name:
            raise ValueError("Model name is required")
        
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        return True
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration for this LLM.
        
        Args:
            config: New configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        old_config = self.config.copy()
        self.config.update(config)
        
        try:
            self.validate_config()
        except ValueError as e:
            # Revert to old configuration if validation fails
            self.config = old_config
            raise ValueError(f"Invalid configuration: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration for this LLM.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def __str__(self) -> str:
        """String representation of the LLM."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the LLM."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', config={self.config})"