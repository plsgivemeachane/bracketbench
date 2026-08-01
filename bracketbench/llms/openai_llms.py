"""
OpenAI LLM implementations for BracketBench.

This module provides concrete implementations of LLMInterface for various OpenAI models,
including GPT-3.5, GPT-4, and other OpenAI models.
"""

import os
import time
from typing import Dict, Any, Optional, List, Union
import logging

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import LLMInterface


class OpenAILLM(LLMInterface):
    """
    Base class for OpenAI LLM implementations.
    
    This class provides common functionality for all OpenAI models,
    including API client management and error handling.
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the OpenAI LLM.
        
        Args:
            model_name: Name of the OpenAI model to use
            config: Configuration dictionary for the model
        """
        super().__init__(model_name, config)
        self.client = None
        self.api_key = None
        self.base_url = None
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30
        
        # Default parameters
        self.default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    
    def initialize(self) -> None:
        """
        Initialize the OpenAI LLM.
        
        This method sets up the OpenAI client and validates the configuration.
        
        Raises:
            RuntimeError: If OpenAI package is not available or initialization fails
            ValueError: If API key is not provided or invalid
        """
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI package is not available. Install it with: pip install openai")
        
        # Get API key from config or environment variable
        self.api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in config or OPENAI_API_KEY environment variable")
        
        # Get base URL from config if provided
        self.base_url = self.config.get("base_url")
        
        # Set up client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        try:
            self.client = OpenAI(**client_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        # Update retry and timeout settings from config
        self.max_retries = self.config.get("max_retries", self.max_retries)
        self.retry_delay = self.config.get("retry_delay", self.retry_delay)
        self.timeout = self.config.get("timeout", self.timeout)
        
        # Update default parameters from config
        default_params = self.config.get("default_params", {})
        self.default_params.update(default_params)
        
        self._is_initialized = True
        self.logger.info(f"Initialized OpenAI LLM with model: {self.model_name}")
    
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
        if not self.is_initialized():
            self.initialize()
        
        # Prepare parameters
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.default_params["max_tokens"],
            "temperature": temperature or self.default_params["temperature"],
            "top_p": kwargs.get("top_p", self.default_params["top_p"]),
            "frequency_penalty": kwargs.get("frequency_penalty", self.default_params["frequency_penalty"]),
            "presence_penalty": kwargs.get("presence_penalty", self.default_params["presence_penalty"]),
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        # Make API call with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    **params,
                    timeout=self.timeout
                )
                
                # Extract and return the generated text
                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message
                    if message.content:
                        return message.content.strip()
                    else:
                        raise RuntimeError("Empty response from OpenAI API")
                else:
                    raise RuntimeError("No choices in response from OpenAI API")
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    break
        
        raise RuntimeError(f"Failed to generate text after {self.max_retries} attempts: {last_error}")
    
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
        if not self.is_initialized():
            self.initialize()
        
        if not prompts:
            return []
        
        # Generate responses for each prompt
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Failed to generate response for prompt: {prompt[:50]}... Error: {e}")
                responses.append(f"ERROR: {str(e)}")
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.model_name,
            "provider": "OpenAI",
            "api_type": "chat",
            "capabilities": ["text-generation", "chat"],
            "parameters": {
                "max_tokens": self.default_params["max_tokens"],
                "temperature": self.default_params["temperature"],
                "top_p": self.default_params["top_p"],
                "frequency_penalty": self.default_params["frequency_penalty"],
                "presence_penalty": self.default_params["presence_penalty"],
            },
            "config": {
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "timeout": self.timeout,
                "base_url": self.base_url,
            }
        }
    
    def validate_config(self) -> bool:
        """
        Validate the configuration for this OpenAI LLM.
        
        Returns:
            True if configuration is valid, False otherwise
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Call parent validation
        super().validate_config()
        
        # Validate OpenAI-specific configuration
        if "api_key" not in self.config and "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key is required in config or OPENAI_API_KEY environment variable")
        
        if "max_retries" in self.config and not isinstance(self.config["max_retries"], int):
            raise ValueError("max_retries must be an integer")
        
        if "retry_delay" in self.config and not isinstance(self.config["retry_delay"], (int, float)):
            raise ValueError("retry_delay must be a number")
        
        if "timeout" in self.config and not isinstance(self.config["timeout"], (int, float)):
            raise ValueError("timeout must be a number")
        
        # Validate default parameters
        if "default_params" in self.config:
            default_params = self.config["default_params"]
            if not isinstance(default_params, dict):
                raise ValueError("default_params must be a dictionary")
            
            for param_name, param_value in default_params.items():
                if param_name in ["temperature", "top_p"] and not (0.0 <= param_value <= 1.0):
                    raise ValueError(f"{param_name} must be between 0.0 and 1.0")
                
                if param_name == "max_tokens" and not isinstance(param_value, int):
                    raise ValueError("max_tokens must be an integer")
        
        return True


class GPT35LLM(OpenAILLM):
    """
    GPT-3.5 Turbo LLM implementation.
    
    This class provides a specific implementation for the GPT-3.5 Turbo model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the GPT-3.5 Turbo LLM.
        
        Args:
            config: Configuration dictionary for the model
        """
        super().__init__("gpt-3.5-turbo", config)
        
        # Set GPT-3.5 Turbo specific default parameters
        self.default_params.update({
            "max_tokens": 2048,
            "temperature": 0.7,
        })
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the GPT-3.5 Turbo model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            "name": "gpt-3.5-turbo",
            "description": "GPT-3.5 Turbo - A capable and cost-effective model for most tasks",
            "context_length": 16385,
            "training_data": "Up to Sep 2021",
            "pricing": {
                "input": "$0.0015 per 1K tokens",
                "output": "$0.002 per 1K tokens"
            }
        })
        return info


class GPT4LLM(OpenAILLM):
    """
    GPT-4 LLM implementation.
    
    This class provides a specific implementation for the GPT-4 model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the GPT-4 LLM.
        
        Args:
            config: Configuration dictionary for the model
        """
        super().__init__("gpt-4", config)
        
        # Set GPT-4 specific default parameters
        self.default_params.update({
            "max_tokens": 4096,
            "temperature": 0.7,
        })
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the GPT-4 model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            "name": "gpt-4",
            "description": "GPT-4 - The most capable model for complex tasks",
            "context_length": 8192,
            "training_data": "Up to Sep 2021",
            "pricing": {
                "input": "$0.03 per 1K tokens",
                "output": "$0.06 per 1K tokens"
            }
        })
        return info


class GPT4TurboLLM(OpenAILLM):
    """
    GPT-4 Turbo LLM implementation.
    
    This class provides a specific implementation for the GPT-4 Turbo model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the GPT-4 Turbo LLM.
        
        Args:
            config: Configuration dictionary for the model
        """
        super().__init__("gpt-4-1106-preview", config)
        
        # Set GPT-4 Turbo specific default parameters
        self.default_params.update({
            "max_tokens": 4096,
            "temperature": 0.7,
        })
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the GPT-4 Turbo model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            "name": "gpt-4-1106-preview",
            "description": "GPT-4 Turbo - The latest GPT-4 model with improved capabilities",
            "context_length": 128000,
            "training_data": "Up to Apr 2023",
            "pricing": {
                "input": "$0.01 per 1K tokens",
                "output": "$0.03 per 1K tokens"
            }
        })
        return info