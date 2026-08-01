"""
OpenRouter LLM implementations for BracketBench.

This module provides concrete implementations of LLMInterface for various OpenRouter models,
including DeepSeek and other models available through OpenRouter.
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


class OpenRouterLLM(LLMInterface):
    """
    Base class for OpenRouter LLM implementations.
    
    This class provides common functionality for all OpenRouter models,
    including API client management and error handling.
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the OpenRouter LLM.
        
        Args:
            model_name: Name of the OpenRouter model to use
            config: Configuration dictionary for the model
        """
        super().__init__(model_name, config)
        self.client = None
        self.api_key = None
        self.base_url = "https://openrouter.ai/api/v1"
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
        Initialize the OpenRouter LLM.
        
        This method sets up the OpenRouter client and validates the configuration.
        
        Raises:
            RuntimeError: If OpenAI package is not available or initialization fails
            ValueError: If API key is not provided or invalid
        """
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI package is not available. Install it with: pip install openai")
        
        # Get API key from config or environment variable
        self.api_key = self.config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set it in config or OPENROUTER_API_KEY environment variable")
        
        # Get base URL from config if provided
        self.base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")
        
        # Set up client
        client_kwargs = {"api_key": self.api_key, "base_url": self.base_url}
        
        # Add default headers for OpenRouter
        default_headers = {
            "HTTP-Referer": "https://github.com/yourusername/bracketbench",
            "X-Title": "BracketBench"
        }
        headers = self.config.get("headers", {})
        default_headers.update(headers)
        client_kwargs["default_headers"] = default_headers
        
        try:
            self.client = OpenAI(**client_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenRouter client: {e}")
        
        # Update retry and timeout settings from config
        self.max_retries = self.config.get("max_retries", self.max_retries)
        self.retry_delay = self.config.get("retry_delay", self.retry_delay)
        self.timeout = self.config.get("timeout", self.timeout)
        
        # Update default parameters from config
        default_params = self.config.get("default_params", {})
        self.default_params.update(default_params)
        
        self._is_initialized = True
        self.logger.info(f"Initialized OpenRouter LLM with model: {self.model_name}")
    
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
                        raise RuntimeError("Empty response from OpenRouter API")
                else:
                    raise RuntimeError("No choices in response from OpenRouter API")
                
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
            "provider": "OpenRouter",
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
        Validate the configuration for this OpenRouter LLM.
        
        Returns:
            True if configuration is valid, False otherwise
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Call parent validation
        super().validate_config()
        
        # Validate OpenRouter-specific configuration
        if "api_key" not in self.config and "OPENROUTER_API_KEY" not in os.environ:
            raise ValueError("OpenRouter API key is required in config or OPENROUTER_API_KEY environment variable")
        
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


class DeepSeekR1LLM(OpenRouterLLM):
    """
    DeepSeek R1 LLM implementation through OpenRouter.
    
    This class provides a specific implementation for the DeepSeek R1 model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the DeepSeek R1 LLM.
        
        Args:
            config: Configuration dictionary for the model
        """
        super().__init__("deepseek/deepseek-r1-0528:free", config)
        
        # Set DeepSeek R1 specific default parameters
        self.default_params.update({
            "max_tokens": 2048,
            "temperature": 0.7,
        })
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the DeepSeek R1 model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            "name": "deepseek/deepseek-r1-0528:free",
            "description": "DeepSeek R1 - A powerful reasoning model available through OpenRouter",
            "context_length": 16384,
            "capabilities": ["text-generation", "chat", "reasoning", "code-generation"],
            "pricing": {
                "input": "Free",
                "output": "Free"
            }
        })
        return info