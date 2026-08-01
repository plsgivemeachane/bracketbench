"""
Unit tests for the LLM classes in BracketBench.

This module contains comprehensive unit tests for the LLM interface,
implementations, and manager classes.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List
import tempfile
import os

from bracketbench.llms.base import LLMInterface
from bracketbench.llms.openai_llms import OpenAILLM, GPT35LLM, GPT4LLM, GPT4TurboLLM
from bracketbench.llms.manager import LLMManager, ModelRegistry


class MockLLM(LLMInterface):
    """
    A mock LLM implementation for testing purposes.
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the mock LLM."""
        super().__init__(model_name, config)
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the mock LLM."""
        self.initialized = True
        self._is_initialized = True
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate a mock response."""
        if not self.is_initialized():
            self.initialize()
        return f"Mock response to: {prompt}"
    
    def generate_batch(
        self, 
        prompts: List[str], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """Generate mock responses for multiple prompts."""
        return [self.generate(prompt, max_tokens, temperature, **kwargs) for prompt in prompts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "name": self.model_name,
            "provider": "MockProvider",
            "version": "1.0.0",
            "capabilities": ["text-generation"]
        }


class TestLLMInterface(unittest.TestCase):
    """Test cases for the LLMInterface abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that LLMInterface cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            LLMInterface("test-model")
    
    def test_concrete_implementation(self) -> None:
        """Test that concrete implementations work correctly."""
        mock_llm = MockLLM("test-model")
        
        # Test initialization
        self.assertEqual(mock_llm.model_name, "test-model")
        self.assertEqual(mock_llm.config, {})
        self.assertFalse(mock_llm.is_initialized())
        
        # Test initialization
        mock_llm.initialize()
        self.assertTrue(mock_llm.is_initialized())
        
        # Test generation
        response = mock_llm.generate("Hello")
        self.assertEqual(response, "Mock response to: Hello")
        
        # Test batch generation
        responses = mock_llm.generate_batch(["Hello", "World"])
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0], "Mock response to: Hello")
        self.assertEqual(responses[1], "Mock response to: World")
        
        # Test model info
        info = mock_llm.get_model_info()
        self.assertEqual(info["name"], "test-model")
        self.assertEqual(info["provider"], "MockProvider")
    
    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Valid config
        mock_llm = MockLLM("test-model", {"param": "value"})
        self.assertTrue(mock_llm.validate_config())
        
        # Missing model name
        with self.assertRaises(ValueError):
            MockLLM("").validate_config()
        
        # Invalid config type
        mock_llm = MockLLM("test-model")
        mock_llm.config = "invalid"
        with self.assertRaises(ValueError):
            mock_llm.validate_config()
    
    def test_config_update(self) -> None:
        """Test configuration update."""
        mock_llm = MockLLM("test-model", {"param1": "value1"})
        
        # Valid update
        mock_llm.update_config({"param2": "value2"})
        self.assertEqual(mock_llm.config, {"param1": "value1", "param2": "value2"})
        
        # Invalid update should revert
        original_config = mock_llm.config.copy()
        with self.assertRaises(ValueError):
            mock_llm.update_config({"model_name": ""})  # Invalid model name
        
        # Config should be unchanged
        self.assertEqual(mock_llm.config, original_config)
    
    def test_string_representation(self) -> None:
        """Test string representation."""
        mock_llm = MockLLM("test-model", {"param": "value"})
        
        str_repr = str(mock_llm)
        self.assertIn("MockLLM", str_repr)
        self.assertIn("test-model", str_repr)
        
        repr_str = repr(mock_llm)
        self.assertIn("MockLLM", repr_str)
        self.assertIn("test-model", repr_str)
        self.assertIn("param", repr_str)


class TestOpenAILLM(unittest.TestCase):
    """Test cases for the OpenAI LLM implementations."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock openai availability
        self.patcher = patch('bracketbench.llms.openai_llms.OPENAI_AVAILABLE', True)
        self.mock_available = self.patcher.start()
        
        # Mock OpenAI class
        self.mock_openai = Mock()
        self.openai_patcher = patch('bracketbench.llms.openai_llms.OpenAI', return_value=self.mock_openai)
        self.mock_openai_class = self.openai_patcher.start()
    
    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.openai_patcher.stop()
        self.patcher.stop()
    
    def test_initialization(self) -> None:
        """Test OpenAI LLM initialization."""
        llm = OpenAILLM("gpt-3.5-turbo", {"api_key": "test-key"})
        
        self.assertEqual(llm.model_name, "gpt-3.5-turbo")
        self.assertEqual(llm.api_key, "test-key")
        self.assertEqual(llm.max_retries, 3)
        self.assertEqual(llm.retry_delay, 1.0)
        self.assertEqual(llm.timeout, 30)
        self.assertFalse(llm.is_initialized())
    
    def test_initialization_without_openai(self) -> None:
        """Test initialization fails when OpenAI is not available."""
        with patch('bracketbench.llms.openai_llms.OPENAI_AVAILABLE', False):
            llm = OpenAILLM("gpt-3.5-turbo", {"api_key": "test-key"})
            
            with self.assertRaises(RuntimeError) as context:
                llm.initialize()
            
            self.assertIn("OpenAI package is not available", str(context.exception))
    
    def test_initialization_without_api_key(self) -> None:
        """Test initialization fails without API key."""
        llm = OpenAILLM("gpt-3.5-turbo")
        
        # Mock environment variable to not have the key
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                llm.initialize()
            
            self.assertIn("OpenAI API key is required", str(context.exception))
    
    def test_successful_initialization(self) -> None:
        """Test successful initialization."""
        llm = OpenAILLM("gpt-3.5-turbo", {"api_key": "test-key"})
        
        llm.initialize()
        
        self.assertTrue(llm.is_initialized())
        self.mock_openai_class.assert_called_once_with(api_key="test-key")
    
    def test_generate_response(self) -> None:
        """Test text generation."""
        # Set up mock response
        mock_choice = Mock()
        mock_choice.message.content = "Generated response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        self.mock_openai.chat.completions.create.return_value = mock_response
        
        llm = OpenAILLM("gpt-3.5-turbo", {"api_key": "test-key"})
        llm.initialize()
        
        response = llm.generate("Hello world")
        
        self.assertEqual(response, "Generated response")
        self.mock_openai.chat.completions.create.assert_called_once()
        
        # Check call arguments
        call_args = self.mock_openai.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], "gpt-3.5-turbo")
        self.assertEqual(call_args[1]["messages"][0]["role"], "user")
        self.assertEqual(call_args[1]["messages"][0]["content"], "Hello world")
    
    def test_generate_batch_responses(self) -> None:
        """Test batch text generation."""
        # Set up mock response
        mock_choice = Mock()
        mock_choice.message.content = "Generated response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        self.mock_openai.chat.completions.create.return_value = mock_response
        
        llm = OpenAILLM("gpt-3.5-turbo", {"api_key": "test-key"})
        llm.initialize()
        
        prompts = ["Hello", "World"]
        responses = llm.generate_batch(prompts)
        
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0], "Generated response")
        self.assertEqual(responses[1], "Generated response")
        
        # Should be called twice
        self.assertEqual(self.mock_openai.chat.completions.create.call_count, 2)
    
    def test_generate_with_error_retry(self) -> None:
        """Test generation with error and retry."""
        # Set up mock to fail first, then succeed
        self.mock_openai.chat.completions.create.side_effect = [
            Exception("API Error"),
            Mock(choices=[Mock(message=Mock(content="Retry success"))])
        ]
        
        llm = OpenAILLM("gpt-3.5-turbo", {"api_key": "test-key"})
        llm.initialize()
        
        response = llm.generate("Hello")
        
        self.assertEqual(response, "Retry success")
        self.assertEqual(self.mock_openai.chat.completions.create.call_count, 2)
    
    def test_generate_with_all_retries_failed(self) -> None:
        """Test generation when all retries fail."""
        # Set up mock to always fail
        self.mock_openai.chat.completions.create.side_effect = Exception("API Error")
        
        llm = OpenAILLM("gpt-3.5-turbo", {"api_key": "test-key"})
        llm.initialize()
        
        with self.assertRaises(RuntimeError) as context:
            llm.generate("Hello")
        
        self.assertIn("Failed to generate text after", str(context.exception))
        self.assertEqual(self.mock_openai.chat.completions.create.call_count, 3)  # max_retries
    
    def test_get_model_info(self) -> None:
        """Test getting model information."""
        llm = OpenAILLM("gpt-3.5-turbo", {"api_key": "test-key"})
        llm.initialize()
        
        info = llm.get_model_info()
        
        self.assertEqual(info["name"], "gpt-3.5-turbo")
        self.assertEqual(info["provider"], "OpenAI")
        self.assertEqual(info["api_type"], "chat")
        self.assertIn("text-generation", info["capabilities"])
        self.assertIn("chat", info["capabilities"])
    
    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Valid config
        llm = OpenAILLM("gpt-3.5-turbo", {
            "api_key": "test-key",
            "max_retries": 5,
            "retry_delay": 2.0,
            "timeout": 60,
            "default_params": {
                "temperature": 0.5,
                "max_tokens": 500
            }
        })
        
        self.assertTrue(llm.validate_config())
        
        # Invalid max_retries
        llm.config["max_retries"] = "invalid"
        with self.assertRaises(ValueError):
            llm.validate_config()
        
        # Invalid temperature
        llm.config["max_retries"] = 3  # Fix previous
        llm.config["default_params"]["temperature"] = 2.0  # Out of range
        with self.assertRaises(ValueError):
            llm.validate_config()
        
        # Invalid default_params type
        llm.config["default_params"] = "invalid"
        with self.assertRaises(ValueError):
            llm.validate_config()


class TestGPT35LLM(unittest.TestCase):
    """Test cases for the GPT-3.5 Turbo LLM implementation."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch('bracketbench.llms.openai_llms.OPENAI_AVAILABLE', True)
        self.mock_available = self.patcher.start()
    
    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.patcher.stop()
    
    def test_initialization(self) -> None:
        """Test GPT-3.5 Turbo initialization."""
        llm = GPT35LLM({"api_key": "test-key"})
        
        self.assertEqual(llm.model_name, "gpt-3.5-turbo")
        self.assertEqual(llm.default_params["max_tokens"], 2048)
        self.assertEqual(llm.default_params["temperature"], 0.7)
    
    def test_get_model_info(self) -> None:
        """Test getting model information."""
        llm = GPT35LLM({"api_key": "test-key"})
        
        info = llm.get_model_info()
        
        self.assertEqual(info["name"], "gpt-3.5-turbo")
        self.assertEqual(info["description"], "GPT-3.5 Turbo - A capable and cost-effective model for most tasks")
        self.assertEqual(info["context_length"], 16385)
        self.assertEqual(info["training_data"], "Up to Sep 2021")
        self.assertIn("pricing", info)


class TestGPT4LLM(unittest.TestCase):
    """Test cases for the GPT-4 LLM implementation."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch('bracketbench.llms.openai_llms.OPENAI_AVAILABLE', True)
        self.mock_available = self.patcher.start()
    
    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.patcher.stop()
    
    def test_initialization(self) -> None:
        """Test GPT-4 initialization."""
        llm = GPT4LLM({"api_key": "test-key"})
        
        self.assertEqual(llm.model_name, "gpt-4")
        self.assertEqual(llm.default_params["max_tokens"], 4096)
        self.assertEqual(llm.default_params["temperature"], 0.7)
    
    def test_get_model_info(self) -> None:
        """Test getting model information."""
        llm = GPT4LLM({"api_key": "test-key"})
        
        info = llm.get_model_info()
        
        self.assertEqual(info["name"], "gpt-4")
        self.assertEqual(info["description"], "GPT-4 - The most capable model for complex tasks")
        self.assertEqual(info["context_length"], 8192)
        self.assertIn("reasoning", info["capabilities"])


class TestGPT4TurboLLM(unittest.TestCase):
    """Test cases for the GPT-4 Turbo LLM implementation."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch('bracketbench.llms.openai_llms.OPENAI_AVAILABLE', True)
        self.mock_available = self.patcher.start()
    
    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.patcher.stop()
    
    def test_initialization(self) -> None:
        """Test GPT-4 Turbo initialization."""
        llm = GPT4TurboLLM({"api_key": "test-key"})
        
        self.assertEqual(llm.model_name, "gpt-4-1106-preview")
        self.assertEqual(llm.default_params["max_tokens"], 4096)
        self.assertEqual(llm.default_params["temperature"], 0.7)
    
    def test_get_model_info(self) -> None:
        """Test getting model information."""
        llm = GPT4TurboLLM({"api_key": "test-key"})
        
        info = llm.get_model_info()
        
        self.assertEqual(info["name"], "gpt-4-1106-preview")
        self.assertEqual(info["description"], "GPT-4 Turbo - The latest GPT-4 model with improved capabilities")
        self.assertEqual(info["context_length"], 128000)
        self.assertEqual(info["training_data"], "Up to Apr 2023")


class TestLLMManager(unittest.TestCase):
    """Test cases for the LLMManager class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.manager = LLMManager()
    
    def test_initialization(self) -> None:
        """Test LLMManager initialization."""
        self.assertIsInstance(self.manager._llm_instances, dict)
        self.assertIsInstance(self.manager._llm_classes, dict)
        self.assertIsInstance(self.manager._model_configs, dict)
        
        # Should have built-in LLMs registered
        self.assertIn("openai", self.manager._llm_classes)
        self.assertIn("gpt-3.5-turbo", self.manager._llm_classes)
        self.assertIn("gpt-4", self.manager._llm_classes)
        self.assertIn("gpt-4-1106-preview", self.manager._llm_classes)
    
    def test_register_llm_class(self) -> None:
        """Test registering an LLM class."""
        self.manager.register_llm_class("test-mock", MockLLM)
        
        self.assertIn("test-mock", self.manager._llm_classes)
        self.assertEqual(self.manager._llm_classes["test-mock"], MockLLM)
    
    def test_register_invalid_llm_class(self) -> None:
        """Test registering an invalid LLM class."""
        with self.assertRaises(TypeError):
            self.manager.register_llm_class("invalid", str)  # Not a valid LLM class
    
    def test_create_llm(self) -> None:
        """Test creating an LLM instance."""
        llm = self.manager.create_llm("test-mock", "test-mock", {"param": "value"})
        
        self.assertIsInstance(llm, MockLLM)
        self.assertEqual(llm.model_name, "test-mock")
        self.assertEqual(llm.config, {"param": "value"})
    
    def test_create_llm_with_unknown_provider(self) -> None:
        """Test creating LLM with unknown provider."""
        with self.assertRaises(ValueError):
            self.manager.create_llm("test-model", "unknown-provider")
    
    def test_create_llm_infer_provider(self) -> None:
        """Test creating LLM with provider inference."""
        # Should infer OpenAI provider for GPT models
        llm = self.manager.create_llm("gpt-3.5-turbo")
        
        # This would normally create an OpenAI LLM, but we'll just check that it doesn't error
        self.assertIsNotNone(llm)
    
    def test_add_model(self) -> None:
        """Test adding a model to the manager."""
        self.manager.add_model("test-model", "test-mock", {"param": "value"})
        
        self.assertIn("test-model", self.manager._llm_instances)
        self.assertIn("test-model", self.manager._model_configs)
        
        llm = self.manager.get_model("test-model")
        self.assertIsInstance(llm, MockLLM)
        self.assertEqual(llm.model_name, "test-model")
    
    def test_add_duplicate_model(self) -> None:
        """Test adding a duplicate model."""
        self.manager.add_model("test-model", "test-mock", {"param": "value"})
        
        # Should replace existing model
        self.manager.add_model("test-model", "test-mock", {"new_param": "new_value"})
        
        llm = self.manager.get_model("test-model")
        self.assertEqual(llm.config, {"new_param": "new_value"})
    
    def test_remove_model(self) -> None:
        """Test removing a model."""
        self.manager.add_model("test-model", "test-mock", {"param": "value"})
        
        self.manager.remove_model("test-model")
        
        self.assertNotIn("test-model", self.manager._llm_instances)
        self.assertNotIn("test-model", self.manager._model_configs)
    
    def test_remove_nonexistent_model(self) -> None:
        """Test removing a nonexistent model."""
        with self.assertRaises(ValueError):
            self.manager.remove_model("nonexistent-model")
    
    def test_get_model(self) -> None:
        """Test getting a model."""
        self.manager.add_model("test-model", "test-mock", {"param": "value"})
        
        llm = self.manager.get_model("test-model")
        self.assertIsInstance(llm, MockLLM)
        self.assertEqual(llm.model_name, "test-model")
    
    def test_get_nonexistent_model(self) -> None:
        """Test getting a nonexistent model."""
        with self.assertRaises(ValueError):
            self.manager.get_model("nonexistent-model")
    
    def test_list_models(self) -> None:
        """Test listing models."""
        self.assertEqual(len(self.manager.list_models()), 0)
        
        self.manager.add_model("model1", "test-mock")
        self.manager.add_model("model2", "test-mock")
        
        models = self.manager.list_models()
        self.assertEqual(len(models), 2)
        self.assertIn("model1", models)
        self.assertIn("model2", models)
    
    def test_get_model_config(self) -> None:
        """Test getting model configuration."""
        config = {"param": "value"}
        self.manager.add_model("test-model", "test-mock", config)
        
        retrieved_config = self.manager.get_model_config("test-model")
        self.assertEqual(retrieved_config, config)
    
    def test_update_model_config(self) -> None:
        """Test updating model configuration."""
        self.manager.add_model("test-model", "test-mock", {"param": "value"})
        
        self.manager.update_model_config("test-model", {"new_param": "new_value"})
        
        config = self.manager.get_model_config("test-model")
        self.assertEqual(config, {"param": "value", "new_param": "new_value"})
    
    def test_get_available_providers(self) -> None:
        """Test getting available providers."""
        providers = self.manager.get_available_providers()
        
        self.assertIsInstance(providers, list)
        self.assertIn("openai", providers)
        self.assertIn("gpt-3.5-turbo", providers)
        self.assertIn("gpt-4", providers)
        self.assertIn("gpt-4-1106-preview", providers)
    
    def test_get_provider_info(self) -> None:
        """Test getting provider information."""
        # Register our mock class
        self.manager.register_llm_class("test-mock", MockLLM)
        
        info = self.manager.get_provider_info("test-mock")
        
        self.assertEqual(info["name"], "test-mock")
        self.assertEqual(info["class"], "MockLLM")
        self.assertIn("description", info)
    
    def test_load_models_from_config(self) -> None:
        """Test loading models from configuration."""
        self.manager.register_llm_class("test-mock", MockLLM)
        
        models_config = [
            {
                "name": "model1",
                "provider": "test-mock",
                "config": {"param": "value1"}
            },
            {
                "name": "model2",
                "provider": "test-mock",
                "config": {"param": "value2"}
            }
        ]
        
        self.manager.load_models_from_config(models_config)
        
        self.assertEqual(len(self.manager.list_models()), 2)
        self.assertIn("model1", self.manager._llm_instances)
        self.assertIn("model2", self.manager._llm_instances)
    
    def test_load_models_from_invalid_config(self) -> None:
        """Test loading models from invalid configuration."""
        # Invalid config type
        with self.assertRaises(ValueError):
            self.manager.load_models_from_config(["not a dict"])
        
        # Missing name
        with self.assertRaises(ValueError):
            self.manager.load_models_from_config([{"provider": "test-mock"}])


class TestModelRegistry(unittest.TestCase):
    """Test cases for the ModelRegistry class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.registry = ModelRegistry()
    
    def test_initialization(self) -> None:
        """Test ModelRegistry initialization."""
        self.assertIsInstance(self.registry._models, dict)
        
        # Should have built-in models registered
        models = self.registry.get_available_models()
        self.assertGreater(len(models), 0)
        
        model_names = [model["name"] for model in models]
        self.assertIn("gpt-3.5-turbo", model_names)
        self.assertIn("gpt-4", model_names)
        self.assertIn("gpt-4-1106-preview", model_names)
    
    def test_register_model(self) -> None:
        """Test registering a model."""
        self.registry.register_model(
            name="test-model",
            provider="test-provider",
            description="Test model",
            capabilities=["text-generation"],
            context_length=4096,
            parameters={"temperature": 0.7}
        )
        
        self.assertIn("test-model", self.registry._models)
        
        info = self.registry.get_model_info("test-model")
        self.assertEqual(info["name"], "test-model")
        self.assertEqual(info["provider"], "test-provider")
        self.assertEqual(info["description"], "Test model")
    
    def test_get_model_info(self) -> None:
        """Test getting model information."""
        info = self.registry.get_model_info("gpt-3.5-turbo")
        
        self.assertEqual(info["name"], "gpt-3.5-turbo")
        self.assertEqual(info["provider"], "openai")
        self.assertIn("description", info)
        self.assertIn("capabilities", info)
        self.assertIn("context_length", info)
        self.assertIn("parameters", info)
    
    def test_get_nonexistent_model_info(self) -> None:
        """Test getting information for nonexistent model."""
        with self.assertRaises(ValueError):
            self.registry.get_model_info("nonexistent-model")
    
    def test_get_available_models(self) -> None:
        """Test getting all available models."""
        models = self.registry.get_available_models()
        
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        
        for model in models:
            self.assertIn("name", model)
            self.assertIn("provider", model)
            self.assertIn("description", model)
    
    def test_get_models_by_provider(self) -> None:
        """Test getting models by provider."""
        openai_models = self.registry.get_models_by_provider("openai")
        
        self.assertIsInstance(openai_models, list)
        self.assertGreater(len(openai_models), 0)
        
        for model in openai_models:
            self.assertEqual(model["provider"], "openai")
    
    def test_get_models_by_capability(self) -> None:
        """Test getting models by capability."""
        code_models = self.registry.get_models_by_capability("code-generation")
        
        self.assertIsInstance(code_models, list)
        self.assertGreater(len(code_models), 0)
        
        for model in code_models:
            self.assertIn("code-generation", model["capabilities"])
    
    def test_search_models(self) -> None:
        """Test searching models."""
        # Search for GPT models
        results = self.registry.search_models("gpt")
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for model in results:
            self.assertIn("gpt", model["name"].lower())
        
        # Search for non-existent term
        results = self.registry.search_models("nonexistent")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()