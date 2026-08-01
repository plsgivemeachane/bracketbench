"""
Test case management module for BracketBench.

This module provides the simplified TestCase class and related functionality for managing
benchmark test cases.
"""

import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """
    Simplified class to represent individual benchmark test cases.
    
    This class provides functionality to store and manage test case data,
    including input prompts and expected outputs.
    """
    
    id: str
    input_prompt: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Post-initialization processing."""
        # Generate ID if not provided
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def validate(self) -> bool:
        """
        Validate the test case data.
        
        Returns:
            True if the test case is valid, False otherwise
            
        Raises:
            ValueError: If the test case data is invalid
        """
        if not self.id:
            raise ValueError("Test case ID is required")
        
        if not self.input_prompt:
            raise ValueError("Input prompt is required")
        
        if not isinstance(self.input_prompt, str):
            raise ValueError("Input prompt must be a string")
        
        if self.expected_output is not None and not isinstance(self.expected_output, str):
            raise ValueError("Expected output must be a string or None")
        
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the test case to a dictionary.
        
        Returns:
            Dictionary representation of the test case
        """
        return {
            "id": self.id,
            "input_prompt": self.input_prompt,
            "expected_output": self.expected_output,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """
        Create a test case from a dictionary.
        
        Args:
            data: Dictionary representation of the test case
            
        Returns:
            TestCase object
            
        Raises:
            ValueError: If the data is invalid
        """
        # Create and validate the test case
        test_case = cls(**data)
        test_case.validate()
        
        return test_case
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update metadata for the test case.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def remove_metadata(self, key: str) -> bool:
        """
        Remove metadata from the test case.
        
        Args:
            key: Metadata key to remove
            
        Returns:
            True if metadata was removed, False if not found
        """
        if key in self.metadata:
            del self.metadata[key]
            return True
        return False
    
    def __str__(self) -> str:
        """String representation of the TestCase."""
        return f"TestCase(id='{self.id}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the TestCase."""
        return f"TestCase(id='{self.id}', input_prompt_length={len(self.input_prompt)})"