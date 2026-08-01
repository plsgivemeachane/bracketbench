"""
Simple scoring module for BracketBench.

This module provides basic scoring functionality that takes generated output
and returns a score point.
"""

import difflib
from typing import Optional, Callable, Dict, Any


class ScoringEngine:
    """
    Simple scoring engine for benchmarking LLM outputs.
    
    This class provides basic scoring functionality to evaluate generated text
    against expected outputs.
    """
    
    def __init__(self) -> None:
        """Initialize the scoring engine."""
        self.scoring_methods: Dict[str, Callable] = {
            "exact_match": self._exact_match_score,
            "similarity": self._similarity_score,
            "length_ratio": self._length_ratio_score,
        }
    
    def score_output(
        self, 
        actual_output: str, 
        expected_output: Optional[str] = None,
        input_prompt: Optional[str] = None,
        method: str = "similarity"
    ) -> float:
        """
        Score the generated output.
        
        Args:
            actual_output: The generated text to score
            expected_output: The expected text (if available)
            input_prompt: The original input prompt (for context)
            method: Scoring method to use
            
        Returns:
            Score as a float between 0.0 and 1.0
            
        Raises:
            ValueError: If the scoring method is not available
        """
        if method not in self.scoring_methods:
            raise ValueError(f"Unknown scoring method: {method}. Available methods: {list(self.scoring_methods.keys())}")
        
        return self.scoring_methods[method](actual_output, expected_output, input_prompt)
    
    def _exact_match_score(
        self, 
        actual_output: str, 
        expected_output: Optional[str] = None,
        input_prompt: Optional[str] = None
    ) -> float:
        """
        Calculate exact match score.
        
        Args:
            actual_output: The generated text
            expected_output: The expected text
            input_prompt: The original input prompt (unused)
            
        Returns:
            Score as a float (1.0 for exact match, 0.0 otherwise)
        """
        if expected_output is None:
            # If no expected output, return a neutral score
            return 0.5
        
        return 1.0 if actual_output.strip() == expected_output.strip() else 0.0
    
    def _similarity_score(
        self, 
        actual_output: str, 
        expected_output: Optional[str] = None,
        input_prompt: Optional[str] = None
    ) -> float:
        """
        Calculate similarity score using sequence matching.
        
        Args:
            actual_output: The generated text
            expected_output: The expected text
            input_prompt: The original input prompt (unused)
            
        Returns:
            Score as a float between 0.0 and 1.0
        """
        if expected_output is None:
            # If no expected output, score based on output length and content
            # This is a simple heuristic - longer, non-empty outputs get higher scores
            if not actual_output.strip():
                return 0.0
            return min(1.0, len(actual_output.strip()) / 100.0)
        
        # Use difflib to calculate similarity
        matcher = difflib.SequenceMatcher(None, actual_output.lower(), expected_output.lower())
        return matcher.ratio()
    
    def _length_ratio_score(
        self, 
        actual_output: str, 
        expected_output: Optional[str] = None,
        input_prompt: Optional[str] = None
    ) -> float:
        """
        Calculate score based on length ratio.
        
        Args:
            actual_output: The generated text
            expected_output: The expected text
            input_prompt: The original input prompt (unused)
            
        Returns:
            Score as a float between 0.0 and 1.0
        """
        if expected_output is None:
            # If no expected output, score based on reasonable length
            # This is a simple heuristic - outputs between 10 and 500 chars get good scores
            length = len(actual_output.strip())
            if length == 0:
                return 0.0
            elif 10 <= length <= 500:
                return 1.0
            elif length < 10:
                return length / 10.0
            else:
                return max(0.0, 1.0 - (length - 500) / 1000.0)
        
        # Calculate ratio of actual length to expected length
        actual_len = len(actual_output.strip())
        expected_len = len(expected_output.strip())
        
        if expected_len == 0:
            return 1.0 if actual_len == 0 else 0.0
        
        ratio = actual_len / expected_len
        
        # Score based on how close the ratio is to 1.0
        # Use a simple decay function for ratios far from 1.0
        return max(0.0, 1.0 - abs(ratio - 1.0))
    
    def register_scoring_method(self, name: str, scoring_function: Callable) -> None:
        """
        Register a custom scoring method.
        
        Args:
            name: Name of the scoring method
            scoring_function: Function that takes (actual_output, expected_output, input_prompt)
                             and returns a score (float)
        """
        self.scoring_methods[name] = scoring_function
    
    def get_available_methods(self) -> list:
        """
        Get list of available scoring methods.
        
        Returns:
            List of available scoring method names
        """
        return list(self.scoring_methods.keys())