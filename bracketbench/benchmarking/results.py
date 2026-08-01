"""
Basic results handling module for BracketBench.

This module provides simple functionality to save scores and optionally
timing information from benchmark runs.
"""

import json
import csv
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .models import BenchmarkResult


class ResultsHandler:
    """
    Simple results handler for benchmark results.
    
    This class provides functionality to save and export benchmark results,
    including scores and timing information.
    """
    
    def __init__(self) -> None:
        """Initialize the results handler."""
        pass
    
    def save_results(self, results: List[BenchmarkResult], filepath: str) -> None:
        """
        Save benchmark results to a file.
        
        Args:
            results: List of BenchmarkResult objects
            filepath: Path to save the results
            
        Raises:
            ValueError: If filepath is invalid
        """
        if not filepath:
            raise ValueError("Filepath is required")
        
        if not results:
            raise ValueError("No results to save")
        
        # Determine file format based on extension
        if filepath.endswith('.json'):
            self._save_as_json(results, filepath)
        elif filepath.endswith('.csv'):
            self._save_as_csv(results, filepath)
        else:
            # Default to JSON
            self._save_as_json(results, filepath)
    
    def _save_as_json(self, results: List[BenchmarkResult], filepath: str) -> None:
        """
        Save results as JSON.
        
        Args:
            results: List of BenchmarkResult objects
            filepath: Path to save the results
        """
        # Convert results to dictionaries
        results_dict = [asdict(result) for result in results]
        
        # Add summary information
        summary = self._calculate_summary(results)
        data = {
            "summary": summary,
            "results": results_dict
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_as_csv(self, results: List[BenchmarkResult], filepath: str) -> None:
        """
        Save results as CSV.
        
        Args:
            results: List of BenchmarkResult objects
            filepath: Path to save the results
        """
        if not results:
            return
        
        # Extract field names from the first result
        fieldnames = list(asdict(results[0]).keys())
        
        # Flatten metadata fields
        for result in results:
            for key, value in result.metadata.items():
                flat_key = f"metadata_{key}"
                if flat_key not in fieldnames:
                    fieldnames.append(flat_key)
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = asdict(result)
                
                # Flatten metadata
                metadata = row.pop('metadata', {})
                for key, value in metadata.items():
                    row[f"metadata_{key}"] = value
                
                writer.writerow(row)
    
    def _calculate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Calculate summary statistics for results.
        
        Args:
            results: List of BenchmarkResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        total_tests = len(results)
        total_score = sum(result.score for result in results)
        average_score = total_score / total_tests if total_tests > 0 else 0.0
        
        total_time = sum(result.execution_time for result in results)
        average_time = total_time / total_tests if total_tests > 0 else 0.0
        
        # Count tests with errors
        error_count = sum(1 for result in results if "error" in result.metadata)
        
        return {
            "total_tests": total_tests,
            "average_score": average_score,
            "total_score": total_score,
            "average_execution_time": average_time,
            "total_execution_time": total_time,
            "error_count": error_count,
            "success_rate": (total_tests - error_count) / total_tests if total_tests > 0 else 0.0
        }
    
    def load_results(self, filepath: str) -> List[BenchmarkResult]:
        """
        Load benchmark results from a file.
        
        Args:
            filepath: Path to load the results from
            
        Returns:
            List of BenchmarkResult objects
            
        Raises:
            ValueError: If filepath is invalid or file format is unsupported
            FileNotFoundError: If the file doesn't exist
        """
        if not filepath:
            raise ValueError("Filepath is required")
        
        # Determine file format based on extension
        if filepath.endswith('.json'):
            return self._load_from_json(filepath)
        elif filepath.endswith('.csv'):
            return self._load_from_csv(filepath)
        else:
            # Try JSON first
            try:
                return self._load_from_json(filepath)
            except:
                # Fall back to CSV
                return self._load_from_csv(filepath)
    
    def _load_from_json(self, filepath: str) -> List[BenchmarkResult]:
        """
        Load results from JSON file.
        
        Args:
            filepath: Path to load the results from
            
        Returns:
            List of BenchmarkResult objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats: with and without summary
        if "results" in data:
            results_data = data["results"]
        else:
            results_data = data
        
        # Convert dictionaries back to BenchmarkResult objects
        results = []
        for result_dict in results_data:
            result = BenchmarkResult(**result_dict)
            results.append(result)
        
        return results
    
    def _load_from_csv(self, filepath: str) -> List[BenchmarkResult]:
        """
        Load results from CSV file.
        
        Args:
            filepath: Path to load the results from
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Extract metadata fields
                metadata = {}
                metadata_keys = [key for key in row.keys() if key.startswith("metadata_")]
                
                for key in metadata_keys:
                    metadata_key = key[9:]  # Remove "metadata_" prefix
                    metadata[metadata_key] = row[key]
                    del row[key]
                
                # Convert string values back to appropriate types
                for key, value in row.items():
                    if key in ["execution_time", "score"]:
                        row[key] = float(value) if value else 0.0
                
                # Create BenchmarkResult object
                row["metadata"] = metadata
                result = BenchmarkResult(**row)
                results.append(result)
        
        return results
    
    def get_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Get summary statistics for results.
        
        Args:
            results: List of BenchmarkResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        return self._calculate_summary(results)