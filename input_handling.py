#!/usr/bin/env python3
"""
Input Handling Module for NLP-Based Code Debugger and Solution Recommender

This module handles various input formats and sources:
1. Direct text input (code snippets and error logs)
2. File uploads (source code files, log files)
3. IDE integration inputs
4. Batch processing of multiple files
5. Input validation and preprocessing

Implements Stage 1 of the project pipeline as specified in the README.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import mimetypes

class InputHandler:
    """
    Handles different types of input for the code debugger system.
    Validates, preprocesses, and structures input data for analysis.
    """
    
    def __init__(self):
        """Initialize the input handler with supported file types and configurations."""
        self.supported_code_extensions = {
            '.py': 'Python',
            '.java': 'Java', 
            '.cpp': 'C++',
            '.cc': 'C++',
            '.cxx': 'C++',
            '.c': 'C',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.kt': 'Kotlin',
            '.swift': 'Swift'
        }
        
        self.supported_log_extensions = {
            '.log': 'log',
            '.txt': 'text',
            '.err': 'error',
            '.out': 'output'
        }
        
        self.max_file_size = 10 * 1024 * 1024  # 10MB limit
        self.max_line_length = 10000  # Maximum characters per line
        
    def process_text_input(self, code_text: str, error_text: str = "", 
                          source_info: Dict = None) -> Dict:
        """
        Process direct text input (code snippet and error log).
        
        Args:
            code_text (str): The source code snippet
            error_text (str): The error log/message (optional)
            source_info (dict): Additional metadata about the source
            
        Returns:
            dict: Structured input data ready for analysis
        """
        if not code_text or not isinstance(code_text, str):
            raise ValueError("Code text must be a non-empty string")
            
        # Clean and validate input
        code_text = self._clean_text(code_text)
        error_text = self._clean_text(error_text) if error_text else ""
        
        # Detect potential issues
        validation_issues = self._validate_text_input(code_text, error_text)
        
        # Structure the input data
        input_data = {
            "type": "text_input",
            "timestamp": self._get_timestamp(),
            "source_info": source_info or {},
            "content": {
                "code": {
                    "text": code_text,
                    "lines": code_text.splitlines(),
                    "line_count": len(code_text.splitlines()),
                    "char_count": len(code_text)
                },
                "error": {
                    "text": error_text,
                    "lines": error_text.splitlines() if error_text else [],
                    "line_count": len(error_text.splitlines()) if error_text else 0
                }
            },
            "validation": {
                "is_valid": len(validation_issues) == 0,
                "issues": validation_issues
            },
            "metadata": {
                "detected_language": self._detect_language_from_code(code_text),
                "has_error_log": bool(error_text),
                "estimated_complexity": self._estimate_complexity(code_text)
            }
        }
        
        return input_data
        
    def process_file_input(self, file_path: str, error_file_path: str = "") -> Dict:
        """
        Process file input (code file and optional error log file).
        
        Args:
            file_path (str): Path to the source code file
            error_file_path (str): Path to the error log file (optional)
            
        Returns:
            dict: Structured input data ready for analysis
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")
            
        if error_file_path and not os.path.exists(error_file_path):
            raise FileNotFoundError(f"Error file not found: {error_file_path}")
            
        # Read and validate files
        file_info = self._get_file_info(file_path)
        code_content = self._read_file_safely(file_path)
        
        error_content = ""
        error_file_info = {}
        if error_file_path:
            error_file_info = self._get_file_info(error_file_path)
            error_content = self._read_file_safely(error_file_path)
            
        # Process the content
        return self.process_text_input(
            code_text=code_content,
            error_text=error_content,
            source_info={
                "source_file": file_info,
                "error_file": error_file_info
            }
        )
        
    def process_batch_input(self, input_config: Dict) -> List[Dict]:
        """
        Process multiple files or inputs in batch mode.
        
        Args:
            input_config (dict): Configuration for batch processing
                {
                    "source_directory": "path/to/code/files",
                    "error_directory": "path/to/error/logs", (optional)
                    "file_patterns": ["*.py", "*.java"], (optional)
                    "recursive": True, (optional)
                    "max_files": 100 (optional)
                }
                
        Returns:
            list: List of structured input data for each processed file
        """
        source_dir = input_config.get("source_directory", "")
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
        error_dir = input_config.get("error_directory", "")
        file_patterns = input_config.get("file_patterns", ["*.py", "*.java", "*.cpp"])
        recursive = input_config.get("recursive", True)
        max_files = input_config.get("max_files", 100)
        
        # Find matching files
        source_files = self._find_matching_files(source_dir, file_patterns, recursive)
        source_files = source_files[:max_files]  # Limit the number of files
        
        results = []
        for source_file in source_files:
            try:
                # Look for corresponding error file
                error_file = ""
                if error_dir:
                    error_file = self._find_error_file(source_file, error_dir)
                    
                # Process the file
                result = self.process_file_input(source_file, error_file)
                result["batch_info"] = {
                    "batch_index": len(results),
                    "total_files": len(source_files),
                    "source_file": source_file,
                    "error_file": error_file
                }
                results.append(result)
                
            except Exception as e:
                # Add error entry for failed files
                results.append({
                    "type": "batch_error",
                    "file_path": source_file,
                    "error": str(e),
                    "timestamp": self._get_timestamp()
                })
                
        return results
        
    def process_ide_input(self, ide_data: Dict) -> Dict:
        """
        Process input from IDE integration (VS Code, IntelliJ, etc.).
        
        Args:
            ide_data (dict): Data from IDE extension/plugin
                {
                    "editor_content": "source code",
                    "error_output": "error messages",
                    "cursor_position": {"line": 10, "column": 5},
                    "selected_text": "highlighted code",
                    "file_path": "current file path",
                    "workspace_info": {"root": "...", "language": "..."}
                }
                
        Returns:
            dict: Structured input data ready for analysis
        """
        editor_content = ide_data.get("editor_content", "")
        error_output = ide_data.get("error_output", "")
        cursor_pos = ide_data.get("cursor_position", {})
        selected_text = ide_data.get("selected_text", "")
        file_path = ide_data.get("file_path", "")
        workspace_info = ide_data.get("workspace_info", {})
        
        # Determine what code to analyze
        code_to_analyze = selected_text if selected_text else editor_content
        
        # Process the input
        result = self.process_text_input(
            code_text=code_to_analyze,
            error_text=error_output,
            source_info={
                "ide_integration": True,
                "file_path": file_path,
                "cursor_position": cursor_pos,
                "has_selection": bool(selected_text),
                "workspace": workspace_info
            }
        )
        
        # Add IDE-specific metadata
        result["ide_context"] = {
            "full_editor_content": editor_content,
            "analyzed_portion": "selection" if selected_text else "full_content",
            "cursor_line": cursor_pos.get("line", 0),
            "workspace_language": workspace_info.get("language", "")
        }
        
        return result
        
    def validate_input_format(self, input_data: Union[str, Dict]) -> Tuple[bool, List[str]]:
        """
        Validate input format and return validation results.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            tuple: (is_valid, list_of_issues)
        """
        issues = []
        
        if isinstance(input_data, str):
            # Simple string validation
            if not input_data.strip():
                issues.append("Input text is empty")
            elif len(input_data) > self.max_file_size:
                issues.append(f"Input text exceeds maximum size ({self.max_file_size} bytes)")
        elif isinstance(input_data, dict):
            # Dictionary validation
            required_fields = ["type", "content"]
            for field in required_fields:
                if field not in input_data:
                    issues.append(f"Missing required field: {field}")
        else:
            issues.append("Invalid input format. Expected string or dictionary")
            
        return len(issues) == 0, issues
        
    # Private helper methods
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text input."""
        if not text:
            return ""
            
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Limit line length
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if len(line) > self.max_line_length:
                line = line[:self.max_line_length] + "... [truncated]"
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)
        
    def _validate_text_input(self, code_text: str, error_text: str) -> List[str]:
        """Validate text input and return list of issues."""
        issues = []
        
        if not code_text.strip():
            issues.append("Code text is empty")
            
        if len(code_text) > self.max_file_size:
            issues.append(f"Code text exceeds maximum size")
            
        if error_text and len(error_text) > self.max_file_size:
            issues.append(f"Error text exceeds maximum size")
            
        # Check for binary content
        if '\x00' in code_text:
            issues.append("Code appears to contain binary data")
            
        return issues
        
    def _detect_language_from_code(self, code_text: str) -> str:
        """Simple language detection based on code patterns."""
        # Basic pattern matching for language detection
        if re.search(r'\bdef\s+\w+\s*\(', code_text):
            return "Python"
        elif re.search(r'\bpublic\s+class\s+\w+', code_text):
            return "Java"
        elif re.search(r'#include\s*<.*>', code_text):
            return "C++"
        elif re.search(r'\bfunction\s+\w+\s*\(', code_text):
            return "JavaScript"
        else:
            return "Unknown"
            
    def _estimate_complexity(self, code_text: str) -> str:
        """Estimate code complexity based on simple metrics."""
        lines = len(code_text.splitlines())
        if lines < 10:
            return "Low"
        elif lines < 50:
            return "Medium"
        else:
            return "High"
            
    def _get_file_info(self, file_path: str) -> Dict:
        """Get file information including size, extension, etc."""
        path_obj = Path(file_path)
        stat = path_obj.stat()
        
        return {
            "path": str(path_obj.absolute()),
            "name": path_obj.name,
            "extension": path_obj.suffix,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "detected_language": self.supported_code_extensions.get(path_obj.suffix, "Unknown")
        }
        
    def _read_file_safely(self, file_path: str) -> str:
        """Read file content safely with error handling."""
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise ValueError(f"File too large: {file_size} bytes")
                
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
                    
            raise ValueError("Could not decode file with any supported encoding")
            
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {str(e)}")
            
    def _find_matching_files(self, directory: str, patterns: List[str], recursive: bool) -> List[str]:
        """Find files matching the given patterns."""
        import fnmatch
        
        matches = []
        search_pattern = "**/*" if recursive else "*"
        
        for pattern in patterns:
            for file_path in Path(directory).glob(search_pattern):
                if file_path.is_file() and fnmatch.fnmatch(file_path.name, pattern):
                    matches.append(str(file_path))
                    
        return list(set(matches))  # Remove duplicates
        
    def _find_error_file(self, source_file: str, error_dir: str) -> str:
        """Find corresponding error file for a source file."""
        source_path = Path(source_file)
        base_name = source_path.stem
        
        # Look for files with same base name but different extensions
        for ext in self.supported_log_extensions:
            error_file = Path(error_dir) / f"{base_name}{ext}"
            if error_file.exists():
                return str(error_file)
                
        return ""
        
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()


# Example usage and testing
if __name__ == "__main__":
    handler = InputHandler()
    
    # Test 1: Text input
    print("=== Testing Text Input ===")
    sample_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

result = calculate_sum([1, 2, 3, undefined_var])
print(result)
"""
    
    sample_error = "NameError: name 'undefined_var' is not defined at line 8"
    
    try:
        result = handler.process_text_input(sample_code, sample_error)
        print("✅ Text input processed successfully")
        print(f"Detected language: {result['metadata']['detected_language']}")
        print(f"Validation status: {result['validation']['is_valid']}")
        print(f"Code lines: {result['content']['code']['line_count']}")
    except Exception as e:
        print(f"❌ Error processing text input: {e}")
    
    # Test 2: File validation
    print("\n=== Testing File Input Validation ===")
    test_files = ["test.py", "nonexistent.java"]
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                result = handler.process_file_input(test_file)
                print(f"✅ File {test_file} processed successfully")
            except Exception as e:
                print(f"❌ Error processing {test_file}: {e}")
        else:
            print(f"⚠️  File {test_file} does not exist (expected for testing)")
    
    # Test 3: Input validation
    print("\n=== Testing Input Validation ===")
    test_inputs = [
        ("Valid code", True),
        ("", False),
        ({"type": "test", "content": "code"}, True),
        ({"missing": "fields"}, False)
    ]
    
    for test_input, expected_valid in test_inputs:
        is_valid, issues = handler.validate_input_format(test_input)
        status = "✅" if is_valid == expected_valid else "❌"
        print(f"{status} Input validation: {is_valid} (expected: {expected_valid})")
        if issues:
            print(f"   Issues: {issues}")