#!/usr/bin/env python3
"""
Input Handling Module - Processes various input formats for the NLP-based code debugger
Handles code files, error logs, and user queries in different formats
Implements Stage 1 of the project pipeline as specified in the README.
"""

import os
import re
import json
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
        
        self.error_log_patterns = {
            'traceback': r'Traceback \(most recent call last\):',
            'exception': r'(\w+Error|\w+Exception):',
            'line_number': r'line (\d+)',
            'file_reference': r'File "([^"]+)"',
            'java_exception': r'Exception in thread "main" (\w+):',
            'compilation_error': r'error: (.+)'
        }
        
        self.max_file_size = 10 * 1024 * 1024  # 10MB limit
        self.max_line_length = 10000  # Maximum characters per line    def detect_input_type(self, input_data: str) -> str:
        """Detect whether input is code, error log, or mixed content"""
        input_lower = input_data.lower().strip()
        
        # Check for error log indicators
        error_indicators = [
            'traceback', 'exception', 'error:', 'at line', 'line:', 
            'failed', 'nullpointerexception', 'indexerror', 'typeerror'
        ]
        
        # Check for code indicators
        code_indicators = [
            'def ', 'class ', 'function ', 'public class', 'import ', 
            'from ', 'include ', 'using ', '#include'
        ]
        
        # Check for mixed content (code + error) first
        if '\n' in input_data and len(input_data.split('\n')) > 3:
            lines = input_data.split('\n')
            has_code = any(any(ind in line.lower() for ind in code_indicators) for line in lines)
            has_error = any(any(ind in line.lower() for ind in error_indicators) for line in lines)
            
            if has_code and has_error:
                return 'mixed'
        
        # Then check individual types
        if any(indicator in input_lower for indicator in error_indicators):
            return 'error_log'
        
        if any(indicator in input_lower for indicator in code_indicators):
            return 'code'
        
        return 'unknown'

    def parse_mixed_input(self, input_data: str) -> Dict[str, str]:
        """Parse input that contains both code and error information"""
        lines = input_data.strip().split('\n')
        
        code_lines = []
        error_lines = []
        
        in_traceback = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if we're in a traceback section
            if 'traceback' in line_lower or in_traceback:
                in_traceback = True
                error_lines.append(line)
                # Traceback usually ends with the final exception line
                if re.search(r'\w+Error|\w+Exception', line) and not line.strip().startswith('File'):
                    in_traceback = False
            # Check for other error indicators
            elif any(indicator in line_lower for indicator in ['error:', 'exception:', 'failed', 'at line']):
                error_lines.append(line)
            # Otherwise treat as code
            elif line.strip():  # Skip empty lines
                code_lines.append(line)
        
        return {
            "code": '\n'.join(code_lines) if code_lines else "",
            "error_log": '\n'.join(error_lines) if error_lines else "",
            "input_type": "mixed"
        }

    def process_text_input(self, code_text: str, error_text: str = "", 
                          source_info: Optional[Dict] = None) -> Dict:
        """
        Process direct text input (code snippet and error log).
        
        Args:
            code_text (str): The source code snippet
            error_text (str): The error log/message (optional)
            source_info (dict): Additional metadata about the source
            
        Returns:
            dict: Structured input data ready for analysis
        """
        if source_info is None:
            source_info = {}
            
        # Clean and validate inputs
        code_text = self._clean_text_input(code_text)
        error_text = self._clean_text_input(error_text)
        
        # Detect language from code
        language = self._detect_language_from_code(code_text)
        
        # Validate inputs
        validation = self._validate_text_input(code_text, error_text)
        
        return {
            "input_type": "text",
            "code": code_text,
            "error_log": error_text,
            "language": language,
            "source_info": source_info,
            "validation": validation,
            "metadata": {
                "code_lines": len(code_text.split('\n')) if code_text else 0,
                "error_lines": len(error_text.split('\n')) if error_text else 0,
                "timestamp": self._get_timestamp()
            }
        }

    def process_file_input(self, file_path: str, error_file_path: str = "") -> Dict:
        """
        Process file input (code file and optional error log file).
        
        Args:
            file_path (str): Path to the source code file
            error_file_path (str): Path to the error log file (optional)
            
        Returns:
            dict: Structured input data ready for analysis
        """
        try:
            # Read and validate code file
            code_content = self._read_file_safely(file_path)
            language = self._detect_language_from_extension(file_path)
            
            # Read error file if provided
            error_content = ""
            if error_file_path and os.path.exists(error_file_path):
                error_content = self._read_file_safely(error_file_path)
            
            # Get file metadata
            file_info = {
                "file_path": os.path.abspath(file_path),
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "file_extension": os.path.splitext(file_path)[1]
            }
            
            if error_file_path:
                file_info["error_file_path"] = os.path.abspath(error_file_path)
            
            # Validate inputs
            validation = self._validate_text_input(code_content, error_content)
            
            return {
                "input_type": "file",
                "code": code_content,
                "error_log": error_content,
                "language": language,
                "source_info": file_info,
                "validation": validation,
                "metadata": {
                    "code_lines": len(code_content.split('\n')),
                    "error_lines": len(error_content.split('\n')) if error_content else 0,
                    "timestamp": self._get_timestamp()
                }
            }
            
        except Exception as e:
            return {
                "input_type": "file",
                "error": f"Failed to process file: {str(e)}",
                "validation": {"is_valid": False, "issues": [str(e)]}
            }

    def process_batch_input(self, input_config: Dict) -> List[Dict]:
        """
        Process multiple files or inputs in batch mode.
        
        Args:
            input_config (dict): Configuration for batch processing
                {
                    "source_directory": "path/to/source/files",
                    "error_directory": "path/to/error/logs", (optional)
                    "file_patterns": ["*.py", "*.java"], (optional)
                    "recursive": True (optional)
                }
                
        Returns:
            list: List of structured input data for each processed file
        """
        results = []
        source_dir = input_config.get("source_directory", "")
        error_dir = input_config.get("error_directory", "")
        patterns = input_config.get("file_patterns", ["*"])
        recursive = input_config.get("recursive", False)
        
        if not os.path.exists(source_dir):
            return [{"error": f"Source directory not found: {source_dir}"}]
        
        try:
            # Find matching files
            source_files = self._find_files(source_dir, patterns, recursive)
            
            for file_path in source_files:
                # Try to find corresponding error file
                error_file = self._find_error_file(file_path, error_dir)
                
                # Process the file pair
                result = self.process_file_input(file_path, error_file)
                results.append(result)
                
        except Exception as e:
            results.append({"error": f"Batch processing failed: {str(e)}"})
        
        return results

    def process_ide_input(self, ide_data: Dict) -> Dict:
        """
        Process input from IDE integration (VS Code, IntelliJ, etc.).
        
        Args:
            ide_data (dict): Data from IDE extension/plugin
                {
                    "code": "source code content",
                    "error_log": "error message from IDE",
                    "file_path": "current file path",
                    "cursor_position": {"line": 42, "column": 15},
                    "language": "python",
                    "project_root": "/path/to/project"
                }
                
        Returns:
            dict: Structured input data ready for analysis
        """
        code_text = ide_data.get("code", "")
        error_text = ide_data.get("error_log", "")
        
        source_info = {
            "ide_source": True,
            "file_path": ide_data.get("file_path", ""),
            "cursor_position": ide_data.get("cursor_position", {}),
            "project_root": ide_data.get("project_root", ""),
            "language_hint": ide_data.get("language", "")
        }
        
        # Use IDE's language hint if available, otherwise detect
        language = ide_data.get("language", "") or self._detect_language_from_code(code_text)
        
        validation = self._validate_text_input(code_text, error_text)
        
        return {
            "input_type": "ide",
            "code": code_text,
            "error_log": error_text,
            "language": language,
            "source_info": source_info,
            "validation": validation,
            "metadata": {
                "code_lines": len(code_text.split('\n')) if code_text else 0,
                "error_lines": len(error_text.split('\n')) if error_text else 0,
                "timestamp": self._get_timestamp()
            }
        }

    def validate_input(self, code: str, error_log: str) -> Dict[str, Union[bool, str, List[str]]]:
        """Validate that the provided code and error log are usable"""
        issues = []
        
        # Validate code
        if not code or len(code.strip()) < 10:
            issues.append("Code input is too short or empty")
        
        # Validate error log
        if not error_log or len(error_log.strip()) < 5:
            issues.append("Error log is too short or empty")
        
        # Check if error log actually contains error information
        if error_log and not re.search(r'\w+(?:Error|Exception)', error_log):
            issues.append("Error log does not appear to contain valid error information")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "suggestion": "Provide both valid code and a complete error log" if issues else "Input validation passed"
        }

    def format_input_for_analysis(self, raw_input: str) -> Dict[str, str]:
        """Format and prepare input for the analysis pipeline"""
        input_type = self.detect_input_type(raw_input)
        
        if input_type == 'mixed':
            parsed = self.parse_mixed_input(raw_input)
            return {
                "code": parsed["code"],
                "error_log": parsed["error_log"],
                "input_type": input_type,
                "status": "success"
            }
        elif input_type == 'code':
            return {
                "code": raw_input,
                "error_log": "",
                "input_type": input_type,
                "status": "needs_error_log"
            }
        elif input_type == 'error_log':
            return {
                "code": "",
                "error_log": raw_input,
                "input_type": input_type,
                "status": "needs_code"
            }
        else:
            return {
                "code": "",
                "error_log": "",
                "input_type": "unknown",
                "status": "unrecognized_input"
            }

    # Helper methods
    def _clean_text_input(self, text: str) -> str:
        """Clean and normalize text input."""
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving code structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Keep original indentation but remove trailing whitespace
            cleaned_line = line.rstrip()
            if len(cleaned_line) <= self.max_line_length:
                cleaned_lines.append(cleaned_line)
            else:
                # Truncate overly long lines
                cleaned_lines.append(cleaned_line[:self.max_line_length] + "... [truncated]")
        
        return '\n'.join(cleaned_lines)

    def _validate_text_input(self, code: str, error_log: str) -> Dict:
        """Validate text input and return list of issues."""
        issues = []
        
        if not code or len(code.strip()) < 5:
            issues.append("Code input is too short or empty")
        
        if not error_log or len(error_log.strip()) < 3:
            issues.append("Error log is missing or too short")
        
        # Check for valid error patterns
        if error_log and not any(re.search(pattern, error_log) for pattern in self.error_log_patterns.values()):
            issues.append("Error log does not contain recognizable error patterns")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }

    def _detect_language_from_code(self, code: str) -> str:
        """Detect programming language from code content."""
        if not code:
            return "unknown"
        
        code_lower = code.lower()
        
        # Language-specific patterns
        if any(keyword in code_lower for keyword in ['def ', 'import ', 'from ', 'print(']):
            return "Python"
        elif any(keyword in code_lower for keyword in ['public class', 'public static void', 'system.out']):
            return "Java"
        elif any(keyword in code_lower for keyword in ['#include', 'int main', 'cout']):
            return "C++"
        elif any(keyword in code_lower for keyword in ['function', 'console.log', 'var ', 'let ', 'const ']):
            return "JavaScript"
        elif any(keyword in code_lower for keyword in ['using system', 'console.writeline', 'public class']):
            return "C#"
        
        return "unknown"

    def _detect_language_from_extension(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        return self.supported_code_extensions.get(ext, "unknown")

    def _read_file_safely(self, file_path: str) -> str:
        """Safely read file content with size and encoding checks."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file with any supported encoding: {file_path}")

    def _find_files(self, directory: str, patterns: List[str], recursive: bool) -> List[str]:
        """Find files matching patterns in directory."""
        import glob
        
        files = []
        search_pattern = "**/" if recursive else ""
        
        for pattern in patterns:
            full_pattern = os.path.join(directory, search_pattern + pattern)
            files.extend(glob.glob(full_pattern, recursive=recursive))
        
        # Filter for supported code files
        supported_files = []
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in self.supported_code_extensions:
                supported_files.append(file_path)
        
        return supported_files

    def _find_error_file(self, code_file: str, error_dir: str) -> str:
        """Find corresponding error file for a code file."""
        if not error_dir or not os.path.exists(error_dir):
            return ""
        
        base_name = os.path.splitext(os.path.basename(code_file))[0]
        
        # Common error file patterns
        error_patterns = [
            f"{base_name}.log",
            f"{base_name}.err",
            f"{base_name}_error.txt",
            f"{base_name}.out"
        ]
        
        for pattern in error_patterns:
            error_file = os.path.join(error_dir, pattern)
            if os.path.exists(error_file):
                return error_file
        
        return ""

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


# Example usage and testing
if __name__ == "__main__":
    print("=== TESTING INPUT HANDLER ===")
    handler = InputHandler()
    
    # Test Case 1: Mixed input (code + error)
    print("\n--- Test Case 1: Mixed Input ---")
    mixed_input = """
def process_list(items):
    for i in range(10):
        print(items[i])

Traceback (most recent call last):
  File "test.py", line 3, in process_list
    print(items[i])
IndexError: list index out of range
"""
    result1 = handler.format_input_for_analysis(mixed_input)
    print(f"Input Type: {result1['input_type']}")
    print(f"Code:\n{result1['code']}")
    print(f"Error Log:\n{result1['error_log']}")
    
    # Test Case 2: Code only
    print("\n--- Test Case 2: Code Only ---")
    code_only = """
def add_values(x, y):
    result = x + y
    return result
"""
    result2 = handler.format_input_for_analysis(code_only)
    print(f"Input Type: {result2['input_type']}")
    print(f"Status: {result2['status']}")
    
    # Test Case 3: Input validation
    print("\n--- Test Case 3: Input Validation ---")
    validation = handler.validate_input(result1['code'], result1['error_log'])
    print(f"Is Valid: {validation['is_valid']}")
    print(f"Issues: {validation['issues']}")
    
    # Test Case 4: Text input processing
    print("\n--- Test Case 4: Text Input Processing ---")
    text_result = handler.process_text_input(
        code_text="def calculate_sum(numbers):\n    return sum(numbers)",
        error_text="TypeError: unsupported operand type(s) for +: 'int' and 'str'"
    )
    print(f"Language detected: {text_result['language']}")
    print(f"Validation: {text_result['validation']}")
    print(f"Metadata: {text_result['metadata']}")
