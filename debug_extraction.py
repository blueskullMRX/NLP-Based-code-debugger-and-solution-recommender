#!/usr/bin/env python3
"""
Debug the variable extraction to see what's happening
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from static_code_analysis import StaticCodeAnalyzer

def debug_variable_extraction():
    """Debug variable extraction step by step."""
    print("=== Debugging Variable Extraction ===")
    
    analyzer = StaticCodeAnalyzer()
    
    # Test the method directly
    test_cases = [
        ("user_name = input('Enter name: ')", "Python"),
        ("result = data_list[index] + calculation", "Python"),
        ("String result = user.getName().toLowerCase();", "Java"),
        ("int data = my_vector[index] + offset;", "C++")
    ]
    
    for code_line, language in test_cases:
        print(f"\nTesting: {code_line}")
        print(f"Language: {language}")
        
        try:
            variables = analyzer.extract_variables_from_code_line(code_line, language)
            print(f"Extracted variables: {variables}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_variable_extraction()
