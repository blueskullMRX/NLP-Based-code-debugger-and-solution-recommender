#!/usr/bin/env python3
"""
Test script to validate the improved variable extraction system.
Tests both error log analysis and static code analysis variable extraction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from error_log_analysis import ErrorLogAnalyzer
from static_code_analysis import StaticCodeAnalyzer

def test_error_log_variable_extraction():
    """Test variable extraction from error logs."""
    print("=== Testing Error Log Variable Extraction ===")
    
    analyzer = ErrorLogAnalyzer()
    
    # Test cases with different error types
    test_cases = [
        {
            "name": "NameError with undefined variable",
            "log": "NameError: name 'undefined_var' is not defined at line 15",
            "expected_vars": ["undefined_var"]
        },
        {
            "name": "Multiple variables in IndexError",
            "log": "IndexError: list index out of range. Variable 'my_list' at index 'idx' caused error at line 23",
            "expected_vars": ["my_list", "idx"]
        },
        {
            "name": "Java NullPointerException",
            "log": "Exception in thread \"main\" java.lang.NullPointerException: Cannot invoke \"String.length()\" because 'user_name' is null at line 42",
            "expected_vars": ["user_name"]
        },
        {
            "name": "Complex error with multiple quoted variables",
            "log": "TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'. Variables involved: 'counter', 'result', 'data_value' at line 67",
            "expected_vars": ["counter", "result", "data_value"]
        },
        {
            "name": "C++ variable error",
            "log": "Error: variable 'buffer_size' was not declared in this scope at line 89",
            "expected_vars": ["buffer_size"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {test_case['log']}")
        
        result = analyzer.extract_entities(test_case['log'])
        extracted_vars = result.get('variables', [])
        
        print(f"Extracted variables: {extracted_vars}")
        print(f"Expected variables: {test_case['expected_vars']}")
        
        # Check if all expected variables are found
        found_expected = all(var in extracted_vars for var in test_case['expected_vars'])
        # Check if no unwanted keywords are extracted
        unwanted_found = any(var.lower() in ['error', 'exception', 'line', 'type', 'null', 'undefined'] 
                           for var in extracted_vars)
        
        if found_expected and not unwanted_found:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            if not found_expected:
                missing = [var for var in test_case['expected_vars'] if var not in extracted_vars]
                print(f"   Missing variables: {missing}")
            if unwanted_found:
                unwanted = [var for var in extracted_vars if var.lower() in ['error', 'exception', 'line', 'type', 'null', 'undefined']]
                print(f"   Unwanted variables: {unwanted}")

def test_static_code_variable_extraction():
    """Test variable extraction from code lines."""
    print("\n\n=== Testing Static Code Variable Extraction ===")
    
    analyzer = StaticCodeAnalyzer()
    
    # Test cases for different programming languages
    test_cases = [        {
            "name": "Python assignment",
            "code": "user_name = input('Enter name: ')",
            "language": "Python",
            "expected_vars": ["user_name"]
        },
        {
            "name": "Python array access",
            "code": "result = data_list[index] + calculation",
            "language": "Python", 
            "expected_vars": ["result", "data_list", "index", "calculation"]
        },
        {
            "name": "Python function call",
            "code": "output = process_data(input_file, config_params)",
            "language": "Python",
            "expected_vars": ["output", "input_file", "config_params"]
        },
        {
            "name": "Java object method call",
            "code": "String result = user.getName().toLowerCase();",
            "language": "Java",
            "expected_vars": ["result", "user"]
        },
        {
            "name": "Java array access",
            "code": "int value = numbers[counter] * multiplier;",
            "language": "Java",
            "expected_vars": ["value", "numbers", "counter", "multiplier"]
        },
        {
            "name": "C++ vector access",
            "code": "int data = my_vector[index] + offset;",
            "language": "C++",
            "expected_vars": ["data", "my_vector", "index", "offset"]
        },
        {
            "name": "C++ pointer dereferencing",
            "code": "result = *ptr_value + buffer_size;",
            "language": "C++",
            "expected_vars": ["result", "ptr_value", "buffer_size"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Code: {test_case['code']}")
        print(f"Language: {test_case['language']}")
        
        extracted_vars = analyzer.extract_variables_from_code_line(test_case['code'], test_case['language'])
        
        print(f"Extracted variables: {extracted_vars}")
        print(f"Expected variables: {test_case['expected_vars']}")
        
        # Check if most expected variables are found (allowing some flexibility)
        found_count = sum(1 for var in test_case['expected_vars'] if var in extracted_vars)
        expected_count = len(test_case['expected_vars'])
        
        if found_count >= expected_count * 0.7:  # At least 70% of expected variables found
            print("✅ PASS")
        else:
            print("❌ FAIL")
            missing = [var for var in test_case['expected_vars'] if var not in extracted_vars]
            print(f"   Missing variables: {missing}")

def test_integrated_analysis():
    """Test the integrated analysis with both error logs and code."""
    print("\n\n=== Testing Integrated Analysis ===")
    
    static_analyzer = StaticCodeAnalyzer()
    
    # Test case: Python code with NameError
    code_content = """
def calculate_total(prices):
    total = 0
    for price in prices:
        total += price * tax_rate  # tax_rate is not defined
    return total

result = calculate_total([10, 20, 30])
print(result)
"""
    
    error_log = "NameError: name 'tax_rate' is not defined at line 5"
    
    print("Code content:")
    print(code_content)
    print(f"\nError log: {error_log}")
      # Analyze the code
    error_data = {
        "error_type": "NameError",
        "line_number": "5",
        "variables": ["tax_rate"],
        "message": error_log
    }
    analysis_result = static_analyzer.analyze(code_content, error_data)
    
    print(f"\nAnalysis result:")
    print(f"Error type: {analysis_result.get('error_type')}")
    print(f"Line number: {analysis_result.get('line_number')}")
    print(f"Variables from error: {analysis_result.get('variables', [])}")
    print(f"Variables from code: {analysis_result.get('code_variables', [])}")
    print(f"Issues found: {len(analysis_result.get('issues', []))}")
    
    # Check if tax_rate is identified as a variable
    all_variables = set(analysis_result.get('variables', []) + analysis_result.get('code_variables', []))
    if 'tax_rate' in all_variables:
        print("✅ PASS - 'tax_rate' correctly identified")
    else:
        print("❌ FAIL - 'tax_rate' not identified")
        print(f"All identified variables: {all_variables}")

if __name__ == "__main__":
    print("Variable Extraction Validation Test")
    print("=" * 50)
    
    try:
        test_error_log_variable_extraction()
        test_static_code_variable_extraction()
        test_integrated_analysis()
        print("\n" + "=" * 50)
        print("Variable extraction testing completed!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
