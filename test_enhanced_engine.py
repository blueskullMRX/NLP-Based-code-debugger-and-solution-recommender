#!/usr/bin/env python3
"""
Test script for the enhanced recommendation engine
"""

from recommendation_engine import RecommendationEngine

def test_enhanced_engine():
    print("=== TESTING ENHANCED RECOMMENDATION ENGINE ===")
    engine = RecommendationEngine()

    # Test Case 1: Python IndexError
    print("\n--- Test Case 1: Python IndexError ---")
    python_code = """
def process_list(items):
    for i in range(10):
        print(items[i])  # Error at line 3
"""
    error_log = "IndexError: list index out of range at line 3"
    recommendations = engine.generate_recommendations(python_code, error_log)
    for rec in recommendations:
        print(f"Summary: {rec['summary']}")
        print(f"Explanation: {rec['explanation']}")
        print(f"Corrected Code:\n{rec['corrected_code']}")
        print(f"Best Practice: {rec['best_practice']}")

    # Test Case 2: Python TypeError
    print("\n--- Test Case 2: Python TypeError ---")
    python_code2 = """
def add_values(x, y):
    result = x + y  # Error: x=5, y='10'
    return result
"""
    error_log2 = "TypeError: unsupported operand type(s) for +: 'int' and 'str' at line 2"
    recommendations2 = engine.generate_recommendations(python_code2, error_log2)
    for rec in recommendations2:
        print(f"Summary: {rec['summary']}")
        print(f"Explanation: {rec['explanation']}")
        print(f"Corrected Code:\n{rec['corrected_code']}")
        print(f"Best Practice: {rec['best_practice']}")

    # Test Case 3: Python NameError
    print("\n--- Test Case 3: Python NameError ---")
    python_code3 = """
def print_value():
    print(undefined_var)  # Error
print_value()
"""
    error_log3 = "NameError: name 'undefined_var' is not defined at line 2"
    recommendations3 = engine.generate_recommendations(python_code3, error_log3)
    for rec in recommendations3:
        print(f"Summary: {rec['summary']}")
        print(f"Explanation: {rec['explanation']}")
        print(f"Corrected Code:\n{rec['corrected_code']}")
        print(f"Best Practice: {rec['best_practice']}")

    # Test Case 4: Java NullPointerException
    print("\n--- Test Case 4: Java NullPointerException ---")
    java_code = """
public class Main {
    public static void main(String[] args) {
        String s = null;
        System.out.println(s.length()); // Error at line 4
    }
}
"""
    java_error_log = "NullPointerException: Cannot invoke method on null object at line 4"
    recommendations4 = engine.generate_recommendations(java_code, java_error_log)
    for rec in recommendations4:
        print(f"Summary: {rec['summary']}")
        print(f"Explanation: {rec['explanation']}")
        print(f"Corrected Code:\n{rec['corrected_code']}")
        print(f"Best Practice: {rec['best_practice']}")

if __name__ == "__main__":
    test_enhanced_engine()
