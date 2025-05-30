#!/usr/bin/env python3
"""
Comprehensive test comparing different recommendation approaches
"""

from enhanced_recommendation_engine import EnhancedRecommendationEngine

def test_comprehensive_cases():
    print("=== COMPREHENSIVE RECOMMENDATION ENGINE TESTING ===\n")
    engine = EnhancedRecommendationEngine()
    
    test_cases = [
        {
            "name": "Python IndexError - List Access",
            "code": """
def process_data(items):
    for i in range(10):
        print(items[i])  # Error here
""",
            "error": "IndexError: list index out of range at line 3"
        },
        {
            "name": "Python TypeError - String Concatenation",
            "code": """
def create_message(name, age):
    message = "Hello " + name + ", you are " + age + " years old"
    return message
""",
            "error": "TypeError: can only concatenate str (not \"int\") to str at line 2"
        },
        {
            "name": "Python NameError - Undefined Variable",
            "code": """
def calculate_total():
    total = price * quantity  # price not defined
    return total
""",
            "error": "NameError: name 'price' is not defined at line 2"
        },
        {
            "name": "Java NullPointerException",
            "code": """
public class Example {
    public static void main(String[] args) {
        String text = null;
        int length = text.length();  // Error here
    }
}
""",
            "error": "NullPointerException: Cannot invoke \"String.length()\" because \"text\" is null at line 4"
        },
        {
            "name": "Python AttributeError - None Object",
            "code": """
def process_data(data_list):
    result = None
    result.append("new_item")  # Error here
    return result
""",
            "error": "AttributeError: 'NoneType' object has no attribute 'append' at line 3"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"=== TEST CASE {i}: {test_case['name']} ===")
        print(f"Code:\n{test_case['code'].strip()}")
        print(f"\nError: {test_case['error']}")
        
        recommendations = engine.generate_recommendations(test_case['code'], test_case['error'])
        
        for rec in recommendations:
            print(f"\n📋 Summary: {rec['summary']}")
            print(f"\n💡 Explanation: {rec['explanation']}")
            print(f"\n🔧 Corrected Code:\n{rec['corrected_code']}")
            print(f"\n✅ Best Practice: {rec['best_practice']}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    test_comprehensive_cases()
