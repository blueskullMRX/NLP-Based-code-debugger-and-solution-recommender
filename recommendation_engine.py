# .\recommendation_engine.py
"""
What it does: Suggests how to fix the error, provides a corrected code snippet, explains why the error happened, and gives tips to avoid it in the future.
Purpose: Combines error details (Stage 3) and code context (Stage 4) to give the user a complete solution.
Output: A list of recommendations (e.g., { "summary": "IndexError at line 3", "fix": "Check list bounds", "corrected_code": "for i in range(len(items)):\n print(items[i])", "explanation": "Index too high", "best_practice": "Use try-except" }).
"""

from typing import Dict, Optional, List
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re
from static_code_analysis import StaticCodeAnalyzer
from error_log_analysis import ErrorLogAnalyzer
from language_detection import LanguageDetector

class RecommendationEngine:
    def __init__(self, model_path: str = "./gpt2_finetuned"):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.error_analyzer = ErrorLogAnalyzer()
            self.code_analyzer = StaticCodeAnalyzer()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPT-2 model or analyzers: {str(e)}")

    def construct_prompt(self, error_data: Dict[str, Optional[str]], code_analysis: Dict[str, Optional[any]]) -> str:
        language = code_analysis.get("language", "Unknown")
        error_type = error_data.get("error_type", "Unknown")
        error_message = error_data.get("message", "No error message")
        line_number = error_data.get("line_number", "Unknown")
        node_text = code_analysis.get("error_location", {}).get("node_text", "Unknown")
        variables = error_data.get("variables", [])
        intent = code_analysis.get("intent", [])

        prompt = (
            f"Fix a {language} error.\n"
            f"Error: {error_type} at line {line_number}\n"
            f"Message: {error_message}\n"
            f"Code line: {node_text}\n"
            f"Variables: {', '.join(variables) if variables else 'None'}\n"
            f"Intent: {', '.join(intent) if intent else 'None'}"
        )
        return prompt

    def parse_output(self, generated_text: str) -> Dict[str, Optional[str]]:
        result = {"corrected_code": None, "explanation": None, "best_practice": None}
        
        code_match = re.search(r"```[a-zA-Z+\s]*\n(.*?)```", generated_text, re.DOTALL)
        if code_match:
            result["corrected_code"] = code_match.group(1).strip()
        
        explanation_match = re.search(r"Explanation:\s*(.*?)(?:\nBest Practice:|$)", generated_text, re.DOTALL)
        if explanation_match:
            result["explanation"] = explanation_match.group(1).strip()
        
        best_practice_match = re.search(r"Best Practice:\s*(.*)", generated_text, re.DOTALL)
        if best_practice_match:
            result["best_practice"] = best_practice_match.group(1).strip()
        
        return result

    def generate_recommendations(self, code: str, error_log: str) -> List[Dict[str, str]]:
        try:
            error_data = self.error_analyzer.analyze(error_log)
            code_analysis = self.code_analyzer.analyze(code, error_data)
        except Exception as e:
            return [{
                "summary": "Analysis failed",
                "fix": "No fix available",
                "explanation": f"Failed to analyze code or error log: {str(e)}",
                "corrected_code": None,
                "best_practice": "Ensure valid code and error log inputs."
            }]

        language = code_analysis.get("language")
        error_type = error_data.get("error_type")
        line_number = error_data.get("line_number")
        node_text = code_analysis.get("error_location", {}).get("node_text")

        if not language or not error_type or not node_text: #validate
            return [{
                "summary": "Invalid analysis data",
                "fix": "No fix available",
                "explanation": "Missing error type, language, or code context from analysis.",
                "corrected_code": None,
                "best_practice": "Ensure analyzers produce valid outputs."
            }]

        prompt = self.construct_prompt(error_data, code_analysis)

        #fix
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True, padding=True).to(self.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.pad_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed_output = self.parse_output(generated_text)
        except Exception as e:
            print(f"Error generating fix with GPT-2: {str(e)}")
            return [{
                "summary": f"{error_type} detected at line {line_number}",
                "fix": "No fix available",
                "explanation": "Failed to generate fix due to model error.",
                "corrected_code": None,
                "best_practice": "Manually review the code."
            }]

        recommendation = {
            "summary": f"{error_type} detected at line {line_number}",
            "fix": "Apply the corrected code provided.",
            "explanation": parsed_output["explanation"] or "Generated fix based on error context.",
            "corrected_code": parsed_output["corrected_code"],
            "best_practice": parsed_output["best_practice"] or "Follow language-specific best practices."
        }

        return [recommendation]

# Testing with diverse code snippets and error logs
if __name__ == "__main__":
    engine = RecommendationEngine()

    # Test Case 1: Python TypeError
    print("\nTest Case 1: Python TypeError")
    python_type_code = """
def add_values(x, y):
    result = x + y  # Error: x=5, y='10'
    return result
"""
    python_type_error_log = "TypeError: unsupported operand type(s) for +: 'int' and 'str' at line 2"
    print("Input Code:")
    print(python_type_code.strip())
    print("Error Log:", python_type_error_log)
    print("Recommendation:")
    print(engine.generate_recommendations(python_type_code, python_type_error_log))

    # Test Case 2: Java ArrayIndexOutOfBoundsException
    print("\nTest Case 2: Java ArrayIndexOutOfBoundsException")
    java_array_code = """
public class Main {
    public static void main(String[] args) {
        int[] array = {1, 2, 3};
        int value = array[5];  // Error
    }
}
"""
    java_array_error_log = "ArrayIndexOutOfBoundsException: Index 5 out of bounds for length 3 at line 4"
    print("Input Code:")
    print(java_array_code.strip())
    print("Error Log:", java_array_error_log)
    print("Recommendation:")
    print(engine.generate_recommendations(java_array_code, java_array_error_log))

    # Test Case 3: C++ Segmentation Fault
    print("\nTest Case 3: C++ Segmentation Fault")
    cpp_seg_code = """
#include <iostream>
int main() {
    int* ptr = nullptr;
    *ptr = 42;  // Error
    return 0;
}
"""
    cpp_seg_error_log = "Segmentation fault: invalid memory access at line 4"
    print("Input Code:")
    print(cpp_seg_code.strip())
    print("Error Log:", cpp_seg_error_log)
    print("Recommendation:")
    print(engine.generate_recommendations(cpp_seg_code, cpp_seg_error_log))

    # Test Case 4: Python NameError
    print("\nTest Case 4: Python NameError")
    python_name_code = """
def print_value():
    print(undefined_var)  # Error
print_value()
"""
    python_name_error_log = "NameError: name 'undefined_var' is not defined at line 2"
    print("Input Code:")
    print(python_name_code.strip())
    print("Error Log:", python_name_error_log)
    print("Recommendation:")
    print(engine.generate_recommendations(python_name_code, python_name_error_log))

    # Test Case 5: Java NullPointerException
    print("\nTest Case 5: Java NullPointerException")
    java_null_code = """
import java.util.List;
import java.util.ArrayList;
public class Main {
    public static void main(String[] args) {
        List<Integer> list = null;
        list.add(10);  // Error
    }
}
"""
    java_null_error_log = "NullPointerException: Cannot invoke 'List.add()' on null object at line 6"
    print("Input Code:")
    print(java_null_code.strip())
    print("Error Log:", java_null_error_log)
    print("Recommendation:")
    print(engine.generate_recommendations(java_null_code, java_null_error_log))