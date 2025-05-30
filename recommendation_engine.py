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
            
            # Enhanced error-specific knowledge base for fallback explanations
            self.error_knowledge = {
                "IndexError": {
                    "causes": ["Accessing list/array with invalid index", "Loop counter exceeds bounds", "Empty list access"],
                    "solutions": ["Check bounds with len()", "Use try-except blocks", "Validate indices before access"],
                    "best_practices": ["Use enumerate() for safe iteration", "Consider using get() for dictionaries", "Implement bounds checking"]
                },
                "TypeError": {
                    "causes": ["Mixing incompatible data types", "Calling wrong method on object", "Incorrect function arguments"],
                    "solutions": ["Convert types explicitly", "Check object types", "Use type hints"],
                    "best_practices": ["Use isinstance() for type checking", "Implement type validation", "Use f-strings for formatting"]
                },
                "NullPointerException": {
                    "causes": ["Accessing null/uninitialized objects", "Method call on null reference", "Uninitialized variables"],
                    "solutions": ["Initialize objects properly", "Add null checks", "Use Optional patterns"],
                    "best_practices": ["Use null-safe operators", "Initialize at declaration", "Implement defensive programming"]
                },
                "NameError": {
                    "causes": ["Using undefined variables", "Typos in variable names", "Scope issues"],
                    "solutions": ["Define variables before use", "Check spelling", "Fix variable scope"],
                    "best_practices": ["Use proper variable naming", "Initialize variables", "Use IDE with linting"]
                },
                "AttributeError": {
                    "causes": ["Accessing non-existent attributes", "Wrong object type", "Undefined methods"],
                    "solutions": ["Check object has attribute", "Verify object type", "Use hasattr()"],
                    "best_practices": ["Use getattr() with defaults", "Implement proper error handling", "Use type checking"]
                }
            }
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

        # Enhanced prompt with better structure
        prompt = (
            f"Prompt: Fix a {language} error.\n"
            f"Error: {error_type} at line {line_number}\n"
            f"Message: {error_message}\n"
            f"Code line: {node_text}\n"
            f"Variables: {', '.join(variables) if variables else 'None'}\n"
            f"Intent: {', '.join(intent) if intent else 'None'}\n"
            f"Completion:"        )
        return prompt

    def parse_output(self, generated_text: str) -> Dict[str, Optional[str]]:
        result = {"corrected_code": None, "explanation": None, "best_practice": None}
        
        # Enhanced parsing with multiple patterns
        # Try different code block patterns
        code_patterns = [
            r"```[\w]*\n(.*?)```",  # Standard markdown code blocks
            r"`[\w]*\n(.*?)\n`",    # Single backtick blocks from dataset
            r"Corrected Code:\s*```[\w]*\n(.*?)```",  # Explicit corrected code blocks
            r"Corrected Code:\s*`[\w]*\n(.*?)\n`"     # Single backtick corrected code
        ]
        
        for pattern in code_patterns:
            code_match = re.search(pattern, generated_text, re.DOTALL)
            if code_match:
                result["corrected_code"] = code_match.group(1).strip()
                break
        
        # Enhanced explanation parsing
        explanation_patterns = [
            r"Explanation:\s*(.*?)(?:\nBest Practice:|$)",
            r"Explanation:\s*(.*?)(?:\n[A-Z]|$)",
            r"The [\w]+ (?:occurs|error)\s*(.*?)(?:\nBest Practice:|$)",
            r"(?:This|The)\s+(?:error|exception)\s+(.*?)(?:\nBest Practice:|$)"
        ]
        
        for pattern in explanation_patterns:
            explanation_match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                if len(explanation) > 10:  # Ensure meaningful explanation
                    result["explanation"] = explanation
                    break
        
        # Enhanced best practice parsing
        practice_patterns = [
            r"Best Practice:\s*(.*?)(?:\n[A-Z]|$)",
            r"Best Practice:\s*(.*)",
            r"(?:Recommendation|Tip|Advice):\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in practice_patterns:
            best_practice_match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
            if best_practice_match:
                practice = best_practice_match.group(1).strip()
                if len(practice) > 5:  # Ensure meaningful practice
                    result["best_practice"] = practice
                    break
        
        return result

    def generate_contextual_explanation(self, error_type: str, error_message: str, 
                                      code_line: str, language: str) -> Dict[str, str]:
        """Generate contextual explanations based on error details"""
        explanation = f"The {error_type} occurs in the code line '{code_line}'"
        best_practice = "Follow language-specific best practices"
        
        # Get base knowledge for error type
        error_info = self.error_knowledge.get(error_type, {})
        
        if error_type == "IndexError":
            if "range" in error_message.lower():
                explanation = f"The IndexError occurs because the index used in '{code_line}' exceeds the bounds of the list or array. This happens when trying to access an element at a position that doesn't exist."
                best_practice = "Always check array/list bounds using len() or use try-except blocks for safe access."
            elif "empty" in error_message.lower():
                explanation = f"The IndexError occurs because '{code_line}' is trying to access elements from an empty list or array."
                best_practice = "Check if the list is not empty before accessing elements, or initialize with default values."
                
        elif error_type == "TypeError":
            if "concatenate" in error_message or "+" in error_message:
                explanation = f"The TypeError occurs in '{code_line}' because you're trying to combine incompatible data types (like string + integer)."
                best_practice = "Use explicit type conversion (str(), int()) or f-strings for string formatting."
            elif "unsupported operand" in error_message:
                explanation = f"The TypeError happens because '{code_line}' uses an operation between incompatible types."
                best_practice = "Verify data types before operations and use appropriate type conversion methods."
                
        elif error_type == "NullPointerException":
            explanation = f"The NullPointerException occurs because '{code_line}' is trying to access methods or properties of a null/uninitialized object."
            best_practice = "Always initialize objects before use and add null checks before accessing object members."
            
        elif error_type == "NameError":
            if "not defined" in error_message:
                explanation = f"The NameError occurs because '{code_line}' references a variable that hasn't been defined or is out of scope."
                best_practice = "Ensure all variables are declared and initialized before use, and check for typos in variable names."
                
        elif error_type == "AttributeError":
            if "NoneType" in error_message:
                explanation = f"The AttributeError occurs because '{code_line}' is trying to access attributes of a None value."
                best_practice = "Check if objects are not None before accessing their attributes, or use getattr() with default values."
                
        return {"explanation": explanation, "best_practice": best_practice}

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
        error_message = error_data.get("message", "")

        if not language or not error_type or not node_text:
            return [{
                "summary": "Invalid analysis data",
                "fix": "No fix available",
                "explanation": "Missing error type, language, or code context from analysis.",
                "corrected_code": None,
                "best_practice": "Ensure analyzers produce valid outputs."
            }]

        prompt = self.construct_prompt(error_data, code_analysis)

        # Try to generate fix with model
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True, 
                padding=True,
                return_attention_mask=True
            ).to(self.device)
            
            # Improved generation parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=512,
                    min_length=len(inputs.input_ids[0]) + 50,
                    num_beams=3,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from generated text
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
                
            parsed_output = self.parse_output(generated_text)
            
        except Exception as e:
            print(f"Error generating fix with GPT-2: {str(e)}")
            parsed_output = {"corrected_code": None, "explanation": None, "best_practice": None}

        # Generate contextual fallback explanation if model output is insufficient
        if not parsed_output["explanation"] or len(parsed_output["explanation"]) < 20:
            contextual = self.generate_contextual_explanation(
                error_type, error_message, node_text, language
            )
            if not parsed_output["explanation"]:
                parsed_output["explanation"] = contextual["explanation"]
            if not parsed_output["best_practice"]:
                parsed_output["best_practice"] = contextual["best_practice"]

        # Enhanced corrected code generation if model fails
        if not parsed_output["corrected_code"]:
            parsed_output["corrected_code"] = self.generate_basic_fix(
                error_type, node_text, language, error_data
            )

        recommendation = {
            "summary": f"{error_type} detected at line {line_number}",
            "fix": "Apply the corrected code provided below",
            "explanation": parsed_output["explanation"] or f"The {error_type} needs to be resolved in the problematic code.",
            "corrected_code": parsed_output["corrected_code"],
            "best_practice": parsed_output["best_practice"] or "Follow language-specific best practices for error prevention."
        }

        return [recommendation]

    def generate_basic_fix(self, error_type: str, code_line: str, language: str, error_data: Dict) -> str:
        """Generate basic code fixes based on error patterns"""
        variables = error_data.get("variables", [])
        
        if error_type == "IndexError" and language == "Python":
            if variables:
                var = variables[0]
                if "range(" in code_line:
                    return f"for i in range(len({var})):\n    # your code here"
                else:
                    return f"if i < len({var}):\n    {code_line}"
            return f"# Add bounds checking before: {code_line}"
            
        elif error_type == "TypeError" and "+" in code_line and language == "Python":
            return code_line.replace("+", "+ str(") + ")"
            
        elif error_type == "NullPointerException" and language == "Java":
            if variables:
                var = variables[0]
                return f"if ({var} != null) {{\n    {code_line}\n}}"
            return f"// Add null check before: {code_line}"
            
        elif error_type == "NameError" and language == "Python":
            if variables:
                var = variables[0]
                return f"{var} = None  # Initialize variable\n{code_line}"
                
        return f"// TODO: Fix {error_type} in: {code_line}"

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