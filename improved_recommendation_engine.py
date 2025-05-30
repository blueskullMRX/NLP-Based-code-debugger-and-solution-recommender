#!/usr/bin/env python3
"""
Improved Recommendation Engine with better fix templates and output quality
Addresses issues with repetitive/incomplete responses from the fine-tuned model
"""

from typing import Dict, Optional, List
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re
from static_code_analysis import StaticCodeAnalyzer
from error_log_analysis import ErrorLogAnalyzer

class ImprovedRecommendationEngine:
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
            
            # Template-based fixes for consistent, high-quality outputs
            self.fix_templates = {
                "IndexError": {
                    "python": {
                        "pattern": "for i in range(len(ITEMS)):\n    print(ITEMS[i])",
                        "explanation": "The IndexError occurs because the loop tries to access indices beyond the list length. Use range(len()) to ensure valid indices.",
                        "best_practice": "Always use range(len(list)) for index-based iteration, or use enumerate() for index-value pairs."
                    }
                },
                "TypeError": {
                    "python": {
                        "pattern": "result = str(VAR1) + str(VAR2)  # Convert to strings before concatenation",
                        "explanation": "The TypeError occurs when trying to perform operations on incompatible types. Use explicit type conversion.",
                        "best_practice": "Use f-strings for formatting: f'{VAR1}{VAR2}' or convert types explicitly with str(), int(), float()."
                    }
                },
                "NameError": {
                    "python": {
                        "pattern": "VARIABLE = None  # Initialize before use\nprint(VARIABLE)",
                        "explanation": "The NameError occurs because the variable is used before being defined. Variables must be initialized before use.",
                        "best_practice": "Always initialize variables before using them. Use meaningful default values or None for optional variables."
                    }
                },
                "NullPointerException": {
                    "java": {
                        "pattern": "if (OBJECT != null) {\n    OBJECT.method();\n} else {\n    System.out.println(\"Object is null\");\n}",
                        "explanation": "The NullPointerException occurs when trying to access methods of a null object. Add null checks before access.",
                        "best_practice": "Always check for null before accessing object members. Consider using Optional<T> for better null safety."
                    }
                },
                "AttributeError": {
                    "python": {
                        "pattern": "if OBJECT is not None:\n    OBJECT.attribute\nelse:\n    print(\"Object is None\")",
                        "explanation": "The AttributeError occurs when trying to access attributes of None. Check for None before accessing attributes.",
                        "best_practice": "Use None checks or getattr() with default values. Consider using hasattr() to check attribute existence."
                    }
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model or analyzers: {str(e)}")

    def apply_template_fix(self, error_type: str, language: str, variables: List[str], code_line: str) -> Dict[str, str]:
        """Apply template-based fix with proper variable substitution"""
        template_data = self.fix_templates.get(error_type, {}).get(language.lower(), {})
        
        if not template_data:
            return {
                "corrected_code": f"# TODO: Fix {error_type} in: {code_line}",
                "explanation": f"The {error_type} occurred and needs to be addressed.",
                "best_practice": f"Follow {language} best practices to prevent {error_type}."
            }
        
        # Get template components
        pattern = template_data["pattern"]
        explanation = template_data["explanation"]
        best_practice = template_data["best_practice"]
        
        # Apply variable substitutions
        corrected_code = pattern
        
        # Replace common placeholders with actual variables or defaults
        if variables:
            corrected_code = corrected_code.replace("VARIABLE", variables[0])
            corrected_code = corrected_code.replace("ITEMS", variables[0])
            corrected_code = corrected_code.replace("OBJECT", variables[0])
            corrected_code = corrected_code.replace("VAR1", variables[0])
            if len(variables) > 1:
                corrected_code = corrected_code.replace("VAR2", variables[1])
            else:
                corrected_code = corrected_code.replace("VAR2", "var2")
        else:
            # Use default variable names if none found
            corrected_code = corrected_code.replace("VARIABLE", "variable")
            corrected_code = corrected_code.replace("ITEMS", "items")
            corrected_code = corrected_code.replace("OBJECT", "obj")
            corrected_code = corrected_code.replace("VAR1", "x")
            corrected_code = corrected_code.replace("VAR2", "y")
        
        # Apply variable substitutions to explanations
        if variables:
            explanation = explanation.replace("VAR1", variables[0])
            explanation = explanation.replace("VAR2", variables[1] if len(variables) > 1 else "variable")
            best_practice = best_practice.replace("VAR1", variables[0])
            best_practice = best_practice.replace("VAR2", variables[1] if len(variables) > 1 else "variable")
        
        return {
            "corrected_code": corrected_code,
            "explanation": explanation,
            "best_practice": best_practice
        }

    def parse_model_output(self, generated_text: str) -> Dict[str, Optional[str]]:
        """Parse model output with multiple extraction strategies"""
        result = {"corrected_code": None, "explanation": None, "best_practice": None}
        
        # Clean the generated text
        cleaned_text = generated_text.strip()
        
        # Try to extract code blocks
        code_patterns = [
            r"```(?:\w+)?\s*\n(.*?)```",
            r"Corrected Code:\s*```(?:\w+)?\s*\n(.*?)```",
            r"```(.*?)```",
            r"Fixed Code:\s*(.*?)(?:\n\n|\n[A-Z]|$)"
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            if match and len(match.group(1).strip()) > 5:
                code = match.group(1).strip()
                # Filter out obviously bad outputs
                if not any(bad in code.lower() for bad in ["error", "undefined", "null reference"]):
                    result["corrected_code"] = code
                    break
        
        # Extract explanations
        explanation_patterns = [
            r"Explanation:\s*(.*?)(?:\n(?:Best|Recommendation)|$)",
            r"The\s+\w+Error\s+(.*?)(?:\n(?:Best|Recommendation)|$)",
            r"This\s+error\s+(.*?)(?:\n(?:Best|Recommendation)|$)"
        ]
        
        for pattern in explanation_patterns:
            match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            if match:
                explanation = match.group(1).strip()
                # Filter meaningful explanations
                if len(explanation) > 20 and any(word in explanation.lower() for word in ["occurs", "happens", "because", "when"]):
                    result["explanation"] = explanation
                    break
        
        # Extract best practices
        practice_patterns = [
            r"Best Practice:\s*(.*?)(?:\n|$)",
            r"Recommendation:\s*(.*?)(?:\n|$)",
            r"To avoid:\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in practice_patterns:
            match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            if match:
                practice = match.group(1).strip()
                if len(practice) > 15:
                    result["best_practice"] = practice
                    break
        
        return result

    def try_model_generation(self, error_data: Dict, code_analysis: Dict) -> Dict[str, Optional[str]]:
        """Try to generate fix using the model with optimized parameters"""
        try:
            language = code_analysis.get("language", "Unknown")
            error_type = error_data.get("error_type", "Unknown")
            error_message = error_data.get("message", "")
            code_line = code_analysis.get("error_location", {}).get("node_text", "")
            
            # Simplified, focused prompt
            prompt = f"Fix {language} {error_type}:\nCode: {code_line}\nError: {error_message}\n\nCorrected Code:\n"
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=200, 
                truncation=True, 
                padding=True
            ).to(self.device)
            
            # Optimized generation parameters for better quality
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=300,
                    num_beams=4,
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from generated text
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
                
            return self.parse_model_output(generated_text)
            
        except Exception as e:
            print(f"Model generation failed: {str(e)}")
            return {"corrected_code": None, "explanation": None, "best_practice": None}

    def generate_recommendations(self, code: str, error_log: str) -> List[Dict[str, str]]:
        """Generate improved recommendations with template fallbacks"""
        try:
            error_data = self.error_analyzer.analyze(error_log)
            code_analysis = self.code_analyzer.analyze(code, error_data)
        except Exception as e:
            return [{
                "summary": "Analysis failed",
                "fix": "Unable to analyze the provided code and error log",
                "explanation": f"Analysis error: {str(e)}",
                "corrected_code": "# Unable to generate corrected code",
                "best_practice": "Ensure the code and error log are valid."
            }]

        language = code_analysis.get("language", "Unknown")
        error_type = error_data.get("error_type", "Unknown")
        line_number = error_data.get("line_number", "Unknown")
        code_line = code_analysis.get("error_location", {}).get("node_text", "")
        variables = error_data.get("variables", [])

        if not all([language, error_type, code_line]):
            return [{
                "summary": "Incomplete analysis",
                "fix": "Missing critical information",
                "explanation": "Unable to identify error type, language, or code line",
                "corrected_code": "# Analysis incomplete",
                "best_practice": "Provide clear error logs with line numbers."
            }]

        # Get template-based fix (reliable fallback)
        template_fix = self.apply_template_fix(error_type, language, variables, code_line)
        
        # Try model generation for potential enhancement
        model_output = self.try_model_generation(error_data, code_analysis)
        
        # Use model output if it's substantially better than template
        corrected_code = template_fix["corrected_code"]
        explanation = template_fix["explanation"]
        best_practice = template_fix["best_practice"]
        
        # Enhance with model output if available and good quality
        if model_output.get("corrected_code") and len(model_output["corrected_code"]) > 20:
            # Only use model output if it contains actual code
            model_code = model_output["corrected_code"]
            if any(keyword in model_code for keyword in ["if", "for", "def", "print", "return", "="]):
                corrected_code = model_code
        
        if model_output.get("explanation") and len(model_output["explanation"]) > 50:
            explanation = model_output["explanation"]
        
        if model_output.get("best_practice") and len(model_output["best_practice"]) > 25:
            best_practice = model_output["best_practice"]

        return [{
            "summary": f"{error_type} detected at line {line_number}",
            "fix": "Apply the corrected code and follow best practices",
            "explanation": explanation,
            "corrected_code": corrected_code,
            "best_practice": best_practice
        }]


# Test the improved engine
if __name__ == "__main__":
    print("=== TESTING IMPROVED RECOMMENDATION ENGINE ===")
    engine = ImprovedRecommendationEngine()

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
        String obj = null;
        obj.length();  // Error at line 4
    }
}
"""
    error_log4 = "NullPointerException: Cannot invoke String.length() on null reference at line 4"
    recommendations4 = engine.generate_recommendations(java_code, error_log4)
    for rec in recommendations4:
        print(f"Summary: {rec['summary']}")
        print(f"Explanation: {rec['explanation']}")
        print(f"Corrected Code:\n{rec['corrected_code']}")
        print(f"Best Practice: {rec['best_practice']}")
