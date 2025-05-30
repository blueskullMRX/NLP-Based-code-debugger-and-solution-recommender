#!/usr/bin/env python3
"""
Enhanced Recommendation Engine with improved code generation and contextual explanations
"""

from typing import Dict, Optional, List
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re
from static_code_analysis import StaticCodeAnalyzer
from error_log_analysis import ErrorLogAnalyzer
from language_detection import LanguageDetector

class EnhancedRecommendationEngine:
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
            
            # Comprehensive error pattern database
            self.error_patterns = {
                "IndexError": {
                    "python": {
                        "causes": ["Index exceeds list/array bounds", "Accessing empty list", "Off-by-one errors"],
                        "fixes": [
                            "Use range(len(list_name)) for iteration",
                            "Add bounds checking with if i < len(list_name)",
                            "Use try-except IndexError blocks"
                        ],
                        "examples": {
                            "range_fix": "for i in range(len({var})):\n    print({var}[i])",
                            "bounds_check": "if i < len({var}):\n    value = {var}[i]",
                            "try_except": "try:\n    value = {var}[i]\nexcept IndexError:\n    value = None"
                        }
                    }
                },
                "TypeError": {
                    "python": {
                        "causes": ["Incompatible type operations", "String + integer concatenation", "Wrong function arguments"],
                        "fixes": [
                            "Use explicit type conversion",
                            "Use f-strings for formatting",
                            "Check types with isinstance()"
                        ],
                        "examples": {
                            "string_concat": "result = f'Age: {age}'",
                            "type_conversion": "result = str(x) + str(y)",
                            "type_check": "if isinstance(x, int) and isinstance(y, int):\n    result = x + y"
                        }
                    }
                },
                "NameError": {
                    "python": {
                        "causes": ["Using undefined variables", "Typos in variable names", "Variable scope issues"],
                        "fixes": [
                            "Define variables before use",
                            "Check variable spelling",
                            "Fix variable scope"
                        ],
                        "examples": {
                            "define_var": "{var} = None  # Initialize variable\nprint({var})",
                            "global_var": "global {var}\n{var} = 'some_value'",
                            "default_value": "{var} = {var} if '{var}' in locals() else 'default'"
                        }
                    }
                },
                "NullPointerException": {
                    "java": {
                        "causes": ["Null object access", "Uninitialized objects", "Method calls on null"],
                        "fixes": [
                            "Add null checks",
                            "Initialize objects properly",
                            "Use Optional pattern"
                        ],
                        "examples": {
                            "null_check": "if ({var} != null) {{\n    {original_code}\n}}",
                            "initialization": "{var} = new {type}();\n{original_code}",
                            "optional": "Optional.ofNullable({var}).ifPresent(obj -> {{\n    // use obj\n}});"
                        }
                    }
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model or analyzers: {str(e)}")

    def analyze_error_context(self, error_data: Dict, code_analysis: Dict) -> Dict:
        """Analyze error context to determine the best fix approach"""
        error_type = error_data.get("error_type", "")
        language = code_analysis.get("language", "").lower()
        code_line = code_analysis.get("error_location", {}).get("node_text", "")
        variables = error_data.get("variables", [])
        
        context = {
            "error_type": error_type,
            "language": language,
            "code_line": code_line,
            "variables": variables,
            "fix_strategy": "generic"
        }
        
        # Determine specific fix strategy based on error patterns
        if error_type == "IndexError" and language == "python":
            if "range(" in code_line:
                context["fix_strategy"] = "range_fix"
            elif any(var in code_line for var in variables):
                context["fix_strategy"] = "bounds_check"
            else:
                context["fix_strategy"] = "try_except"
                
        elif error_type == "TypeError" and language == "python":
            if "+" in code_line and any(keyword in code_line for keyword in ["str", "int", "'"]):
                context["fix_strategy"] = "string_concat"
            else:
                context["fix_strategy"] = "type_conversion"
                
        elif error_type == "NameError" and language == "python":
            context["fix_strategy"] = "define_var"
            
        elif error_type == "NullPointerException" and language == "java":
            if any(var in code_line for var in variables):
                context["fix_strategy"] = "null_check"
            else:
                context["fix_strategy"] = "initialization"
        
        return context

    def generate_corrected_code(self, context: Dict) -> str:
        """Generate corrected code based on error context"""
        error_type = context["error_type"]
        language = context["language"]
        fix_strategy = context["fix_strategy"]
        code_line = context["code_line"]
        variables = context["variables"]
        
        if error_type in self.error_patterns and language in self.error_patterns[error_type]:
            pattern_data = self.error_patterns[error_type][language]
            examples = pattern_data.get("examples", {})
            
            if fix_strategy in examples:
                template = examples[fix_strategy]
                
                # Replace placeholders in template
                if variables:
                    var = variables[0]
                    corrected = template.format(var=var, original_code=code_line)
                else:
                    corrected = template.format(var="variable", original_code=code_line)
                    
                return corrected
        
        # Fallback to generic fixes
        return self.generate_generic_fix(error_type, code_line, language, variables)

    def generate_generic_fix(self, error_type: str, code_line: str, language: str, variables: List) -> str:
        """Generate generic fixes for common errors"""
        if error_type == "IndexError" and language == "python":
            if variables:
                return f"# Check bounds before accessing {variables[0]}\nif i < len({variables[0]}):\n    {code_line}"
            return f"# Add bounds checking\n{code_line}"
            
        elif error_type == "TypeError" and language == "python":
            return f"# Convert types explicitly\n{code_line.replace('+', '+ str(') + ')'}"
            
        elif error_type == "NameError" and language == "python":
            if variables:
                return f"{variables[0]} = None  # Initialize variable\n{code_line}"
            return f"# Define variable before use\n{code_line}"
            
        elif error_type == "NullPointerException" and language == "java":
            if variables:
                return f"if ({variables[0]} != null) {{\n    {code_line}\n}}"
            return f"// Add null check\n{code_line}"
            
        return f"// TODO: Fix {error_type}\n{code_line}"

    def generate_contextual_explanation(self, context: Dict, error_message: str) -> str:
        """Generate detailed, contextual explanations"""
        error_type = context["error_type"]
        language = context["language"]
        code_line = context["code_line"]
        variables = context["variables"]
        
        base_explanations = {
            "IndexError": f"The IndexError occurs because the code '{code_line}' is trying to access an array or list element at an index that doesn't exist.",
            "TypeError": f"The TypeError happens in '{code_line}' because incompatible data types are being used together.",
            "NameError": f"The NameError occurs because '{code_line}' references a variable that hasn't been defined.",
            "NullPointerException": f"The NullPointerException happens because '{code_line}' is trying to use a null or uninitialized object.",
            "AttributeError": f"The AttributeError occurs because '{code_line}' is trying to access an attribute that doesn't exist."
        }
        
        explanation = base_explanations.get(error_type, f"The {error_type} occurs in the code line '{code_line}'.")
        
        # Add specific context based on error message
        if "range" in error_message.lower() and error_type == "IndexError":
            explanation += " This typically happens when a loop tries to access more elements than exist in the list."
        elif "concatenate" in error_message.lower() and error_type == "TypeError":
            explanation += " This occurs when trying to combine a string with a number using the + operator."
        elif "not defined" in error_message.lower() and error_type == "NameError":
            explanation += " This happens when using a variable before declaring it or due to typos in variable names."
        elif error_type == "NullPointerException":
            explanation += " This is a common issue when objects are not properly initialized before use."
            
        return explanation

    def generate_best_practice(self, context: Dict) -> str:
        """Generate best practices based on error context"""
        error_type = context["error_type"]
        language = context["language"]
        
        practices = {
            "IndexError": {
                "python": "Always validate array bounds using len() or use enumerate() for safe iteration. Consider using try-except blocks for robust error handling."
            },
            "TypeError": {
                "python": "Use explicit type conversion (str(), int()) or f-strings for string formatting. Implement type hints and validation."
            },
            "NameError": {
                "python": "Initialize all variables before use and use proper naming conventions. Consider using IDE with linting support."
            },
            "NullPointerException": {
                "java": "Always initialize objects at declaration and use null checks or Optional patterns for safe coding."
            },
            "AttributeError": {
                "python": "Use hasattr() to check for attributes or getattr() with default values. Implement proper error handling."
            }
        }
        
        if error_type in practices and language in practices[error_type]:
            return practices[error_type][language]
        
        return f"Follow {language} best practices for {error_type} prevention and use defensive programming techniques."

    def generate_recommendations(self, code: str, error_log: str) -> List[Dict[str, str]]:
        """Generate comprehensive recommendations with improved explanations"""
        try:
            error_data = self.error_analyzer.analyze(error_log)
            code_analysis = self.code_analyzer.analyze(code, error_data)
        except Exception as e:
            return [{
                "summary": "Analysis failed",
                "fix": "Unable to analyze the provided code and error",
                "explanation": f"Analysis error: {str(e)}",
                "corrected_code": "# Unable to generate corrected code",
                "best_practice": "Ensure valid code and error log inputs."
            }]

        # Validate analysis results
        language = code_analysis.get("language")
        error_type = error_data.get("error_type")
        line_number = error_data.get("line_number")
        error_message = error_data.get("message", "")

        if not all([language, error_type, line_number]):
            return [{
                "summary": "Incomplete analysis",
                "fix": "Missing critical error information",
                "explanation": "Unable to extract sufficient information from the error log or code.",
                "corrected_code": "# Analysis incomplete",
                "best_practice": "Provide complete error logs and valid code snippets."
            }]

        # Analyze error context
        context = self.analyze_error_context(error_data, code_analysis)
        
        # Generate corrected code
        corrected_code = self.generate_corrected_code(context)
        
        # Generate contextual explanation
        explanation = self.generate_contextual_explanation(context, error_message)
        
        # Generate best practice advice
        best_practice = self.generate_best_practice(context)

        return [{
            "summary": f"{error_type} detected at line {line_number} in {language} code",
            "fix": "Apply the corrected code and follow the best practices below",
            "explanation": explanation,
            "corrected_code": corrected_code,
            "best_practice": best_practice
        }]

# Test the enhanced engine
if __name__ == "__main__":
    engine = EnhancedRecommendationEngine()
    
    # Test case: Python IndexError
    test_code = """
def process_list(items):
    for i in range(10):
        print(items[i])  # Error at line 3
"""
    test_error = "IndexError: list index out of range at line 3"
    
    print("=== ENHANCED RECOMMENDATION ENGINE TEST ===")
    recommendations = engine.generate_recommendations(test_code, test_error)
    
    for rec in recommendations:
        print(f"\nSummary: {rec['summary']}")
        print(f"\nExplanation: {rec['explanation']}")
        print(f"\nCorrected Code:\n{rec['corrected_code']}")
        print(f"\nBest Practice: {rec['best_practice']}")
