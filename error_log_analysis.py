from transformers import (
    pipeline,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    )
from language_detection import LanguageDetector
from typing import Dict, Optional
import spacy, re, torch

"""
What it does: Reads the error message to understand what went wrong (e.g., identifies “IndexError” and the line number where it happened).
Purpose: Extracts key details like error type, line number, and variables involved to pinpoint the problem.
Output: A dictionary with error details (e.g., { "error_type": "IndexError", "line_number": "3", "variables": ["items"] }).
"""

class ErrorLogAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

        #CodeBERT fine-tuned
        model_name = "microsoft/codebert-base"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.classifier = pipeline(
                task="text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                framework="pt"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CodeBERT pipeline: {str(e)}")

        #error type mapping
        self.error_types = {
            "SyntaxError": "syntax",
            "IndexError": "runtime",
            "NameError": "runtime",
            "TypeError": "runtime",
            "NullPointerException": "runtime",
            "SemanticError": "semantic",
            "LogicError": "logical"
        }

    def extract_entities(self, log_text: str) -> Dict[str, Optional[str]]:

        if not log_text or not isinstance(log_text, str):
            return {"error_type": None, "line_number": None, "variables": [], "message": None}

        doc = self.nlp(log_text)
        entities = {"variables": []}
        
        error_type_match = re.search(r"(?:[\w\.]+Exception|[\w]+Error)", log_text, re.IGNORECASE)
        line_number_match = re.search(r"at line (\d+)|line:?\s*(\d+)", log_text, re.IGNORECASE)

        entities["error_type"] = error_type_match.group(0) if error_type_match else None
        entities["line_number"] = line_number_match.group(1) or line_number_match.group(2) if line_number_match else None
        entities["message"] = log_text.strip()

        #extract variable names
        for token in doc:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                if token.text.isidentifier():
                    entities["variables"].append(token.text)

        return entities

    def classify_error(self, log_text: str) -> str:

        if not log_text or not isinstance(log_text, str):
            return "unknown"

        # CodeBERT for classification
        # In practice, fine-tune CodeBERT on a labeled dataset of error logs
        result = self.classifier(log_text, truncation=True, max_length=512)
        
        # Mocked mapping (replace with actual fine-tuned model output)
        entities = self.extract_entities(log_text)
        error_type = entities.get("error_type", "Unknown")
        return self.error_types.get(error_type, "unknown")

    def analyze(self, log_text: str) -> Dict[str, Optional[str]]:
        entities = self.extract_entities(log_text)
        classification = self.classify_error(log_text)
        
        return {
            "error_type": entities["error_type"],
            "line_number": entities["line_number"],
            "variables": entities["variables"],
            "message": entities["message"],
            "classification": classification
        }

# Example usage
if __name__ == "__main__":
    analyzer = ErrorLogAnalyzer()

    # Test cases
    python_error = "IndexError: list index out of range at line 24"
    
    java_error = "NullPointerException: Cannot invoke method on null object at line 42"
    
    ambiguous_error = "Error: something went wrong"

    python_syntax_error = (
        "SyntaxError: invalid syntax\n"
        "  File \"main.py\", line 10\n"
        "    print('Hello World'\n"
        "                        ^\n"
        "SyntaxError: unexpected EOF while parsing"
    )

    java_semantic_error = (
        "SemanticError: incompatible types: int cannot be converted to String\n"
        "    at Calculator.addNumbers(Calculator.java:15)\n"
        "    at Calculator.main(Calculator.java:5)"
    )

    python_type_error = (
        "TypeError: unsupported operand type(s) for +: 'int' and 'str'\n"
        "  File \"utils.py\", line 33, in add_values\n"
        "    result = a + b"
    )

    java_logic_error = (
        "LogicError: Division by zero not handled\n"
        "    at Calculator.divideNumbers(Calculator.java:22)\n"
        "    at Calculator.main(Calculator.java:8)"
    )

    multi_line_python_error = (
        "Traceback (most recent call last):\n"
        "  File \"script.py\", line 18, in <module>\n"
        "    main()\n"
        "  File \"script.py\", line 12, in main\n"
        "    process_data(data)\n"
        "  File \"script.py\", line 7, in process_data\n"
        "    print(data[10])\n"
        "IndexError: list index out of range"
    )

    java_stack_trace = (
        "Exception in thread \"main\" java.lang.ArrayIndexOutOfBoundsException: 5\n"
        "    at Example.main(Example.java:13)\n"
        "    at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)\n"
        "    at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)\n"
        "    at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)\n"
        "    at java.lang.reflect.Method.invoke(Method.java:498)"
    )
    empty_error = ""

    print("Python Error Analysis:")
    print(analyzer.analyze(python_error))

    print("\nJava Error Analysis:")
    print(analyzer.analyze(java_error))

    print("\nAmbiguous Error Analysis:")
    print(analyzer.analyze(ambiguous_error))

    print("\nPython syntax error : ")
    print(analyzer.analyze(python_syntax_error))
    
    print(f"\nPython type error : ")
    print(analyzer.analyze(python_type_error))
    
    print(f"\nJava logic Error :")
    print(f"{analyzer.analyze(java_logic_error)}")

    print("\nEmpty Error Analysis:")
    print(analyzer.analyze(empty_error))