import pygments
from pygments.lexers import guess_lexer, get_all_lexers, get_lexer_by_name
from pygments.util import ClassNotFound
import re
from typing import Optional

class LanguageDetector:
    def __init__(self):
        self.language_patterns = {
            'Python': [
                r'\bdef\s+\w+\s*\(',  # Function definition
                r'\bprint\s*\(',       # Print statement
                r'\bimport\s+\w+',     # Import statement
                r'\.py\b'              # File extension
            ],
            'Java': [
                r'\bpublic\s+class\s+\w+',  # Class declaration
                r'\bSystem\.out\.println',  # Common Java output
                r'\bvoid\s+\w+\s*\(',      # Method declaration
                r'\.java\b'                 # File extension
            ],
            'C++': [
                r'#include\s*<\w+>',         # Include directive
                r'\bstd::\w+',               # Standard namespace usage
                r'\bint\s+main\s*\(',        # Main function
                r'\.cpp\b|\.cxx\b|\.h\b'    # File extensions
            ]
        }

    def detect_language(self, code_snippet: str) -> str:
        if not code_snippet or len(code_snippet.strip()) < 5:
            return "Unknown"
        #pygments lexical analysis
        try:
            lexer = guess_lexer(code_snippet)
            if lexer.name in self.language_patterns:
                return lexer.name
        except ClassNotFound:
            pass  # proceed to regex fallback

        # regex-based fallback
        snippet_lower = code_snippet.lower()
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, snippet_lower, re.IGNORECASE):
                    try:
                        lexer = get_lexer_by_name(lang)
                        return lexer.name
                    except ClassNotFound:
                        continue

        return "Unknown"

    def get_supported_languages(self) -> list:
        return [lexer[0] for lexer in get_all_lexers()]

# Example usage
if __name__ == "__main__":
    detector = LanguageDetector()

    # Test cases
    python_code = """
    def hello():
        print("Hello, World!")
    """
    
    java_code = """
    public class Main {
        public static void main(String[] args) {
            System.out.println("Hello, World!");
        }
    }
    """
    cpp_code = """
    #include <iostream>
    int main() {
        std::cout << "Hello, World!" << std::endl;
        return 0;
    }
    """
    ambiguous_code = "x = 10"
    empty_code = ""
    non_code = "This is not code."

    print(f"Python code detected as: {detector.detect_language(python_code)}")
    print(f"Java code detected as: {detector.detect_language(java_code)}")
    print(f"C++ code detected as: {detector.detect_language(cpp_code)}")
    print(f"Ambiguous code detected as: {detector.detect_language(ambiguous_code)}")
    print(f"Empty code detected as: {detector.detect_language(empty_code)}")
    print(f"Non-code detected as: {detector.detect_language(non_code)}")