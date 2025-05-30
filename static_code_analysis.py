#.\static_code_analysis.py
import ast, re, tree_sitter
from tree_sitter import Language, Parser
from typing import Dict, Optional, List
from language_detection import LanguageDetector

"""
Implements Stage 4 of the project, analyzing code snippets to identify error locations and infer programmer intent, as specified in the README. It maps error details (e.g., line number from Stage 3) to specific code blocks and detects structures like loops or functions.
"""

class StaticCodeAnalyzer:
    """
    Analyzes code snippets to identify error locations and infer programmer intent.
    Uses ast for Python and regex for Java/C++.
    """
    def __init__(self):
        # Initialize language detector from Stage 2
        self.language_detector = LanguageDetector()
        
        # Define regex patterns for Java and C++ intent and error location
        self.intent_patterns = {
            'Java': {
                'function': r'\b(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\([^)]*\)\s*{',
                'loop': r'\b(for|while)\s*\(',
                'class': r'\b(public|private|protected)?\s*class\s+\w+\s*{'
            },
            'C++': {
                'function': r'\b\w+\s+\w+\s*\([^)]*\)\s*{',
                'loop': r'\b(for|while)\s*\(',
                'class': r'\bclass\s+\w+\s*{'
            }
        }
        self.error_patterns = {
            'Java': r'\b\w+\.\w+\s*\(|.*\[.*\]',  # Method calls or array access
            'C++': r'\bstd::\w+\b.*\[.*\]|.*\[.*\]',  # Vector/array access
        }

    def parse_python_code(self, code_snippet: str) -> Optional[ast.AST]:
        """
        Parses Python code into an AST using the ast module.

        Args:
            code_snippet (str): The Python code to parse.

        Returns:
            ast.AST or None: The parsed AST or None if parsing fails.
        """
        try:
            return ast.parse(code_snippet)
        except SyntaxError as e:
            print(f"Error parsing Python code: {str(e)}")
            return None

    def find_error_location_python(self, tree: ast.AST, line_number: Optional[str], code_lines: List[str]) -> Dict[str, Optional[str]]:
        """
        Identifies the code block at the specified line number in Python AST.

        Args:
            tree (ast.AST): The parsed Python AST.
            line_number (str): The line number from the error log.
            code_lines (List[str]): Lines of the code snippet.

        Returns:
            dict: Details of the affected code block (type, text, start/end lines).
        """
        if not tree or not line_number:
            return {"node_type": None, "node_text": None, "start_line": None, "end_line": None}

        try:
            line_num = int(line_number) - 1  # Convert to 0-based indexing
            result = {"node_type": None, "node_text": None, "start_line": None, "end_line": None}

            def traverse(node):
                # Prioritize innermost node that exactly matches the line number
                if hasattr(node, 'lineno') and node.lineno == line_num + 1:
                    result["node_type"] = type(node).__name__
                    result["node_text"] = code_lines[line_num].strip() if 0 <= line_num < len(code_lines) else None
                    result["start_line"] = node.lineno
                    result["end_line"] = getattr(node, 'end_lineno', node.lineno) or node.lineno
                    return True
                for child in ast.iter_child_nodes(node):
                    if traverse(child):
                        return True
                return False

            traverse(tree)
            return result
        except ValueError:
            return {"node_type": None, "node_text": None, "start_line": None, "end_line": None}

    def find_error_location_non_python(self, code_snippet: str, language: str, line_number: Optional[str], code_lines: List[str]) -> Dict[str, Optional[str]]:
        """
        Identifies the code block at the specified line number using regex for Java/C++.

        Args:
            code_snippet (str): The code snippet.
            language (str): The programming language.
            line_number (str): The line number from the error log.
            code_lines (List[str]): Lines of the code snippet.

        Returns:
            dict: Details of the affected code block (type, text, start/end lines).
        """
        if not code_snippet or not line_number or language not in self.error_patterns:
            return {"node_type": None, "node_text": None, "start_line": None, "end_line": None}

        try:
            line_num = int(line_number) - 1  # Convert to 0-based indexing
            node_text = code_lines[line_num].strip() if 0 <= line_num < len(code_lines) else None
            if not node_text:
                return {"node_type": None, "node_text": None, "start_line": None, "end_line": None}

            # Check if the line matches error-prone patterns (e.g., method calls, array access)
            node_type = "statement"
            if re.search(self.error_patterns[language], node_text, re.IGNORECASE):
                node_type = "function_call" if '.' in node_text else "array_access"
            elif re.search(r'\b(for|while)\s*\(', node_text, re.IGNORECASE):
                node_type = "loop"
            return {
                "node_type": node_type,
                "node_text": node_text,
                "start_line": line_number,
                "end_line": line_number
            }
        except ValueError:
            return {"node_type": None, "node_text": None, "start_line": None, "end_line": None}

    def infer_intent(self, code_snippet: str, language: str) -> List[str]:
        """
        Infers programmer intent using regex for Java/C++ or AST for Python.

        Args:
            code_snippet (str): The code snippet.
            language (str): The programming language.

        Returns:
            list: List of inferred intent keywords (e.g., 'loop', 'function', 'class').
        """
        intent = set()
        if language == 'Python':
            tree = self.parse_python_code(code_snippet)
            if tree:
                def traverse_python(node):
                    if isinstance(node, ast.FunctionDef):
                        intent.add('function')
                    elif isinstance(node, (ast.For, ast.While)):
                        intent.add('loop')
                    elif isinstance(node, ast.ClassDef):
                        intent.add('class')
                    for child in ast.iter_child_nodes(node):
                        traverse_python(child)
                traverse_python(tree)
        else:
            patterns = self.intent_patterns.get(language, {})
            for intent_type, pattern in patterns.items():
                if re.search(pattern, code_snippet, re.IGNORECASE):
                    intent.add(intent_type)
        return list(intent)
    
    def analyze(self, code_snippet: str, error_data: Dict[str, Optional[str]]) -> Dict[str, Optional[any]]:
        """
        Analyzes code to identify error locations and programmer intent.

        Args:
            code_snippet (str): The code snippet to analyze.
            error_data (dict): Error data from error_log_analysis.

        Returns:
            dict: Analysis results with language, error location, intent, and extracted variables.
        """
        # Detect language using Stage 2 module
        language = self.language_detector.detect_language(code_snippet)
        code_lines = code_snippet.splitlines()

        # Initialize default result
        result = {
            "language": language,
            "error_location": {"node_type": None, "node_text": None, "start_line": None, "end_line": None},
            "intent": [],
            "code_variables": []  # Variables extracted from the actual code
        }

        if language not in ['Python', 'Java', 'C++']:
            return result

        # Parse code and find error location
        if language == 'Python':
            tree = self.parse_python_code(code_snippet)
            result["error_location"] = self.find_error_location_python(tree, error_data.get("line_number"), code_lines)
        else:
            result["error_location"] = self.find_error_location_non_python(code_snippet, language, error_data.get("line_number"), code_lines)

        # Extract variables from the problematic code line if we found it
        error_line = result["error_location"].get("node_text")
        if error_line:
            result["code_variables"] = self.extract_variables_from_code_line(error_line, language)
        
        # If we didn't find variables from the error line, try extracting from the entire snippet
        if not result["code_variables"]:
            # Extract variables from the line number mentioned in error_data
            line_num = error_data.get("line_number")
            if line_num and line_num.isdigit():
                line_idx = int(line_num) - 1
                if 0 <= line_idx < len(code_lines):
                    result["code_variables"] = self.extract_variables_from_code_line(code_lines[line_idx], language)

        # Infer intent
        result["intent"] = self.infer_intent(code_snippet, language)

        return result

    def extract_variables_from_code_line(self, code_line: str, language: str) -> List[str]:
        """
        Extract variable names from a specific code line.
        
        Args:
            code_line (str): The code line to analyze
            language (str): Programming language
            
        Returns:
            List[str]: List of variable names found in the code line        """
        variables = []
        
        if language == 'Python':
            # Extract Python variables using regex patterns
            # Variable assignments: var = value
            assignments = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code_line)
            variables.extend(assignments)
            
            # Array/list access: var[index] or var.method()
            array_access = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[\[\.]', code_line)
            variables.extend(array_access)
            
            # Variables inside brackets: array[variable]
            bracket_vars = re.findall(r'\[([a-zA-Z_][a-zA-Z0-9_]*)\]', code_line)
            variables.extend(bracket_vars)
            
            # Function calls with variables: func(var1, var2)
            func_args = re.findall(r'\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[,\)]', code_line)
            variables.extend(func_args)
            
            # Multiple function arguments: func(var1, var2, var3)
            func_multi_args = re.findall(r',\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[,\)]', code_line)
            variables.extend(func_multi_args)
            
            # Variables in expressions: var + other, var - other, var * other
            expr_vars = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[+\-*/]', code_line)
            variables.extend(expr_vars)
              # Variables after operators: + var, - var, * var
            post_op_vars = re.findall(r'[+\-*/]\s*([a-zA-Z_][a-zA-Z0-9_]*)', code_line)
            variables.extend(post_op_vars)
            
        elif language == 'Java':
            # Java variable patterns
            # Object method calls: obj.method()
            obj_calls = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\.', code_line)
            variables.extend(obj_calls)
            
            # Array access: array[index]
            array_access = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[', code_line)
            variables.extend(array_access)
            
            # Variables inside brackets: array[variable]
            bracket_vars = re.findall(r'\[([a-zA-Z_][a-zA-Z0-9_]*)\]', code_line)
            variables.extend(bracket_vars)
            
            # Variable declarations: Type var = value
            declarations = re.findall(r'\b(?:int|String|double|float|boolean|char|Object|List|ArrayList)\s+([a-zA-Z_][a-zA-Z0-9_]*)', code_line)
            variables.extend(declarations)
            
            # Variable assignments: var = value
            assignments = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code_line)
            variables.extend(assignments)
            
            # Variables in expressions: var * other, var + other
            expr_vars = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[+\-*/]', code_line)
            variables.extend(expr_vars)
              # Variables after operators: * var, + var
            post_op_vars = re.findall(r'[+\-*/]\s*([a-zA-Z_][a-zA-Z0-9_]*)', code_line)
            variables.extend(post_op_vars)
            
        elif language == 'C++':
            # C++ variable patterns
            # Vector access: vec[index] or vec.at(index)
            vector_access = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[\[\.]', code_line)
            variables.extend(vector_access)
            
            # Variables inside brackets: vector[variable]
            bracket_vars = re.findall(r'\[([a-zA-Z_][a-zA-Z0-9_]*)\]', code_line)
            variables.extend(bracket_vars)
            
            # Pointer dereferencing: *ptr
            pointer_deref = re.findall(r'\*\s*([a-zA-Z_][a-zA-Z0-9_]*)', code_line)
            variables.extend(pointer_deref)
            
            # Variable declarations: type var = value
            declarations = re.findall(r'\b(?:int|double|float|char|std::string|std::vector)\s*<?[^>]*>?\s+([a-zA-Z_][a-zA-Z0-9_]*)', code_line)
            variables.extend(declarations)
            
            # Variable assignments: var = value
            assignments = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code_line)
            variables.extend(assignments)
            
            # Variables in expressions: var + other, var * other
            expr_vars = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[+\-*/]', code_line)
            variables.extend(expr_vars)
            
            # Variables after operators: + var, * var
            post_op_vars = re.findall(r'[+\-*/]\s*([a-zA-Z_][a-zA-Z0-9_]*)', code_line)
            variables.extend(post_op_vars)
        
        # Remove duplicates and filter valid identifiers
        filtered_vars = []
        exclude_words = {'if', 'else', 'for', 'while', 'class', 'def', 'function', 'return', 'print', 'console', 'std', 'cout', 'cin', 'endl', 'System', 'out', 'println', 'public', 'private', 'static', 'void', 'main', 'args', 'new', 'delete', 'include', 'using', 'namespace'}
        
        for var in variables:
            if var and var.lower() not in exclude_words and var.isidentifier() and len(var) > 1:
                if var not in filtered_vars:
                    filtered_vars.append(var)
                    
        return filtered_vars

# Example usage
if __name__ == "__main__":
    analyzer = StaticCodeAnalyzer()

    # Test case: Python code with IndexError
    python_code = """
def process_list(items):
    for i in range(10):
        print(items[i])  # Error at line 3
"""
    error_data = {
        "error_type": "IndexError",
        "line_number": "3",
        "variables": ["items"],
        "message": "IndexError: list index out of range at line 3",
        "classification": "runtime"
    }

    # Test case: Java code with NullPointerException
    java_code = """
public class Main {
    public static void main(String[] args) {
        String s = null;
        System.out.println(s.length()); // Error at line 4
    }
}
"""
    java_error_data = {
        "error_type": "NullPointerException",
        "line_number": "4",
        "variables": ["s"],
        "message": "NullPointerException: Cannot invoke method on null object at line 4",
        "classification": "runtime"
    }

    # Test case: C++ code with out_of_range
    cpp_code = """
#include <vector>
int main() {
    std::vector<int> vec;
    int x = vec[10]; // Error at line 4
    return 0;
}
"""
    cpp_error_data = {
        "error_type": "out_of_range",
        "line_number": "4",
        "variables": ["vec"],
        "message": "vector::_M_range_check: __n (which is 10) >= this->size() at line 4",
        "classification": "runtime"
    }

    print("Python Code Analysis:")
    print(analyzer.analyze(python_code, error_data))

    print("\nJava Code Analysis:")
    print(analyzer.analyze(java_code, java_error_data))

    print("\nC++ Code Analysis:")
    print(analyzer.analyze(cpp_code, cpp_error_data))