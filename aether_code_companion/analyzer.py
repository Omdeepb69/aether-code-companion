# aether_code_companion/analyzer.py

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

# Attempt to import StyleLearner, handle potential ImportError if style_learner.py doesn't exist yet
try:
    from .style_learner import StyleLearner
except ImportError:
    # Define a placeholder class if StyleLearner is not available
    # This allows analyzer.py to be imported and used for rule-based checks
    # even if the ML component is not fully implemented or installed.
    class StyleLearner:
        """Placeholder for StyleLearner if not found."""
        def __init__(self, model_path: Optional[str] = None):
            logging.warning("StyleLearner class not found. Style checking will be disabled.")
            self._model = None

        def check_node_style(self, node: ast.AST, source_lines: List[str]) -> List[str]:
            """Placeholder style check method."""
            # In a real implementation, this would analyze the node
            # based on the learned style model.
            return []

        def is_ready(self) -> bool:
            """Check if the style learner model is loaded."""
            return False # Placeholder always returns False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Structures ---

class Suggestion:
    """Represents a single analysis suggestion."""
    def __init__(self, line: int, column: int, code: str, message: str, rule_id: Optional[str] = None, suggestion_type: str = "suggestion"):
        self.line = line
        self.column = column
        self.code = code # The specific code snippet related to the suggestion
        self.message = message
        self.rule_id = rule_id
        self.suggestion_type = suggestion_type # e.g., 'anti-pattern', 'complexity', 'style'

    def __repr__(self) -> str:
        return (f"Suggestion(line={self.line}, col={self.column}, rule='{self.rule_id or 'style'}', "
                f"type='{self.suggestion_type}', msg='{self.message[:50]}...')")

    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion to a dictionary."""
        return {
            "line": self.line,
            "column": self.column,
            "code": self.code,
            "message": self.message,
            "rule_id": self.rule_id,
            "suggestion_type": self.suggestion_type,
        }

# --- Rule Loading ---

def load_rules(rules_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Loads analysis rules from a JSON file.

    Args:
        rules_path: Path to the JSON file containing the rules.

    Returns:
        A dictionary representing the loaded rules.

    Raises:
        FileNotFoundError: If the rules file does not exist.
        json.JSONDecodeError: If the rules file is not valid JSON.
        ValueError: If the rules file format is invalid.
    """
    rules_file = Path(rules_path)
    if not rules_file.is_file():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")

    try:
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        # Basic validation (can be expanded)
        if not isinstance(rules_data, dict):
            raise ValueError("Rules file must contain a JSON object (dictionary).")
        for rule_id, rule_details in rules_data.items():
            if not isinstance(rule_details, dict) or "name" not in rule_details or "description" not in rule_details:
                 raise ValueError(f"Invalid format for rule '{rule_id}'. Must be a dictionary with 'name' and 'description'.")
        logging.info(f"Successfully loaded {len(rules_data)} rules from {rules_path}")
        return rules_data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {rules_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading rules from {rules_path}: {e}")
        raise

# --- AST Visitor for Analysis ---

class CodeAnalyzerVisitor(ast.NodeVisitor):
    """
    Traverses the Abstract Syntax Tree (AST) to find issues based on rules
    and style deviations.
    """
    def __init__(self, rules_data: Dict[str, Dict[str, Any]],
                 style_model: Optional[StyleLearner],
                 source_lines: List[str]):
        """
        Initializes the visitor.

        Args:
            rules_data: Dictionary of loaded rules.
            style_model: An instance of StyleLearner (or None).
            source_lines: List of source code lines (for context and line counts).
        """
        self.rules_data = rules_data
        self.style_model = style_model
        self.source_lines = source_lines
        self.suggestions: List[Suggestion] = []

    def _add_suggestion(self, node: ast.AST, rule_id: str, message_override: Optional[str] = None, code_snippet: Optional[str] = None):
        """Helper method to add a suggestion."""
        if rule_id not in self.rules_data:
            logging.warning(f"Attempted to add suggestion for unknown rule_id: {rule_id}")
            return

        rule = self.rules_data[rule_id]
        line_num = getattr(node, 'lineno', 0)
        col_num = getattr(node, 'col_offset', 0)

        # Try to get a relevant code snippet
        if code_snippet is None:
            try:
                # ast.get_source_segment is available in Python 3.8+
                # For broader compatibility, we might need a fallback or manual slicing
                if hasattr(ast, 'get_source_segment'):
                     code_snippet = ast.get_source_segment(
                         '\n'.join(self.source_lines), node, padded=False
                     ) or f"Code at line {line_num}"
                elif line_num > 0 and line_num <= len(self.source_lines):
                     code_snippet = self.source_lines[line_num - 1].strip() # Fallback to line
                else:
                     code_snippet = f"Code near line {line_num}"
            except Exception: # Catch potential errors in segment retrieval
                code_snippet = f"Code near line {line_num}"


        suggestion = Suggestion(
            line=line_num,
            column=col_num,
            code=code_snippet,
            message=message_override or rule["description"],
            rule_id=rule_id,
            suggestion_type=rule.get("type", "suggestion") # Use rule type or default
        )
        self.suggestions.append(suggestion)
        logging.debug(f"Added suggestion: {suggestion}")

    def _add_style_suggestion(self, node: ast.AST, message: str, code_snippet: Optional[str] = None):
        """Helper method to add a style-based suggestion."""
        line_num = getattr(node, 'lineno', 0)
        col_num = getattr(node, 'col_offset', 0)

        if code_snippet is None:
             try:
                 if hasattr(ast, 'get_source_segment'):
                     code_snippet = ast.get_source_segment(
                         '\n'.join(self.source_lines), node, padded=False
                     ) or f"Code at line {line_num}"
                 elif line_num > 0 and line_num <= len(self.source_lines):
                     code_snippet = self.source_lines[line_num - 1].strip()
                 else:
                     code_snippet = f"Code near line {line_num}"
             except Exception:
                 code_snippet = f"Code near line {line_num}"

        suggestion = Suggestion(
            line=line_num,
            column=col_num,
            code=code_snippet,
            message=message,
            rule_id="style-deviation", # Specific ID for style issues
            suggestion_type="style"
        )
        self.suggestions.append(suggestion)
        logging.debug(f"Added style suggestion: {suggestion}")

    # --- Visitor Methods for Specific Rules ---

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Checks function definitions."""
        # 1. Check for long functions (rule: 'long_function')
        if "long_function" in self.rules_data:
            rule_config = self.rules_data["long_function"].get("config", {})
            max_lines = rule_config.get("max_lines", 50) # Default max lines
            start_line = node.lineno
            # end_lineno is available in Python 3.8+
            end_line = getattr(node, 'end_lineno', start_line)
            if end_line is not None:
                func_length = end_line - start_line + 1
                # A simple line count; could be refined to exclude comments/blank lines
                if func_length > max_lines:
                    message = f"Function '{node.name}' is too long ({func_length} lines). Maximum recommended is {max_lines}."
                    self._add_suggestion(node, "long_function", message_override=message, code_snippet=f"def {node.name}(...):")

        # 2. Check for mutable default arguments (rule: 'mutable_default_arg')
        if "mutable_default_arg" in self.rules_data:
            for default in node.args.defaults:
                # Check if the default value is a list, dict, or set literal/call
                if isinstance(default, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)) or \
                   (isinstance(default, ast.Call) and isinstance(default.func, ast.Name) and default.func.id in ('list', 'dict', 'set')):
                    self._add_suggestion(default, "mutable_default_arg", code_snippet=ast.dump(default)) # Use AST dump for snippet

        # Continue traversing child nodes
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        """Checks except handlers."""
        # Check for bare except clauses (rule: 'bare_except')
        if "bare_except" in self.rules_data and node.type is None:
            self._add_suggestion(node, "bare_except", code_snippet="except:")

        # Continue traversing child nodes
        self.generic_visit(node)

    # --- Generic Visitor for Style Checks ---

    def visit(self, node: ast.AST):
        """Override generic visit to potentially apply style checks."""
        # Apply style check if model is available and ready
        if self.style_model and self.style_model.is_ready():
            try:
                style_issues = self.style_model.check_node_style(node, self.source_lines)
                for issue_msg in style_issues:
                    self._add_style_suggestion(node, issue_msg)
            except Exception as e:
                logging.error(f"Error during style check on node {type(node).__name__}: {e}", exc_info=True)

        # Call the original generic_visit to traverse the tree
        super().visit(node)


# --- Main Analysis Function ---

def analyze_code(file_path: Union[str, Path],
                 rules_data: Dict[str, Dict[str, Any]],
                 style_model: Optional[StyleLearner] = None) -> List[Suggestion]:
    """
    Analyzes a Python file for anti-patterns and style deviations.

    Args:
        file_path: Path to the Python file to analyze.
        rules_data: Dictionary of loaded rules.
        style_model: An optional instance of StyleLearner for style checks.

    Returns:
        A list of Suggestion objects found in the code.
    """
    path = Path(file_path)
    suggestions: List[Suggestion] = []
    logging.info(f"Analyzing file: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            source_lines = source_code.splitlines() # Keep lines for context
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return suggestions # Return empty list
    except IOError as e:
        logging.error(f"Error reading file {path}: {e}")
        return suggestions
    except Exception as e:
        logging.error(f"An unexpected error occurred reading {path}: {e}")
        return suggestions

    try:
        # Parse the source code into an AST
        tree = ast.parse(source_code, filename=str(path))
    except SyntaxError as e:
        logging.error(f"Syntax error in {path} at line {e.lineno}, offset {e.offset}: {e.msg}")
        # Add a suggestion for the syntax error itself
        suggestions.append(Suggestion(
            line=e.lineno or 1,
            column=e.offset or 0,
            code=e.text.strip() if e.text else "Syntax Error Location",
            message=f"Syntax Error: {e.msg}",
            rule_id="syntax-error",
            suggestion_type="error"
        ))
        return suggestions # Stop analysis on syntax error
    except Exception as e:
        logging.error(f"Failed to parse AST for {path}: {e}")
        return suggestions

    try:
        # Create and run the visitor
        visitor = CodeAnalyzerVisitor(rules_data, style_model, source_lines)
        visitor.visit(tree)
        suggestions.extend(visitor.suggestions)
    except Exception as e:
        logging.error(f"Error during AST traversal for {path}: {e}", exc_info=True)
        # Optionally add a generic analysis error suggestion

    logging.info(f"Analysis complete for {path}. Found {len(suggestions)} suggestions.")
    return suggestions

# --- Example Usage (Optional) ---

if __name__ == "__main__":
    # This block is for demonstration and testing purposes.
    # In the actual project, this logic would likely be in main.py or watcher.py.

    # Define paths (relative to where this script might be run from)
    # Adjust these paths as necessary for your project structure.
    EXAMPLE_CODE_FILE = Path("./example_code.py")
    RULES_FILE = Path("./aether_code_companion/rules.json") # Assuming rules.json is in the package dir

    # Create a dummy example code file
    EXAMPLE_CODE_CONTENT = """
import os # Unused import (example - rule not implemented here yet)

class BadClass:
    def __init__(self, items=[]): # Mutable default argument
        self.items = items

    def long_method(self, x):
        # This method is intentionally long for testing
        a = 1
        b = 2
        c = 3
        d = 4
        e = 5
        f = 6
        g = 7
        h = 8
        i = 9
        j = 10
        k = 11
        l = 12
        m = 13
        n = 14
        o = 15
        p = 16
        q = 17
        r = 18
        s = 19
        t = 20
        u = 21
        v = 22
        w = 23
        x_var = 24 # Reusing parameter name (style issue - example)
        y = 25
        z = 26
        aa = 27
        bb = 28
        cc = 29
        dd = 30
        ee = 31
        ff = 32
        gg = 33
        hh = 34
        ii = 35
        jj = 36
        kk = 37
        ll = 38
        mm = 39
        nn = 40
        oo = 41
        pp = 42
        qq = 43
        rr = 44
        ss = 45
        tt = 46
        uu = 47
        vv = 48
        ww = 49
        xx = 50
        yy = 51 # Should trigger long_function with max_lines=50
        return x * yy

def process_data(data):
    try:
        result = data['value'] / data['count']
    except: # Bare except
        print("An error occurred")
        result = None
    return result

# Example of code that might trigger a style deviation (if model was trained)
def VERY_LONG_FUNCTION_NAME_THAT_IS_NOT_PEP8_COMPLIANT(param1, param2):
    pass

# Syntax error example (uncomment to test)
# def invalid syntax(

"""
    EXAMPLE_CODE_FILE.write_text(EXAMPLE_CODE_CONTENT, encoding='utf-8')

    # Create a dummy rules.json file
    RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    RULES_CONTENT = """
{
  "long_function": {
    "name": "Long Function",
    "description": "Function exceeds the recommended maximum length.",
    "type": "complexity",
    "config": {
      "max_lines": 30
    }
  },
  "bare_except": {
    "name": "Bare Except Clause",
    "description": "Using 'except:' without specifying an exception type can hide errors.",
    "type": "anti-pattern"
  },
  "mutable_default_arg": {
      "name": "Mutable Default Argument",
      "description": "Using mutable default arguments (like lists or dicts) can lead to unexpected behavior.",
      "type": "anti-pattern"
  }
}
"""
    RULES_FILE.write_text(RULES_CONTENT, encoding='utf-8')

    print(f"--- Running Analyzer Example ---")
    print(f"Analyzing: {EXAMPLE_CODE_FILE.resolve()}")
    print(f"Using rules: {RULES_FILE.resolve()}")

    try:
        # Load rules
        rules = load_rules(RULES_FILE)

        # Initialize a placeholder style learner (no actual model)
        # Replace with `StyleLearner(model_path="path/to/your/model.pkl")` if you have one
        style_learner = StyleLearner() # Will log a warning

        # Analyze the code
        found_suggestions = analyze_code(EXAMPLE_CODE_FILE, rules, style_learner)

        # Print suggestions
        if found_suggestions:
            print("\n--- Analysis Suggestions ---")
            for suggestion in found_suggestions:
                print(f"  - L{suggestion.line}:{suggestion.column} [{suggestion.suggestion_type}/{suggestion.rule_id or 'style'}] {suggestion.message}")
                print(f"    Code: {suggestion.code}")
        else:
            print("\n--- No suggestions found. ---")

    except FileNotFoundError as e:
        print(f"\nError: Required file not found: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")
        logging.exception("Detailed traceback:") # Log full traceback if needed

    # Clean up dummy files
    # EXAMPLE_CODE_FILE.unlink()
    # RULES_FILE.unlink() # Keep rules for potential reuse if running again
    print("\n--- Analyzer Example Finished ---")