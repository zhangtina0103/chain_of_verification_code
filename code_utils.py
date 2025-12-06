"""Utilities for code extraction and validation"""

import re
from typing import Tuple, Optional


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from LLM response.
    Handles markdown code blocks and bare code.
    """
    # Remove markdown code blocks
    code = response.strip()

    # Pattern 1: ```python ... ```
    python_block = re.search(r'```python\s*\n(.*?)\n```', code, re.DOTALL)
    if python_block:
        return python_block.group(1).strip()

    # Pattern 2: ``` ... ```
    generic_block = re.search(r'```\s*\n(.*?)\n```', code, re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()

    # Pattern 3: No markdown, return as-is
    return code


def extract_function_name(code: str) -> Optional[str]:
    """Extract the main function name from code"""
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else None


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Python syntax without executing.
    Returns (is_valid, error_message)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def safe_eval_literal(literal_str: str):
    """Safely evaluate Python literal (list, dict, tuple, etc.)"""
    import ast
    try:
        return ast.literal_eval(literal_str)
    except:
        raise ValueError(f"Could not parse as Python literal: {literal_str[:100]}")
