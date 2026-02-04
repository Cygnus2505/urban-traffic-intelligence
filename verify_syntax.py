#!/usr/bin/env python
"""Script to verify the code structure without requiring all dependencies."""
import sys
import ast
from pathlib import Path

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)

def main():
    """Check all Python files for syntax errors."""
    root = Path(__file__).parent
    python_files = list(root.rglob("*.py"))
    
    print(f"Checking {len(python_files)} Python files for syntax errors...")
    print("=" * 80)
    
    errors = []
    for py_file in python_files:
        if '__pycache__' in str(py_file) or '.venv' in str(py_file):
            continue
        
        valid, error = check_python_syntax(py_file)
        if valid:
            print(f"✓ {py_file.relative_to(root)}")
        else:
            print(f"✗ {py_file.relative_to(root)}: {error}")
            errors.append((py_file, error))
    
    print("=" * 80)
    if errors:
        print(f"\n{len(errors)} file(s) with syntax errors:")
        for file, error in errors:
            print(f"  - {file}: {error}")
        return 1
    else:
        print(f"\n✓ All {len(python_files)} files have valid Python syntax!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
