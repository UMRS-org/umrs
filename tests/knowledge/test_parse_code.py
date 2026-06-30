"""Tests for the code parser (Python, C++, CUDA)."""

from __future__ import annotations

from umrs.knowledge.parse_code import parse_code


# ── Python Tests ────────────────────────────────────────────────────────────


def test_parse_python_simple_function():
    """Test parsing a Python file with a simple function."""
    source = b"""
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""

    result = parse_code("math_utils.py", source)

    assert result.error is None
    assert result.language == "python"
    assert len(result.functions) == 1
    assert result.functions[0].name == "add"
    assert "add" in result.apis
    assert "a" in result.functions[0].parameters
    assert "b" in result.functions[0].parameters


def test_parse_python_class_with_methods():
    """Test parsing a Python file with a class and methods."""
    source = b"""
class Calculator:
    \"\"\"A simple calculator.\"\"\"

    def __init__(self):
        self.result = 0

    def add(self, x, y):
        return x + y
"""

    result = parse_code("calculator.py", source)

    assert result.error is None
    assert result.language == "python"
    assert len(result.classes) == 1
    assert result.classes[0].name == "Calculator"
    assert len(result.classes[0].methods) == 2
    assert result.classes[0].methods[0].name == "__init__"
    assert result.classes[0].methods[1].name == "add"


def test_parse_python_imports():
    """Test parsing a Python file with imports."""
    source = b"""
import numpy as np
from typing import List, Dict
import os.path
"""

    result = parse_code("imports.py", source)

    assert result.error is None
    assert "numpy" in result.dependencies
    assert "typing" in result.dependencies
    assert "os.path" in result.dependencies


def test_parse_python_main_block():
    """Test parsing a Python file with a main block."""
    source = b"""
def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
"""

    result = parse_code("main.py", source)

    assert result.error is None
    assert result.execution_flow == "has_main_block"


def test_parse_python_algorithm_heuristic():
    """Test that functions with 'algorithm' in docstring are detected as algorithms."""
    source = b"""
def sort_data(data):
    \"\"\"Sort data using the quicksort algorithm.\"\"\"
    return sorted(data)
"""

    result = parse_code("sort.py", source)

    assert result.error is None
    assert "sort_data" in result.algorithms


def test_parse_python_syntax_error():
    """Test parsing a Python file with a syntax error."""
    source = b"def broken(:\n"

    result = parse_code("broken.py", source)

    assert result.error is not None
    assert "syntax" in result.error.lower()


# ── C++/CUDA Tests ──────────────────────────────────────────────────────────


def test_parse_cpp_function():
    """Test parsing a C++ file with a function."""
    source = b"""
#include <iostream>

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
"""

    result = parse_code("main.cpp", source)

    assert result.error is None
    assert result.language == "cpp"
    assert "iostream" in result.dependencies
    assert result.execution_flow == "has_main"


def test_parse_cuda_kernel():
    """Test parsing a CUDA file with a kernel function."""
    source = b"""
#include <cuda_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""

    result = parse_code("vector_add.cu", source)

    assert result.error is None
    assert result.language == "cuda"
    assert len(result.kernels) == 1
    assert result.kernels[0].name == "vector_add"
    assert result.kernels[0].is_kernel
    assert "vector_add" in result.apis


def test_parse_cpp_class():
    """Test parsing a C++ file with a class."""
    source = b"""
class Vector3 {
public:
    float x, y, z;
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
};
"""

    result = parse_code("vector3.cpp", source)

    assert result.error is None
    assert result.language == "cpp"
    assert len(result.classes) == 1
    assert result.classes[0].name == "Vector3"


def test_parse_cpp_includes():
    """Test parsing a C++ file with includes."""
    source = b"""
#include <vector>
#include <string>
#include "my_header.h"
"""

    result = parse_code("includes.cpp", source)

    assert result.error is None
    assert "vector" in result.dependencies
    assert "string" in result.dependencies
    assert "my_header.h" in result.dependencies


# ── Edge Cases ──────────────────────────────────────────────────────────────


def test_parse_empty_file():
    """Test parsing an empty file."""
    result = parse_code("empty.py", b"")

    assert result.error is None
    assert result.language == "python"
    assert len(result.functions) == 0
    assert len(result.classes) == 0


def test_parse_unknown_extension():
    """Test parsing a file with an unknown extension."""
    result = parse_code("data.xyz", b"some content")

    assert result.error is None
    assert result.language == "unknown"


def test_parse_python_summary_contains_function_count():
    """Test that the summary contains the function count."""
    source = b"""
def func1():
    pass

def func2():
    pass
"""

    result = parse_code("funcs.py", source)

    assert result.error is None
    assert "Functions: 2" in result.summary


def test_parse_python_tags_contain_function_names():
    """Test that tags contain function names."""
    source = b"""
def my_function():
    pass
"""

    result = parse_code("tags.py", source)

    assert result.error is None
    assert "fn:my_function" in result.tags


def test_parse_python_tags_contain_class_names():
    """Test that tags contain class names."""
    source = b"""
class MyClass:
    pass
"""

    result = parse_code("tags.py", source)

    assert result.error is None
    assert "class:MyClass" in result.tags


def test_parse_python_tags_contain_kernel_names():
    """Test that tags contain kernel names."""
    source = b"""
__global__ void my_kernel() {
}
"""

    result = parse_code("kernel.cu", source)

    assert result.error is None
    assert "kernel:my_kernel" in result.tags
