########################################################################
# BASELINE CODE GENERATION PROMPTS
########################################################################

# 4 different categories for baseline
BASELINE_BY_CATEGORY = {
    "algorithms": """Write a single Python function that solves this algorithmic task.

Requirements:
- Use no external libraries beyond standard library (collections, heapq, itertools, etc. are OK)
- Output ONLY valid Python code with no markdown formatting
- No explanation or comments
- Must be a complete, runnable function

Task:
{question}

Python code:""",

    "debugging": """You are fixing broken Python code.

The user has described a bug or provided failing code.

Task:
- Correct the code to fix all bugs
- Preserve the same function signature and interface
- Output ONLY working code with no markdown formatting
- No explanation

User query:
{question}

Fixed Python code:""",

    "api_usage": """Write a single Python function that demonstrates correct usage of this API or library.

Rules:
- Include necessary imports at the top
- Create one complete function (not a class or app)
- Do NOT include example usage or main() calls
- Must be runnable without modification
- Output ONLY Python code with no markdown formatting
- No explanation

Task:
{question}

Python code:""",

    "data_processing": """Write a single Python function that loads, transforms, and returns structured data.

Requirements:
- Use ONLY standard Python libraries (json, csv, re, etc.)
- Must be a single function that takes input and returns output
- Output ONLY code with no markdown formatting
- No explanation

Task:
{question}

Python code:"""
}

########################################################################
# VERIFICATION TEST GENERATION PROMPTS
########################################################################

VERIFY_PLAN_BY_CATEGORY = {
    "algorithms": """Generate 5-8 test cases for this function.

Return a Python list of tuples where each tuple is (input_args, expected_output).

Rules:
- Output ONLY a Python list literal (no variables, no code, just the list)
- Include edge cases (empty input, single element, large input, etc.)
- Use only Python literals (no function calls)
- Format: [(input1, expected1), (input2, expected2), ...]
- For functions with multiple args, use tuple: (((arg1, arg2), expected_output))

Task:
{question}

Candidate function:
```python
{baseline_code}
```

Test cases as Python list:""",

    "debugging": """Generate 3-5 test inputs that are LIKELY to trigger bugs or edge case failures.

Return ONLY a Python list of input values.

We don't know the expected outputs - we're testing if the code runs without errors.

Rules:
- Output ONLY a Python list literal
- Focus on edge cases that might break the code
- Format: [input1, input2, input3, ...]

Task:
{question}

Candidate code:
```python
{baseline_code}
```

Test inputs as Python list:""",

    "api_usage": """Generate minimal Python code that calls this function and prints the result.

This code will test if the API usage is correct.

Rules:
- Output ONLY Python code
- Import any mock data needed
- Call the function with realistic arguments
- Print or return the result
- Keep it minimal (2-5 lines)

Candidate function:
```python
{baseline_code}
```

Test harness code:""",

    "data_processing": """Generate sample input data and expected output for this data processing function.

Return ONLY valid Python code that defines two variables:
- TEST_INPUT = <sample input data>
- EXPECTED_OUTPUT = <expected result>

Rules:
- Use Python literals only (strings, lists, dicts)
- Make the test realistic but small
- Output ONLY the two variable assignments

Task:
{question}

Candidate function:
```python
{baseline_code}
```

Test data:"""
}


########################################################################
# REFINEMENT PROMPTS
########################################################################

FINAL_REWRITE_BY_CATEGORY = {
    "algorithms": """The function failed one or more test cases.

Original task:
{question}

Your previous code:
```python
{baseline_code}
```

Test failures:
{failures}

Rewrite the function to pass all test cases.

Requirements:
- Output ONLY the corrected Python code
- No markdown formatting
- No explanation

Corrected code:""",

    "debugging": """The function raised errors when executed on test inputs.

Original task:
{question}

Your previous code:
```python
{baseline_code}
```

Execution errors:
{stderr}

Failed on inputs: {failed_inputs}

Rewrite the code to eliminate ALL errors.

Requirements:
- Output ONLY the corrected Python code
- No markdown formatting
- No explanation

Corrected code:""",

    "api_usage": """The generated function failed when executed.

Task:
{question}

Your previous code:
```python
{baseline_code}
```

Error message:
{stderr}

Rewrite the function to correctly use the API.

Requirements:
- Output ONLY the corrected Python code
- No markdown formatting
- No explanation

Corrected code:""",

    "data_processing": """The function output did not match expected result.

Task:
{question}

Your previous code:
```python
{baseline_code}
```

Expected output:
{expected}

Actual output:
{actual}

Rewrite the function to produce the correct result.

Requirements:
- Output ONLY the corrected Python code
- No markdown formatting
- No explanation

Corrected code:"""
}


########################################################################
# ROUTER PROMPT
########################################################################

ROUTER_CHAIN_PROMPT = """Classify this coding question into exactly ONE category.

Categories:
- algorithms: Writing pure functions on data structures (sorting, searching, graph traversal, dynamic programming, etc.)
- debugging: Fixing broken code, handling errors, or correcting failing implementations
- api_usage: Using external libraries or frameworks (requests, FastAPI, pandas, numpy, OpenAI API, etc.)
- data_processing: Reading, transforming, or writing structured data (CSV, JSON, XML, logs, etc.)

Examples:

algorithms:
- "Implement breadth-first search"
- "Write a function to find the longest palindrome substring"
- "Sort an array using merge sort"

debugging:
- "Fix this Python function that crashes on empty lists"
- "Why does my recursive function cause stack overflow?"
- "Debug this code that returns wrong results"

api_usage:
- "Write a FastAPI POST endpoint"
- "Use requests to fetch data from an API"
- "Create a pandas DataFrame from a dictionary"

data_processing:
- "Parse CSV and return dict sorted by key"
- "Convert JSON to nested dictionary"
- "Extract email addresses from log file"

Question: {question}

Output ONLY one word (the category name):"""
