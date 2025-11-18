import numpy as np
import sys
from io import StringIO
from typing import List, Dict, Tuple
import contextlib
from human_eval.data import read_problems
from datasets import load_dataset
import os, json


"""
First compute number of successes
"""
def execute_code(code: str, test_case: str, timeout: int = 5) -> bool:
    """
    Execute a code string with a test case and return whether it passes.

    Args:
        code: The function implementation as a string
        test_case: The test case to run (can be assertion or function call)
        timeout: Maximum execution time in seconds

    Returns:
        True if test passes, False otherwise
    """
    try:
        # Create a namespace for execution
        namespace = {}

        # Execute the function definition
        exec(code, namespace)

        # Execute the test case
        exec(test_case, namespace)

        return True
    except AssertionError:
        # Test case assertion failed
        return False
    except Exception as e:
        # Any other error (syntax, runtime, etc.)
        return False


def num_successes(code: str, test_cases: List[str]) -> int:
    """
    Compute the number of test cases that a code snippet passes.

    Args:
        code: The function implementation as a string
        test_cases: List of test case strings to evaluate

    Returns:
        Number of successful test cases
    """
    successes = 0

    for test_case in test_cases:
        if execute_code(code, test_case):
            successes += 1

    return successes

def evaluate_samples(samples: List[str], test_cases: List[str]) -> Tuple[int, int]:
    """
    Evaluate multiple code samples against test cases.

    Args:
        samples: List of generated code samples (strings)
        test_cases: List of test cases to evaluate against

    Returns:
        Tuple of (n, c) where:
        - n: total number of samples
        - c: number of correct samples (samples that pass all tests)
    """
    n = len(samples)
    c = 0

    for sample in samples:
        # A sample is correct if it passes ALL test cases
        if num_successes(sample, test_cases) == len(test_cases):
            c += 1

    return n, c

"""
Compute pass@k
"""
def pass_at_k(n: int, c: int, k: int) -> float:
    """
    n: total number of samples
    c: number of correct samples
    k: k in pass@k
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k /
                         np.arange(n - c + 1, n + 1))


def evaluate_humaneval_problem(problem: Dict, generated_samples: List[str]) -> Dict:
    """
    Evaluate generated samples for a HumanEval problem.

    Args:
        problem: Dictionary containing:
            - 'prompt': function signature and docstring
            - 'test': test cases as string
            - 'entry_point': function name
        generated_samples: List of generated function implementations

    Returns:
        Dictionary with evaluation results
    """
    # Extract test cases (HumanEval format has them in a single string)
    test_code = problem['test']

    # Combine prompt with each generated sample
    full_samples = [problem['prompt'] + sample for sample in generated_samples]

    # Count correct samples
    c = 0
    for sample in full_samples:
        try:
            # Execute code and tests together
            namespace = {}
            exec(sample, namespace)
            exec(test_code, namespace)

            # Call the check function
            namespace['check'](namespace[problem['entry_point']])
            c += 1
        except:
            pass

    n = len(generated_samples)

    return {
        'n': n,
        'c': c,
        'pass@1': pass_at_k(n, c, 1) if n >= 1 else 0,
        'pass@10': pass_at_k(n, c, 10) if n >= 10 else 0,
    }


def evaluate_mbpp_problem(problem: Dict, generated_samples: List[str]) -> Dict:
    """
    Evaluate generated samples for an MBPP problem.

    Args:
        problem: Dictionary containing:
            - 'text': problem description
            - 'code': reference solution
            - 'test_list': list of test assertions
        generated_samples: List of generated function implementations

    Returns:
        Dictionary with evaluation results
    """
    test_cases = problem['test_list']

    n, c = evaluate_samples(generated_samples, test_cases)

    return {
        'n': n,
        'c': c,
        'pass@1': pass_at_k(n, c, 1) if n >= 1 else 0,
        'pass@10': pass_at_k(n, c, 10) if n >= 10 else 0,
    }

########################################
# Load JSON outputs
########################################
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


########################################
# Evaluate HumanEval
########################################
def evaluate_all_humaneval(output_dir):
    problems = read_problems()
    results = []

    for task_id, problem in problems.items():
        json_path = os.path.join(output_dir, f"{task_id.replace('/', '_')}.json")
        if not os.path.exists(json_path):
            print("MISSING:", json_path)
            continue

        data = load_json(json_path)
        samples = data["samples"]

        scores = evaluate_humaneval_problem(problem, samples)
        results.append(scores)

    pass1  = np.mean([r["pass@1"]  for r in results])
    pass10 = np.mean([r["pass@10"] for r in results])

    print("HUMANEVAL pass@1:",  pass1)
    print("HUMANEVAL pass@10:", pass10)

########################################
# Evaluate MBPP
########################################
def evaluate_all_mbpp(output_dir):
    mbpp = load_dataset("mbpp", "full")["test"]
    results = []

    for ex in mbpp:
        task_id = f"MBPP_{ex['task_id']}.json"
        json_path = os.path.join(output_dir, task_id)
        if not os.path.exists(json_path):
            print("MISSING:", json_path)
            continue

        data = load_json(json_path)
        samples = data["samples"]

        problem = {
            "text": ex["text"],
            "code": ex["code"],
            "test_list": ex["test_list"],
        }

        scores = evaluate_mbpp_problem(problem, samples)
        results.append(scores)

    pass1  = np.mean([r["pass@1"]  for r in results])
    pass10 = np.mean([r["pass@10"] for r in results])

    print("MBPP pass@1:",  pass1)
    print("MBPP pass@10:", pass10)

"""
Assuming generated code form:
model_outputs/
│
├── humaneval/
│   ├── HumanEval_0.json
│   ├── HumanEval_1.json
│   ├── HumanEval_2.json
│   └── ...
│
└── mbpp/
    ├── MBPP_0.json
    ├── MBPP_1.json
    ├── MBPP_2.json
    └── ...
"""

########################################
if __name__ == "__main__":
    evaluate_all_humaneval("model_outputs/humaneval/")
    evaluate_all_mbpp("model_outputs/mbpp/")

    # Example HumanEval problem
#     humaneval_problem = {
#         'prompt': 'def add(a, b):\n    """Add two numbers"""\n',
#         'test': '''
# def check(candidate):
#     assert candidate(1, 2) == 3
#     assert candidate(0, 0) == 0
#     assert candidate(-1, 1) == 0
# ''',
#         'entry_point': 'add'
#     }

#     # Example generated samples
#     samples_humaneval = [
#         '    return a + b\n',  # Correct
#         '    return a - b\n',  # Incorrect
#         '    return a + b\n',  # Correct
#     ]

#     print("HumanEval Evaluation:")
#     results = evaluate_humaneval_problem(humaneval_problem, samples_humaneval)
#     print(f"Total samples (n): {results['n']}")
#     print(f"Correct samples (c): {results['c']}")
#     print(f"pass@1: {results['pass@1']:.3f}")
#     print()

#     # Example MBPP problem
#     mbpp_problem = {
#         'text': 'Write a function to add two numbers',
#         'code': 'def add(a, b):\n    return a + b',
#         'test_list': [
#             'assert add(1, 2) == 3',
#             'assert add(0, 0) == 0',
#             'assert add(-1, 1) == 0'
#         ]
#     }

#     # Example complete function samples for MBPP
#     samples_mbpp = [
#         'def add(a, b):\n    return a + b',  # Correct
#         'def add(a, b):\n    return a - b',  # Incorrect
#         'def add(a, b):\n    return a + b',  # Correct
#     ]

#     print("MBPP Evaluation:")
#     results = evaluate_mbpp_problem(mbpp_problem, samples_mbpp)
#     print(f"Total samples (n): {results['n']}")
#     print(f"Correct samples (c): {results['c']}")
#     print(f"pass@1: {results['pass@1']:.3f}")
