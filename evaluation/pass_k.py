import numpy as np
import sys
from io import StringIO
from typing import List, Dict, Tuple
import contextlib
from human_eval.data import read_problems
from datasets import load_dataset
import os, json
from datetime import datetime

"""
Enhanced evaluation script with detailed failure logging for both baseline and refined samples.
Creates comprehensive log files showing which test cases failed for each problem.
"""

def execute_code_with_details(code: str, test_case: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Execute a code string with a test case and return whether it passes and error details.

    Args:
        code: The function implementation as a string
        test_case: The test case to run (can be assertion or function call)
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        # Create a namespace for execution
        namespace = {}

        # Execute the function definition
        exec(code, namespace)

        # Execute the test case
        exec(test_case, namespace)

        return True, ""
    except AssertionError as e:
        # Test case assertion failed
        return False, f"AssertionError: {str(e)}"
    except Exception as e:
        # Any other error (syntax, runtime, etc.)
        return False, f"{type(e).__name__}: {str(e)}"


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
        success, _ = execute_code_with_details(code, test_case)
        if success:
            successes += 1

    return successes

def evaluate_samples_with_logging(samples: List[str], test_cases: List[str]) -> Tuple[int, int, List[Dict]]:
    """
    Evaluate multiple code samples against test cases with detailed logging.

    Args:
        samples: List of generated code samples (strings)
        test_cases: List of test cases to evaluate against

    Returns:
        Tuple of (n, c, details) where:
        - n: total number of samples
        - c: number of correct samples (samples that pass all tests)
        - details: List of dicts with detailed results for each sample
    """
    n = len(samples)
    c = 0
    details = []

    for idx, sample in enumerate(samples):
        sample_result = {
            'sample_idx': idx,
            'passed_all': False,
            'num_passed': 0,
            'num_total': len(test_cases),
            'test_results': []
        }

        passed_count = 0
        for test_idx, test_case in enumerate(test_cases):
            success, error_msg = execute_code_with_details(sample, test_case)

            sample_result['test_results'].append({
                'test_idx': test_idx,
                'test_case': test_case,
                'passed': success,
                'error': error_msg if not success else None
            })

            if success:
                passed_count += 1

        sample_result['num_passed'] = passed_count
        sample_result['passed_all'] = (passed_count == len(test_cases))

        if sample_result['passed_all']:
            c += 1

        details.append(sample_result)

    return n, c, details

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    n: total number of samples
    c: number of correct samples
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def extract_humaneval_test_cases(test_code: str) -> List[str]:
    """
    Extract individual test assertions from HumanEval test code.

    Args:
        test_code: The test code containing a check function

    Returns:
        List of individual assert statements
    """
    test_cases = []
    lines = test_code.split('\n')

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('assert '):
            test_cases.append(stripped)

    return test_cases


def evaluate_humaneval_problem_with_logging(problem: Dict, generated_samples: List[str]) -> Dict:
    """
    Evaluate generated samples for a HumanEval problem with detailed logging.

    Args:
        problem: Dictionary containing:
            - 'prompt': function signature and docstring
            - 'test': test cases as string
            - 'entry_point': function name
        generated_samples: List of generated function implementations

    Returns:
        Dictionary with evaluation results and detailed logs
    """
    test_code = problem['test']
    full_samples = [problem['prompt'] + sample for sample in generated_samples]

    # Extract individual test assertions
    test_assertions = extract_humaneval_test_cases(test_code)

    c = 0
    sample_details = []

    for idx, sample in enumerate(full_samples):
        sample_result = {
            'sample_idx': idx,
            'passed_all': False,
            'num_passed': 0,
            'num_total': len(test_assertions),
            'test_results': [],
            'code': generated_samples[idx]
        }

        # First, try to run all tests together
        all_passed = False
        try:
            namespace = {}
            exec(sample, namespace)
            exec(test_code, namespace)
            namespace['check'](namespace[problem['entry_point']])
            all_passed = True
            c += 1
        except Exception as overall_error:
            pass

        # Now test each assertion individually to see which ones fail
        if test_assertions:
            for test_idx, assertion in enumerate(test_assertions):
                test_result = {
                    'test_idx': test_idx,
                    'test_case': assertion,
                    'passed': False,
                    'error': None
                }

                try:
                    namespace = {}
                    exec(sample, namespace)
                    # Create a candidate reference for the assertion
                    namespace['candidate'] = namespace[problem['entry_point']]
                    exec(assertion, namespace)
                    test_result['passed'] = True
                    sample_result['num_passed'] += 1
                except Exception as e:
                    test_result['error'] = f"{type(e).__name__}: {str(e)}"

                sample_result['test_results'].append(test_result)

            sample_result['passed_all'] = all_passed
        else:
            # If we couldn't extract individual assertions, just use overall result
            sample_result['passed_all'] = all_passed
            if not all_passed:
                sample_result['test_results'].append({
                    'test_idx': 0,
                    'test_case': 'check() function',
                    'passed': False,
                    'error': str(overall_error) if 'overall_error' in locals() else 'Unknown error'
                })

        sample_details.append(sample_result)

    n = len(generated_samples)

    return {
        'n': n,
        'c': c,
        'pass@1': pass_at_k(n, c, 1) if n >= 1 else 0,
        'pass@10': pass_at_k(n, c, 10) if n >= 10 else 0,
        'sample_details': sample_details
    }


def evaluate_mbpp_problem_with_logging(problem: Dict, generated_samples: List[str]) -> Dict:
    """
    Evaluate generated samples for an MBPP problem with detailed logging.

    Args:
        problem: Dictionary containing:
            - 'text': problem description
            - 'code': reference solution
            - 'test_list': list of test assertions
        generated_samples: List of generated function implementations

    Returns:
        Dictionary with evaluation results and detailed logs
    """
    test_cases = problem['test_list']
    n, c, details = evaluate_samples_with_logging(generated_samples, test_cases)

    return {
        'n': n,
        'c': c,
        'pass@1': pass_at_k(n, c, 1) if n >= 1 else 0,
        'pass@10': pass_at_k(n, c, 10) if n >= 10 else 0,
        'sample_details': details
    }

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_log_file(log_path: str, results: List[Dict], dataset_name: str, sample_type: str):
    """
    Write detailed failure logs to a file.

    Args:
        log_path: Path to write the log file
        results: List of result dictionaries
        dataset_name: Name of the dataset (HumanEval or MBPP)
        sample_type: Type of samples (baseline or refined)
    """
    with open(log_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"{dataset_name.upper()} - {sample_type.upper()} SAMPLES - DETAILED FAILURE LOG\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

        total_problems = len(results)
        total_samples = sum(r['n'] for r in results)
        total_correct = sum(r['c'] for r in results)
        avg_pass1 = np.mean([r['pass@1'] for r in results])

        f.write(f"SUMMARY:\n")
        f.write(f"  Total problems evaluated: {total_problems}\n")
        f.write(f"  Total samples: {total_samples}\n")
        f.write(f"  Correct samples: {total_correct}\n")
        f.write(f"  Average pass@1: {avg_pass1:.4f}\n")

        if all(r['n'] >= 10 for r in results):
            avg_pass10 = np.mean([r['pass@10'] for r in results])
            f.write(f"  Average pass@10: {avg_pass10:.4f}\n")

        f.write(f"\n{'='*80}\n\n")

        # Write details for each problem
        for result in results:
            task_id = result['task_id']
            f.write(f"\nPROBLEM: {task_id}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Samples: {result['n']}, Correct: {result['c']}, pass@1: {result['pass@1']:.4f}\n\n")

            if 'sample_details' in result:
                for detail in result['sample_details']:
                    sample_idx = detail['sample_idx']

                    # Both MBPP and enhanced HumanEval now have test_results
                    if 'test_results' in detail and detail['test_results']:
                        passed = detail.get('passed_all', False)
                        num_passed = detail.get('num_passed', 0)
                        num_total = detail.get('num_total', len(detail['test_results']))

                        status = "✓ PASSED ALL" if passed else "✗ FAILED"
                        f.write(f"  Sample {sample_idx}: {status} ({num_passed}/{num_total} tests passed)\n")

                        if not passed:
                            # Show which specific test cases failed
                            for test_result in detail['test_results']:
                                if not test_result['passed']:
                                    test_idx = test_result['test_idx']
                                    test_case = test_result['test_case']
                                    error = test_result.get('error', 'Unknown error')
                                    f.write(f"    ✗ Test {test_idx} FAILED:\n")
                                    f.write(f"      Test: {test_case}\n")
                                    f.write(f"      Error: {error}\n")
                        else:
                            # Optionally show passed tests (commented out to keep logs concise)
                            # for test_result in detail['test_results']:
                            #     if test_result['passed']:
                            #         test_idx = test_result['test_idx']
                            #         test_case = test_result['test_case']
                            #         f.write(f"    ✓ Test {test_idx} passed: {test_case}\n")
                            pass
                    else:
                        # Fallback for old format (shouldn't happen with new code)
                        passed = detail.get('passed', False)
                        status = "✓ PASSED" if passed else "✗ FAILED"
                        f.write(f"  Sample {sample_idx}: {status}\n")

                        if not passed and 'error' in detail and detail['error']:
                            f.write(f"    Error: {detail['error']}\n")

                    f.write("\n")

            f.write(f"{'-'*80}\n")


def evaluate_all_humaneval_with_logging(output_dir, sample_field='refined_samples', log_dir='logs'):
    """
    Evaluate HumanEval problems with detailed logging.

    Args:
        output_dir: Directory containing JSON files
        sample_field: Which field to evaluate ('baseline_samples', 'refined_samples', or 'samples')
        log_dir: Directory to write log files
    """
    os.makedirs(log_dir, exist_ok=True)

    problems = read_problems()
    results = []

    for task_id, problem in problems.items():
        json_path = os.path.join(output_dir, f"{task_id.replace('/', '_')}.json")
        if not os.path.exists(json_path):
            print(f"MISSING: {json_path}")
            continue

        data = load_json(json_path)

        if sample_field not in data:
            print(f"WARNING: {sample_field} not found in {json_path}")
            continue

        samples = data[sample_field]

        if not samples:
            print(f"WARNING: Empty {sample_field} in {json_path}")
            continue

        scores = evaluate_humaneval_problem_with_logging(problem, samples)
        scores['task_id'] = task_id
        results.append(scores)

    if not results:
        print(f"No results found for {sample_field}")
        return None

    # Write log file
    log_filename = f"humaneval_{sample_field}_failures.log"
    log_path = os.path.join(log_dir, log_filename)
    write_log_file(log_path, results, "HumanEval", sample_field)

    pass1 = np.mean([r["pass@1"] for r in results])
    pass10 = np.mean([r["pass@10"] for r in results]) if all(r['n'] >= 10 for r in results) else None

    print(f"\n{'='*50}")
    print(f"HUMANEVAL - {sample_field}")
    print(f"{'='*50}")
    print(f"Evaluated {len(results)} problems")
    print(f"pass@1:  {pass1:.4f}")
    if pass10 is not None:
        print(f"pass@10: {pass10:.4f}")
    print(f"Log file written to: {log_path}")

    return results


def evaluate_all_mbpp_with_logging(output_dir, sample_field='refined_samples', log_dir='logs'):
    """
    Evaluate MBPP problems with detailed logging.

    Args:
        output_dir: Directory containing JSON files
        sample_field: Which field to evaluate ('baseline_samples', 'refined_samples', or 'samples')
        log_dir: Directory to write log files
    """
    os.makedirs(log_dir, exist_ok=True)

    mbpp = load_dataset("mbpp", "full")["test"]
    results = []

    for ex in mbpp:
        task_id = f"MBPP_{ex['task_id']}.json"
        json_path = os.path.join(output_dir, task_id)
        if not os.path.exists(json_path):
            print(f"MISSING: {json_path}")
            continue

        data = load_json(json_path)

        if sample_field not in data:
            print(f"WARNING: {sample_field} not found in {json_path}")
            continue

        samples = data[sample_field]

        if not samples:
            print(f"WARNING: Empty {sample_field} in {json_path}")
            continue

        problem = {
            "text": ex["text"],
            "code": ex["code"],
            "test_list": ex["test_list"],
        }

        scores = evaluate_mbpp_problem_with_logging(problem, samples)
        scores['task_id'] = ex['task_id']
        results.append(scores)

    if not results:
        print(f"No results found for {sample_field}")
        return None

    # Write log file
    log_filename = f"mbpp_{sample_field}_failures.log"
    log_path = os.path.join(log_dir, log_filename)
    write_log_file(log_path, results, "MBPP", sample_field)

    pass1 = np.mean([r["pass@1"] for r in results])
    pass10 = np.mean([r["pass@10"] for r in results]) if all(r['n'] >= 10 for r in results) else None

    print(f"\n{'='*50}")
    print(f"MBPP - {sample_field}")
    print(f"{'='*50}")
    print(f"Evaluated {len(results)} problems")
    print(f"pass@1:  {pass1:.4f}")
    if pass10 is not None:
        print(f"pass@10: {pass10:.4f}")
    print(f"Log file written to: {log_path}")

    return results


def compare_baseline_vs_refined_with_logging(output_dir, dataset='humaneval', log_dir='logs'):
    """
    Compare baseline and refined samples side-by-side with detailed logging.

    Args:
        output_dir: Directory containing JSON files
        dataset: 'humaneval' or 'mbpp'
        log_dir: Directory to write log files
    """
    print(f"\n{'='*60}")
    print(f"COMPARISON: Baseline vs Refined ({dataset.upper()})")
    print(f"{'='*60}")

    if dataset.lower() == 'humaneval':
        baseline_results = evaluate_all_humaneval_with_logging(output_dir, 'baseline_samples', log_dir)
        refined_results = evaluate_all_humaneval_with_logging(output_dir, 'refined_samples', log_dir)
    elif dataset.lower() == 'mbpp':
        baseline_results = evaluate_all_mbpp_with_logging(output_dir, 'baseline_samples', log_dir)
        refined_results = evaluate_all_mbpp_with_logging(output_dir, 'refined_samples', log_dir)
    else:
        print(f"Unknown dataset: {dataset}")
        return

    if baseline_results and refined_results:
        baseline_pass1 = np.mean([r["pass@1"] for r in baseline_results])
        refined_pass1 = np.mean([r["pass@1"] for r in refined_results])
        improvement = refined_pass1 - baseline_pass1

        print(f"\nSummary:")
        print(f"Baseline pass@1:  {baseline_pass1:.4f}")
        print(f"Refined pass@1:   {refined_pass1:.4f}")
        print(f"Improvement:      {improvement:+.4f} ({improvement/baseline_pass1*100:+.2f}%)")

        # Write comparison summary
        comparison_path = os.path.join(log_dir, f"{dataset}_comparison_summary.txt")
        with open(comparison_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"COMPARISON: Baseline vs Refined ({dataset.upper()})\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Baseline pass@1:  {baseline_pass1:.4f}\n")
            f.write(f"Refined pass@1:   {refined_pass1:.4f}\n")
            f.write(f"Improvement:      {improvement:+.4f} ({improvement/baseline_pass1*100:+.2f}%)\n\n")

            if all(r['n'] >= 10 for r in baseline_results) and all(r['n'] >= 10 for r in refined_results):
                baseline_pass10 = np.mean([r["pass@10"] for r in baseline_results])
                refined_pass10 = np.mean([r["pass@10"] for r in refined_results])
                improvement_pass10 = refined_pass10 - baseline_pass10
                f.write(f"Baseline pass@10:  {baseline_pass10:.4f}\n")
                f.write(f"Refined pass@10:   {refined_pass10:.4f}\n")
                f.write(f"Improvement:       {improvement_pass10:+.4f} ({improvement_pass10/baseline_pass10*100:+.2f}%)\n")

        print(f"Comparison summary written to: {comparison_path}")


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    print("\n" + "="*60)
    print("EVALUATING HUMANEVAL WITH DETAILED LOGGING")
    print("="*60)

    # Compare baseline vs refined for HumanEval
    compare_baseline_vs_refined_with_logging("model_outputs/humaneval/", dataset='humaneval', log_dir='logs')

    print("\n" + "="*60)
    print("EVALUATING MBPP WITH DETAILED LOGGING")
    print("="*60)

    # Compare baseline vs refined for MBPP
    compare_baseline_vs_refined_with_logging("model_outputs/mbpp/", dataset='mbpp', log_dir='logs')

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nLog files created in 'logs/' directory:")
    print("  - humaneval_baseline_samples_failures.log")
    print("  - humaneval_refined_samples_failures.log")
    print("  - humaneval_comparison_summary.txt")
    print("  - mbpp_baseline_samples_failures.log")
    print("  - mbpp_refined_samples_failures.log")
    print("  - mbpp_comparison_summary.txt")
