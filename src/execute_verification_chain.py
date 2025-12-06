## from __future__ import annotations
#
#import os
#import re
#import itertools
#import openai
#import tiktoken
#import json
#from dotenv import load_dotenv
#
#from typing import Any, Dict, List, Optional
#
#from pydantic import Extra
#
#from langchain.schema.language_model import BaseLanguageModel
#from langchain.callbacks.manager import (
#    AsyncCallbackManagerForChainRun,
#    CallbackManagerForChainRun,
#)
#from langchain.schema import (
#    AIMessage,
#    HumanMessage,
#    SystemMessage
#)
#from langchain.chains.base import Chain
#from langchain.prompts.base import BasePromptTemplate
#from langchain.tools import DuckDuckGoSearchRun
#import langchain
#from langchain.chat_models import ChatOpenAI
#from langchain.tools import DuckDuckGoSearchRun
#from langchain.schema import (
#    AIMessage,
#    HumanMessage,
#    SystemMessage
#)
#from langchain.chains.llm import LLMChain
#from langchain.prompts import PromptTemplate
#from langchain.chains import SequentialChain
#
#import prompts
#
#
#
#class ExecuteVerificationChain(Chain):
#    """
#    Implements the logic to execute the verification question for factual acuracy
#    """
#
#    prompt: BasePromptTemplate
#    llm: BaseLanguageModel
#    input_key: str = "verification_questions"
#    output_key: str = "verification_answers"
#    use_search_tool: bool = True
#    search_tool: Any = DuckDuckGoSearchRun()
#
#    class Config:
#        """Configuration for this pydantic object."""
#
#        extra = Extra.forbid
#        arbitrary_types_allowed = True
#
#    @property
#    def input_keys(self) -> List[str]:
#        """Will be whatever keys the prompt expects.
#
#        :meta private:
#        """
#        return [self.input_key]
#
#    @property
#    def output_keys(self) -> List[str]:
#        """Will always return text key.
#
#        :meta private:
#        """
#        return [self.output_key]
#
#    def search_for_verification_question(self,
#                                         verification_question: str
#                                        ) -> str:
#        search_result = self.search_tool.run(verification_question)
#        return search_result
#
#    def _call(
#        self,
#        inputs: Dict[str, Any],
#        run_manager: Optional[CallbackManagerForChainRun] = None,
#    ) -> Dict[str, str]:
#        verification_answers_list = list() # Will contain the answers of each verification questions
#        question_answer_pair = "" # Final output of verification question and answer pair
#
#        # Convert all the verification questions into a list of string
#        sub_inputs = {k:v for k,v in inputs.items() if k==self.input_key}
#        verification_questions_prompt_value = self.prompt.format_prompt(**sub_inputs)
#        verification_questions_str = verification_questions_prompt_value.text
#        verification_questions_list = verification_questions_str.split("\n")
#
#        # Setting up prompt for both search tool and llm self evaluation
#        execution_prompt_search_tool = PromptTemplate.from_template(prompts.EXECUTE_PLAN_PROMPT_SEARCH_TOOL)
#        execution_prompt_self_llm = PromptTemplate.from_template(prompts.EXECUTE_PLAN_PROMPT_SELF_LLM)
#
#        # Executing the verification questions, either using search tool or self llm
#        for question in verification_questions_list:
#            if self.use_search_tool:
#                search_result = self.search_for_verification_question(question)
#                execution_prompt_value = execution_prompt_search_tool.format_prompt(**{"search_result": search_result, "verification_question": question})
#            else:
#                execution_prompt_value = execution_prompt_self_llm.format_prompt(**{"verification_question": question})
#            verification_answer_llm_result = self.llm.generate_prompt([execution_prompt_value], callbacks=run_manager.get_child() if run_manager else None)
#            verification_answer_str = verification_answer_llm_result.generations[0][0].text
#            verification_answers_list.append(verification_answer_str)
#
#        # Create verification question and answer pair
#        for question, answer in itertools.zip_longest(verification_questions_list, verification_answers_list):
#            question_answer_pair += "Question: {} Answer: {}\n".format(question, answer)
#
#        if run_manager:
#            run_manager.on_text("Log something about this run")
#
#        return {self.output_key: question_answer_pair}
#
#    async def _acall(
#        self,
#        inputs: Dict[str, Any],
#        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
#    ) -> Dict[str, str]:
#        # Your custom chain logic goes here
#        # This is just an example that mimics LLMChain
#        prompt_value = self.prompt.format_prompt(**inputs)
#
#        # Whenever you call a language model, or another chain, you should pass
#        # a callback manager to it. This allows the inner run to be tracked by
#        # any callbacks that are registered on the outer run.
#        # You can always obtain a callback manager for this by calling
#        # `run_manager.get_child()` as shown below.
#        response = await self.llm.agenerate_prompt(
#            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
#        )
#
#        # If you want to log something about this run, you can do so by calling
#        # methods on the `run_manager`, as shown below. This will trigger any
#        # callbacks that are registered for that event.
#        if run_manager:
#            await run_manager.on_text("Log something about this run")
#
#        return {self.output_key: response.generations[0][0].text}
#
#    @property
#    def _chain_type(self) -> str:
#        return "execute_verification_chain"


"""
Sandbox executor for running generated code with category-specific strategies
"""

import sys
import io
import traceback
from typing import Any, Dict, List, Optional
from contextlib import redirect_stdout, redirect_stderr

from code_utils import extract_function_name, safe_eval_literal


class SandboxExecutor:
    """Executes code in isolated namespace with category-specific logic"""

    def __init__(self):
        self.timeout_seconds = 5  # Safety timeout

    def execute(self,
                category: str,
                baseline_code: str,
                verification_payload: str) -> Dict[str, Any]:
        """
        Main dispatcher for execution

        Returns:
            {
                "success": bool,
                "stdout": str,
                "stderr": str,
                "failures": List[str],
                "failed_inputs": List[Any],
                "expected": Any,
                "actual": Any
            }
        """
        if category == "algorithms":
            return self._run_algorithm_tests(baseline_code, verification_payload)
        elif category == "debugging":
            return self._run_debugging_tests(baseline_code, verification_payload)
        elif category == "api_usage":
            return self._run_api_test(baseline_code, verification_payload)
        elif category == "data_processing":
            return self._run_data_processing_test(baseline_code, verification_payload)
        else:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Unknown category: {category}",
                "failures": [f"Unknown category: {category}"]
            }

    def _run_algorithm_tests(self, func_code: str, tests_str: str) -> Dict[str, Any]:
        """Run algorithm with test cases [(input, expected), ...]"""
        result = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "failures": [],
            "failed_inputs": []
        }

        try:
            # Parse test cases
            test_cases = safe_eval_literal(tests_str)

            # Execute function code
            namespace = {}
            exec(func_code, namespace)

            # Find the function
            func_name = extract_function_name(func_code)
            if not func_name or func_name not in namespace:
                result["success"] = False
                result["stderr"] = "Could not find function in code"
                result["failures"].append("No function found")
                return result

            func = namespace[func_name]

            # Run each test case
            for i, (test_input, expected) in enumerate(test_cases):
                try:
                    # Handle both single args and multiple args
                    if isinstance(test_input, tuple):
                        actual = func(*test_input)
                    else:
                        actual = func(test_input)

                    if actual != expected:
                        result["success"] = False
                        failure_msg = f"Test {i+1} failed: input={test_input}, expected={expected}, got={actual}"
                        result["failures"].append(failure_msg)
                        result["failed_inputs"].append(test_input)

                except Exception as e:
                    result["success"] = False
                    failure_msg = f"Test {i+1} raised exception: {type(e).__name__}: {str(e)}"
                    result["failures"].append(failure_msg)
                    result["failed_inputs"].append(test_input)
                    result["stderr"] += f"{failure_msg}\n"

        except Exception as e:
            result["success"] = False
            result["stderr"] = f"Execution error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result["failures"].append(str(e))

        return result

    def _run_debugging_tests(self, func_code: str, inputs_str: str) -> Dict[str, Any]:
        """Run debugging tests - just check if code runs without errors"""
        result = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "failures": [],
            "failed_inputs": []
        }

        try:
            # Parse test inputs
            test_inputs = safe_eval_literal(inputs_str)

            # Execute function code
            namespace = {}
            exec(func_code, namespace)

            # Find the function
            func_name = extract_function_name(func_code)
            if not func_name or func_name not in namespace:
                result["success"] = False
                result["stderr"] = "Could not find function in code"
                result["failures"].append("No function found")
                return result

            func = namespace[func_name]

            # Run each test input
            for i, test_input in enumerate(test_inputs):
                try:
                    # Capture output
                    stdout_capture = io.StringIO()
                    with redirect_stdout(stdout_capture):
                        if isinstance(test_input, tuple):
                            _ = func(*test_input)
                        else:
                            _ = func(test_input)

                    result["stdout"] += stdout_capture.getvalue()

                except Exception as e:
                    result["success"] = False
                    error_msg = f"Input {i+1} ({test_input}) raised: {type(e).__name__}: {str(e)}"
                    result["failures"].append(error_msg)
                    result["failed_inputs"].append(test_input)
                    result["stderr"] += f"{error_msg}\n{traceback.format_exc()}\n"

        except Exception as e:
            result["success"] = False
            result["stderr"] = f"Execution error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result["failures"].append(str(e))

        return result

    def _run_api_test(self, func_code: str, harness_code: str) -> Dict[str, Any]:
        """Run API usage test with harness code"""
        result = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "failures": []
        }

        try:
            # Combine function and harness
            full_code = func_code + "\n\n" + harness_code

            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                namespace = {}
                exec(full_code, namespace)

            result["stdout"] = stdout_capture.getvalue()
            result["stderr"] = stderr_capture.getvalue()

            if result["stderr"]:
                result["success"] = False
                result["failures"].append("Code produced errors")

        except Exception as e:
            result["success"] = False
            result["stderr"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result["failures"].append(str(e))

        return result

    def _run_data_processing_test(self, func_code: str, test_data_code: str) -> Dict[str, Any]:
        """Run data processing test with TEST_INPUT and EXPECTED_OUTPUT"""
        result = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "failures": [],
            "expected": None,
            "actual": None
        }

        try:
            # Execute test data code to get TEST_INPUT and EXPECTED_OUTPUT
            test_namespace = {}
            exec(test_data_code, test_namespace)

            test_input = test_namespace.get("TEST_INPUT")
            expected_output = test_namespace.get("EXPECTED_OUTPUT")

            if test_input is None or expected_output is None:
                result["success"] = False
                result["stderr"] = "Test data must define TEST_INPUT and EXPECTED_OUTPUT"
                result["failures"].append("Missing test data variables")
                return result

            # Execute function code
            func_namespace = {}
            exec(func_code, func_namespace)

            # Find the function
            func_name = extract_function_name(func_code)
            if not func_name or func_name not in func_namespace:
                result["success"] = False
                result["stderr"] = "Could not find function in code"
                result["failures"].append("No function found")
                return result

            func = func_namespace[func_name]

            # Run the function
            actual_output = func(test_input)

            result["expected"] = expected_output
            result["actual"] = actual_output

            # Compare outputs
            if actual_output != expected_output:
                result["success"] = False
                result["failures"].append(f"Output mismatch: expected {expected_output}, got {actual_output}")

        except Exception as e:
            result["success"] = False
            result["stderr"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result["failures"].append(str(e))

        return result
