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
#from execute_verification_chain import ExecuteVerificationChain
#
#
#class WikiDataCategoryListCOVEChain(object):
#    def __init__(self, llm):
#        self.llm = llm
#
#    def __call__(self):
#        # Create baseline response chain
#        baseline_response_prompt_template = PromptTemplate(input_variables=["original_question"],
#                                                           template=prompts.BASELINE_PROMPT_WIKI)
#        baseline_response_chain = LLMChain(llm=self.llm,
#                                           prompt=baseline_response_prompt_template,
#                                           output_key="baseline_response")
#        # Create plan verification chain
#        ## Create plan verification template
#        verification_question_template_prompt_template = PromptTemplate(input_variables=["original_question"],
#                                                                        template=prompts.VERIFICATION_QUESTION_TEMPLATE_PROMPT_WIKI)
#        verification_question_template_chain = LLMChain(llm=self.llm,
#                                                        prompt=verification_question_template_prompt_template,
#                                                        output_key="verification_question_template")
#        ## Create plan verification questions
#        verification_question_generation_prompt_template = PromptTemplate(input_variables=["original_question",
#                                                                                           "baseline_response",
#                                                                                           "verification_question_template"],
#                                                                          template=prompts.VERIFICATION_QUESTION_PROMPT_WIKI)
#        verification_question_generation_chain = LLMChain(llm=self.llm,
#                                                          prompt=verification_question_generation_prompt_template,
#                                                          output_key="verification_questions")
#        # Create execution verification
#        execute_verification_question_prompt_template = PromptTemplate(input_variables=["verification_questions"],
#                                                                       template=prompts.EXECUTE_PLAN_PROMPT)
#        execute_verification_question_chain = ExecuteVerificationChain(llm=self.llm,
#                                                                       prompt=execute_verification_question_prompt_template,
#                                                                       output_key="verification_answers")
#        # Create final refined response
#        final_answer_prompt_template = PromptTemplate(input_variables=["original_question",
#                                                                       "baseline_response",
#                                                                       "verification_answers"],
#                                                      template=prompts.FINAL_REFINED_PROMPT)
#        final_answer_chain = LLMChain(llm=self.llm,
#                                      prompt=final_answer_prompt_template,
#                                      output_key="final_answer")
#
#        # Create sequesntial chain
#        wiki_data_category_list_cove_chain = SequentialChain(
#                                                        chains=[baseline_response_chain,
#                                                                verification_question_template_chain,
#                                                                verification_question_generation_chain,
#                                                                execute_verification_question_chain,
#                                                                final_answer_chain],
#                                                        input_variables=["original_question"],
#                                                        # Here we return multiple variables
#                                                        output_variables=["original_question",
#                                                                          "baseline_response",
#                                                                          "verification_question_template",
#                                                                          "verification_questions",
#                                                                          "verification_answers",
#                                                                          "final_answer"],
#                                                        verbose=False)
#        return wiki_data_category_list_cove_chain
#
#
#class MultiSpanCOVEChain(object):
#    def __init__(self, llm):
#        self.llm = llm
#
#    def __call__(self):
#        # Create baseline response chain
#        baseline_response_prompt_template = PromptTemplate(input_variables=["original_question"],
#                                                           template=prompts.BASELINE_PROMPT_MULTI)
#        baseline_response_chain = LLMChain(llm=self.llm,
#                                           prompt=baseline_response_prompt_template,
#                                           output_key="baseline_response")
#        ## Create plan verification questions
#        verification_question_generation_prompt_template = PromptTemplate(input_variables=["original_question",
#                                                                                           "baseline_response"],
#                                                                          template=prompts.VERIFICATION_QUESTION_PROMPT_MULTI)
#        verification_question_generation_chain = LLMChain(llm=self.llm,
#                                                          prompt=verification_question_generation_prompt_template,
#                                                          output_key="verification_questions")
#        # Create execution verification
#        execute_verification_question_prompt_template = PromptTemplate(input_variables=["verification_questions"],
#                                                                       template=prompts.EXECUTE_PLAN_PROMPT)
#        execute_verification_question_chain = ExecuteVerificationChain(llm=self.llm,
#                                                                       prompt=execute_verification_question_prompt_template,
#                                                                       output_key="verification_answers")
#        # Create final refined response
#        final_answer_prompt_template = PromptTemplate(input_variables=["original_question",
#                                                                       "baseline_response",
#                                                                       "verification_answers"],
#                                                      template=prompts.FINAL_REFINED_PROMPT)
#        final_answer_chain = LLMChain(llm=self.llm,
#                                      prompt=final_answer_prompt_template,
#                                      output_key="final_answer")
#
#        # Create sequesntial chain
#        multi_span_cove_chain = SequentialChain(
#                                                chains=[baseline_response_chain,
#                                                        verification_question_generation_chain,
#                                                        execute_verification_question_chain,
#                                                        final_answer_chain],
#                                                input_variables=["original_question"],
#                                                # Here we return multiple variables
#                                                output_variables=["original_question",
#                                                                  "baseline_response",
#                                                                  "verification_questions",
#                                                                  "verification_answers",
#                                                                  "final_answer"],
#                                                verbose=False)
#        return multi_span_cove_chain
#
#
#class LongFormCOVEChain(object):
#    def __init__(self, llm):
#        self.llm = llm
#
#    def __call__(self):
#        # Create baseline response chain
#        baseline_response_prompt_template = PromptTemplate(input_variables=["original_question"],
#                                                           template=prompts.BASELINE_PROMPT_LONG)
#        baseline_response_chain = LLMChain(llm=self.llm,
#                                           prompt=baseline_response_prompt_template,
#                                           output_key="baseline_response")
#        ## Create plan verification questions
#        verification_question_generation_prompt_template = PromptTemplate(input_variables=["original_question",
#                                                                                           "baseline_response"],
#                                                                          template=prompts.VERIFICATION_QUESTION_PROMPT_LONG)
#        verification_question_generation_chain = LLMChain(llm=self.llm,
#                                                          prompt=verification_question_generation_prompt_template,
#                                                          output_key="verification_questions")
#        # Create execution verification
#        execute_verification_question_prompt_template = PromptTemplate(input_variables=["verification_questions"],
#                                                                       template=prompts.EXECUTE_PLAN_PROMPT)
#        execute_verification_question_chain = ExecuteVerificationChain(llm=self.llm,
#                                                                       prompt=execute_verification_question_prompt_template,
#                                                                       output_key="verification_answers")
#        # Create final refined response
#        final_answer_prompt_template = PromptTemplate(input_variables=["original_question",
#                                                                       "baseline_response",
#                                                                       "verification_answers"],
#                                                      template=prompts.FINAL_REFINED_PROMPT)
#        final_answer_chain = LLMChain(llm=self.llm,
#                                      prompt=final_answer_prompt_template,
#                                      output_key="final_answer")
#
#        # Create sequesntial chain
#        long_form_cove_chain = SequentialChain(
#                                                chains=[baseline_response_chain,
#                                                        verification_question_generation_chain,
#                                                        execute_verification_question_chain,
#                                                        final_answer_chain],
#                                                input_variables=["original_question"],
#                                                # Here we return multiple variables
#                                                output_variables=["original_question",
#                                                                  "baseline_response",
#                                                                  "verification_questions",
#                                                                  "verification_answers",
#                                                                  "final_answer"],
#                                                verbose=False)
#        return long_form_cove_chain


"""
Code generation chains for each category
"""

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

import prompts
from execute_verification_chain import SandboxExecutor
from code_utils import extract_code_from_response


class CodeGenerationChain:
    """Base class for code generation with verification"""

    def __init__(self, generation_llm, verification_llm, category: str):
        self.generation_llm = generation_llm  # For baseline code generation (7B)
        self.verification_llm = verification_llm  # For verification and refinement (13B)
        self.category = category
        self.executor = SandboxExecutor()

    def create_chain(self):
        """Create the full chain for this category"""

        # Step 1: Baseline code generation (using generation_llm - 7B)
        baseline_prompt = PromptTemplate(
            input_variables=["original_question"],
            template=prompts.BASELINE_BY_CATEGORY[self.category]
        )
        baseline_chain = LLMChain(
            llm=self.generation_llm,
            prompt=baseline_prompt,
            output_key="baseline_response"
        )

        # Step 2: Verification test generation (using verification_llm - 13B)
        verification_prompt = PromptTemplate(
            input_variables=["original_question", "baseline_response"],
            template=prompts.VERIFY_PLAN_BY_CATEGORY[self.category]
        )
        verification_chain = LLMChain(
            llm=self.verification_llm,
            prompt=verification_prompt,
            output_key="verification_payload"
        )

        # Step 3: Execute and refine (handled in __call__)
        # We can't use SequentialChain here because execution is imperative

        return baseline_chain, verification_chain

    def __call__(self, inputs: dict) -> dict:
        """Execute the full pipeline"""
        baseline_chain, verification_chain = self.create_chain()

        # Generate baseline code
        baseline_result = baseline_chain(inputs)
        baseline_code_raw = baseline_result["baseline_response"]
        baseline_code = extract_code_from_response(baseline_code_raw)

        # Generate verification tests
        verification_inputs = {
            "original_question": inputs["original_question"],
            "baseline_response": baseline_code
        }
        verification_result = verification_chain(verification_inputs)
        verification_payload = verification_result["verification_payload"]

        # Execute tests
        exec_result = self.executor.execute(
            self.category,
            baseline_code,
            verification_payload
        )

        # If successful, return baseline
        if exec_result["success"]:
            return {
                "original_question": inputs["original_question"],
                "baseline_response": baseline_code,
                "verification_payload": verification_payload,
                "execution_result": exec_result,
                "final_answer": baseline_code,
                "refinement_attempted": False
            }

        # Otherwise, refine (using verification_llm - 13B)
        refinement_prompt = self._create_refinement_prompt(
            inputs["original_question"],
            baseline_code,
            exec_result
        )

        # Call verification LLM with string prompt
        refinement_response = self.verification_llm(refinement_prompt)
        refinement_text = refinement_response.content if hasattr(refinement_response, 'content') else str(refinement_response)
        refined_code = extract_code_from_response(refinement_text)

        return {
            "original_question": inputs["original_question"],
            "baseline_response": baseline_code,
            "verification_payload": verification_payload,
            "execution_result": exec_result,
            "final_answer": refined_code,
            "refinement_attempted": True
        }

    def _create_refinement_prompt(self, question: str, baseline_code: str, exec_result: dict) -> str:
        """Create refinement prompt based on category"""
        template = prompts.FINAL_REWRITE_BY_CATEGORY[self.category]

        # Build context based on category
        if self.category == "algorithms":
            return template.format(
                question=question,
                baseline_code=baseline_code,
                failures="\n".join(exec_result["failures"])
            )
        elif self.category == "debugging":
            return template.format(
                question=question,
                baseline_code=baseline_code,
                stderr=exec_result["stderr"],
                failed_inputs=exec_result.get("failed_inputs", [])
            )
        elif self.category == "api_usage":
            return template.format(
                question=question,
                baseline_code=baseline_code,
                stderr=exec_result["stderr"]
            )
        elif self.category == "data_processing":
            return template.format(
                question=question,
                baseline_code=baseline_code,
                expected=exec_result.get("expected", "N/A"),
                actual=exec_result.get("actual", "N/A")
            )
        else:
            return template.format(
                question=question,
                baseline_code=baseline_code,
                failures="\n".join(exec_result["failures"])
            )


class AlgorithmsChain(CodeGenerationChain):
    def __init__(self, generation_llm, verification_llm):
        super().__init__(generation_llm, verification_llm, "algorithms")


class DebuggingChain(CodeGenerationChain):
    def __init__(self, generation_llm, verification_llm):
        super().__init__(generation_llm, verification_llm, "debugging")


class APIUsageChain(CodeGenerationChain):
    def __init__(self, generation_llm, verification_llm):
        super().__init__(generation_llm, verification_llm, "api_usage")


class DataProcessingChain(CodeGenerationChain):
    def __init__(self, generation_llm, verification_llm):
        super().__init__(generation_llm, verification_llm, "data_processing")
