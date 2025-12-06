"""
Code Llama wrapper for LangChain compatibility
"""

from typing import List, Optional, Any, Dict
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class CodeLlamaLLM(BaseLanguageModel):
    """Code Llama wrapper compatible with LangChain"""

    model_name: str = Field(default="codellama/CodeLlama-7b-Instruct-hf")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1000)
    device: str = Field(default="auto")
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Lazy loading - only load when first used
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            print(f"Loading Code Llama model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if self.device != "auto" else None
            )
            if self.device == "auto":
                if torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"
            else:
                self._device = self.device

            # Move model to device if not using device_map
            if self.device != "auto" and hasattr(self._model, 'to'):
                self._model = self._model.to(self._device)

            self._model.eval()
            print(f"Model loaded on device: {self._device}")

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for Code Llama Instruct"""
        # Code Llama Instruct format
        return f"[INST] {prompt} [/INST]"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate text from prompts"""
        self._load_model()

        results = []
        for prompt in prompts:
            # Format prompt for Code Llama
            formatted_prompt = self._format_prompt(prompt)

            # Tokenize
            inputs = self._tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self._device)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            # Decode
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Handle stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]

            results.append(generated_text.strip())

        # Return in LangChain format
        from langchain.schema import LLMResult, Generation
        generations = [[Generation(text=text)] for text in results]
        return LLMResult(generations=generations)

    def _llm_type(self) -> str:
        return "code_llama"

    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """Generate from prompts - required by LangChain"""
        # Convert prompt values to strings if needed
        prompt_strings = []
        for prompt in prompts:
            if hasattr(prompt, 'to_string'):
                prompt_strings.append(prompt.to_string())
            elif hasattr(prompt, 'text'):
                prompt_strings.append(prompt.text)
            elif isinstance(prompt, str):
                prompt_strings.append(prompt)
            else:
                prompt_strings.append(str(prompt))
        return self._generate(prompt_strings, stop=stop, run_manager=callbacks, **kwargs)

    def predict(self, text: str, stop=None, **kwargs) -> str:
        """Predict from text - required by LangChain"""
        result = self._generate([text], stop=stop, **kwargs)
        return result.generations[0][0].text

    def predict_messages(self, messages, stop=None, **kwargs):
        """Predict from messages - required by LangChain"""
        # Convert messages to prompt string
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(msg.content)
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n".join(prompt_parts)
        result = self._generate([prompt], stop=stop, **kwargs)
        generated_text = result.generations[0][0].text
        return AIMessage(content=generated_text)

    def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """Async generate - not implemented, raise error"""
        raise NotImplementedError("Async generation not supported")

    def apredict(self, text: str, stop=None, **kwargs):
        """Async predict - not implemented, raise error"""
        raise NotImplementedError("Async prediction not supported")

    def apredict_messages(self, messages, stop=None, **kwargs):
        """Async predict messages - not implemented, raise error"""
        raise NotImplementedError("Async prediction not supported")

    def __call__(self, prompt_or_messages, **kwargs):
        """Call the LLM with either a string prompt or list of messages"""
        # Handle string prompt (for refinement)
        if isinstance(prompt_or_messages, str):
            result = self._generate([prompt_or_messages], **kwargs)
            generated_text = result.generations[0][0].text
            # Return AIMessage for consistency
            return AIMessage(content=generated_text)

        # Handle list of messages (for routing)
        elif isinstance(prompt_or_messages, list):
            # Convert messages to prompt string
            prompt_parts = []
            for msg in prompt_or_messages:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"System: {msg.content}")
                elif isinstance(msg, HumanMessage):
                    prompt_parts.append(msg.content)
                elif isinstance(msg, AIMessage):
                    prompt_parts.append(f"Assistant: {msg.content}")

            prompt = "\n".join(prompt_parts)

            # Generate
            result = self._generate([prompt], **kwargs)
            generated_text = result.generations[0][0].text

            return AIMessage(content=generated_text)
        else:
            raise ValueError(f"Expected str or list of messages, got {type(prompt_or_messages)}")

    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke with a string prompt (for refinement)"""
        result = self._generate([prompt], **kwargs)
        return result.generations[0][0].text
