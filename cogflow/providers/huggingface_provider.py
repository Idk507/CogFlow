"""
CogFlow HuggingFace Provider

Implementation of BaseLLMProvider for HuggingFace models.
"""

import os
from typing import Any, Dict, List, Optional, AsyncIterator

from ..base.llm import (
    BaseLLMProvider,
    LLMResponse,
    LLMProviderError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)
from ..base.types import Message


class HuggingFaceProvider(BaseLLMProvider):
    """
    HuggingFace LLM provider.
    
    Supports HuggingFace Inference API and local transformers models.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        api_key: Optional[str] = None,
        use_local: bool = False,
        device: str = "auto",
        **kwargs: Any
    ):
        """
        Initialize HuggingFace provider.
        
        Args:
            model_name: HuggingFace model ID
            api_key: HuggingFace API token (or set HUGGINGFACE_API_KEY env var)
            use_local: Whether to use local transformers instead of API
            device: Device for local inference ('cpu', 'cuda', 'auto')
            **kwargs: Additional options
        """
        api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        super().__init__(model_name, api_key, **kwargs)
        
        self.use_local = use_local
        self.device = device
        self._client = None
        self._pipeline = None
        self._tokenizer = None
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    def _get_inference_client(self):
        """Get HuggingFace Inference Client."""
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
            except ImportError:
                raise ImportError(
                    "huggingface_hub package not installed. "
                    "Install with: pip install huggingface-hub"
                )
            
            self._client = InferenceClient(
                model=self.model_name,
                token=self.api_key,
            )
        return self._client
    
    def _get_pipeline(self):
        """Get local transformers pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline, AutoTokenizer
                import torch
            except ImportError:
                raise ImportError(
                    "transformers package not installed. "
                    "Install with: pip install transformers torch"
                )
            
            device_map = self.device
            if self.device == "auto":
                device_map = "auto" if torch.cuda.is_available() else "cpu"
            
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map=device_map,
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        return self._pipeline
    
    def _handle_error(self, e: Exception) -> None:
        """Convert HuggingFace errors to CogFlow errors."""
        error_msg = str(e)
        
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            raise RateLimitError(error_msg, self.provider_name)
        elif "unauthorized" in error_msg.lower() or "401" in error_msg:
            raise AuthenticationError(error_msg, self.provider_name)
        elif "not found" in error_msg.lower() or "404" in error_msg:
            raise ModelNotFoundError(error_msg, self.provider_name)
        else:
            raise LLMProviderError(error_msg, self.provider_name)
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a response from a prompt."""
        try:
            if self.use_local:
                return await self._generate_local(
                    prompt, temperature, max_tokens, stop, **kwargs
                )
            else:
                return await self._generate_api(
                    prompt, temperature, max_tokens, stop, **kwargs
                )
        except Exception as e:
            self._handle_error(e)
            raise
    
    async def _generate_api(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        **kwargs: Any
    ) -> LLMResponse:
        """Generate using HuggingFace Inference API."""
        client = self._get_inference_client()
        
        # Use text_generation for completion
        response = client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.01,
            stop_sequences=stop,
            return_full_text=False,
            **kwargs
        )
        
        return LLMResponse(
            content=response,
            model=self.model_name,
            usage={
                "prompt_tokens": len(prompt.split()) * 2,  # Rough estimate
                "completion_tokens": len(response.split()) * 2,
            },
            finish_reason="stop",
            metadata={"provider": "huggingface_inference_api"}
        )
    
    async def _generate_local(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        **kwargs: Any
    ) -> LLMResponse:
        """Generate using local transformers pipeline."""
        import asyncio
        
        pipe = self._get_pipeline()
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                do_sample=temperature > 0,
                return_full_text=False,
                **kwargs
            )
        )
        
        generated_text = result[0]["generated_text"]
        
        # Handle stop sequences
        if stop:
            for seq in stop:
                if seq in generated_text:
                    generated_text = generated_text.split(seq)[0]
        
        return LLMResponse(
            content=generated_text,
            model=self.model_name,
            usage={
                "prompt_tokens": len(self._tokenizer.encode(prompt)) if self._tokenizer else 0,
                "completion_tokens": len(self._tokenizer.encode(generated_text)) if self._tokenizer else 0,
            },
            finish_reason="stop",
            metadata={"provider": "huggingface_local"}
        )
    
    async def generate_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a response from a chat conversation."""
        # Convert messages to a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.role.value.upper()
            prompt_parts.append(f"{role}: {msg.content}")
        prompt_parts.append("ASSISTANT:")
        
        prompt = "\n".join(prompt_parts)
        
        return await self.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or ["USER:", "\nUSER:"],
            **kwargs
        )
    
    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream generated tokens."""
        if self.use_local:
            # Local streaming not implemented - yield full response
            response = await self.generate(
                prompt, temperature, max_tokens, **kwargs
            )
            yield response.content
        else:
            client = self._get_inference_client()
            
            for token in client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                stream=True,
                **kwargs
            ):
                yield token
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        
        # Try to load tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            return len(tokenizer.encode(text))
        except Exception:
            # Fallback: rough estimate
            return len(text) // 4
