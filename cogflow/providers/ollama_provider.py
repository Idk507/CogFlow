"""
CogFlow Ollama Provider

Implementation of BaseLLMProvider for local Ollama models.
"""

import os
from typing import Any, Dict, List, Optional, AsyncIterator

from ..base.llm import (
    BaseLLMProvider,
    LLMResponse,
    LLMProviderError,
    ModelNotFoundError,
)
from ..base.types import Message


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider.
    
    Supports local Ollama models (llama2, mistral, codellama, etc.).
    """
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs: Any
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model_name: Ollama model name (llama2, mistral, etc.)
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            **kwargs: Additional options
        """
        super().__init__(model_name, api_key=None, **kwargs)
        
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout
        self._client = None
        self._async_client = None
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    def _get_client(self):
        """Get or create synchronous Ollama client."""
        if self._client is None:
            try:
                from ollama import Client
            except ImportError:
                raise ImportError(
                    "ollama package not installed. "
                    "Install with: pip install ollama"
                )
            
            self._client = Client(host=self.base_url)
        return self._client
    
    def _get_async_client(self):
        """Get or create async Ollama client."""
        if self._async_client is None:
            try:
                from ollama import AsyncClient
            except ImportError:
                raise ImportError(
                    "ollama package not installed. "
                    "Install with: pip install ollama"
                )
            
            self._async_client = AsyncClient(host=self.base_url)
        return self._async_client
    
    def _handle_error(self, e: Exception) -> None:
        """Convert Ollama errors to CogFlow errors."""
        error_msg = str(e)
        
        if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "does not exist" in error_msg.lower()):
            raise ModelNotFoundError(
                f"Model '{self.model_name}' not found. "
                f"Pull it with: ollama pull {self.model_name}",
                self.provider_name
            )
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
            client = self._get_async_client()
            
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            if stop:
                options["stop"] = stop
            
            response = await client.generate(
                model=self.model_name,
                prompt=prompt,
                options=options,
                **kwargs
            )
            
            return LLMResponse(
                content=response.get("response", ""),
                model=self.model_name,
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": (
                        response.get("prompt_eval_count", 0) +
                        response.get("eval_count", 0)
                    ),
                },
                finish_reason=response.get("done_reason", "stop"),
                metadata={
                    "total_duration": response.get("total_duration"),
                    "load_duration": response.get("load_duration"),
                    "eval_duration": response.get("eval_duration"),
                }
            )
        except Exception as e:
            self._handle_error(e)
            raise
    
    async def generate_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a response from a chat conversation."""
        try:
            client = self._get_async_client()
            
            # Convert messages to Ollama format
            api_messages = []
            for msg in messages:
                api_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            if stop:
                options["stop"] = stop
            
            response = await client.chat(
                model=self.model_name,
                messages=api_messages,
                options=options,
                **kwargs
            )
            
            message = response.get("message", {})
            
            return LLMResponse(
                content=message.get("content", ""),
                model=self.model_name,
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": (
                        response.get("prompt_eval_count", 0) +
                        response.get("eval_count", 0)
                    ),
                },
                finish_reason=response.get("done_reason", "stop"),
                metadata={
                    "total_duration": response.get("total_duration"),
                    "load_duration": response.get("load_duration"),
                }
            )
        except Exception as e:
            self._handle_error(e)
            raise
    
    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream generated tokens."""
        try:
            client = self._get_async_client()
            
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            
            async for chunk in await client.generate(
                model=self.model_name,
                prompt=prompt,
                options=options,
                stream=True,
                **kwargs
            ):
                if chunk.get("response"):
                    yield chunk["response"]
        except Exception as e:
            self._handle_error(e)
    
    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count.
        
        Note: Ollama doesn't provide a tokenizer API,
        so this is an approximation.
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def generate_sync(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Synchronous generation (for non-async contexts)."""
        try:
            client = self._get_client()
            
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            if stop:
                options["stop"] = stop
            
            response = client.generate(
                model=self.model_name,
                prompt=prompt,
                options=options,
                **kwargs
            )
            
            return LLMResponse(
                content=response.get("response", ""),
                model=self.model_name,
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                },
                finish_reason=response.get("done_reason", "stop"),
                metadata={
                    "total_duration": response.get("total_duration"),
                }
            )
        except Exception as e:
            self._handle_error(e)
            raise
    
    async def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            client = self._get_async_client()
            response = await client.list()
            return [model["name"] for model in response.get("models", [])]
        except Exception as e:
            self._handle_error(e)
            raise
    
    async def pull_model(self, model_name: Optional[str] = None) -> None:
        """Pull a model from the Ollama library."""
        try:
            client = self._get_async_client()
            await client.pull(model_name or self.model_name)
        except Exception as e:
            self._handle_error(e)
            raise
