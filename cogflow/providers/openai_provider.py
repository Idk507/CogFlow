"""
CogFlow OpenAI Provider

Implementation of BaseLLMProvider for OpenAI models.
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


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider.
    
    Supports GPT-3.5, GPT-4, and other OpenAI models.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model_name: Model to use (gpt-4, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            organization: Optional organization ID
            base_url: Optional custom base URL
            **kwargs: Additional options
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(model_name, api_key, **kwargs)
        
        self.organization = organization
        self.base_url = base_url
        self._client = None
        self._async_client = None
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def _get_client(self):
        """Get or create synchronous OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
            
            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )
        return self._client
    
    def _get_async_client(self):
        """Get or create async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
            
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )
        return self._async_client
    
    def _handle_error(self, e: Exception) -> None:
        """Convert OpenAI errors to CogFlow errors."""
        error_msg = str(e)
        
        if "rate_limit" in error_msg.lower():
            raise RateLimitError(error_msg, self.provider_name)
        elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            raise AuthenticationError(error_msg, self.provider_name)
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
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
        messages = [{"role": "user", "content": prompt}]
        
        try:
            client = self._get_async_client()
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs
            )
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason or "stop",
                metadata={"id": response.id}
            )
        except Exception as e:
            self._handle_error(e)
            raise  # This line won't be reached but keeps type checker happy
    
    async def generate_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a response from a chat conversation."""
        api_messages = [msg.to_dict() for msg in messages]
        
        try:
            client = self._get_async_client()
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs
            )
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason or "stop",
                metadata={"id": response.id}
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
        messages = [{"role": "user", "content": prompt}]
        
        try:
            client = self._get_async_client()
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self._handle_error(e)
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
        except ImportError:
            # Rough estimate if tiktoken not available
            return len(text) // 4
        
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    def generate_sync(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Synchronous generation (for non-async contexts)."""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs
            )
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason or "stop",
                metadata={"id": response.id}
            )
        except Exception as e:
            self._handle_error(e)
            raise
