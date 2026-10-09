"""
CogFlow Anthropic Provider

Implementation of BaseLLMProvider for Anthropic Claude models.
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
from ..base.types import Message, MessageRole


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM provider.
    
    Supports Claude 3 (Opus, Sonnet, Haiku) and earlier models.
    """
    
    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 2,
        **kwargs: Any
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            model_name: Model to use (claude-3-opus, claude-3-sonnet, etc.)
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            base_url: Optional custom base URL
            max_retries: Number of retries on failure
            **kwargs: Additional options
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(model_name, api_key, **kwargs)
        
        self.base_url = base_url
        self.max_retries = max_retries
        self._client = None
        self._async_client = None
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def _get_client(self):
        """Get or create synchronous Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
            
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=self.max_retries,
            )
        return self._client
    
    def _get_async_client(self):
        """Get or create async Anthropic client."""
        if self._async_client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
            
            self._async_client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=self.max_retries,
            )
        return self._async_client
    
    def _handle_error(self, e: Exception) -> None:
        """Convert Anthropic errors to CogFlow errors."""
        error_msg = str(e)
        
        if "rate_limit" in error_msg.lower():
            raise RateLimitError(error_msg, self.provider_name)
        elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            raise AuthenticationError(error_msg, self.provider_name)
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            raise ModelNotFoundError(error_msg, self.provider_name)
        else:
            raise LLMProviderError(error_msg, self.provider_name)
    
    def _convert_messages(self, messages: List[Message]) -> tuple[str, List[Dict]]:
        """Convert CogFlow messages to Anthropic format."""
        system_prompt = ""
        api_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "assistant"
                api_messages.append({
                    "role": role,
                    "content": msg.content
                })
        
        return system_prompt, api_messages
    
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
            
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                stop_sequences=stop or [],
                **kwargs
            )
            
            content = ""
            if response.content:
                content = response.content[0].text
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                finish_reason=response.stop_reason or "end_turn",
                metadata={"id": response.id}
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
        system_prompt, api_messages = self._convert_messages(messages)
        
        try:
            client = self._get_async_client()
            
            create_kwargs = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": api_messages,
                "stop_sequences": stop or [],
                **kwargs
            }
            
            if system_prompt:
                create_kwargs["system"] = system_prompt
            
            response = await client.messages.create(**create_kwargs)
            
            content = ""
            if response.content:
                content = response.content[0].text
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                finish_reason=response.stop_reason or "end_turn",
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
        try:
            client = self._get_async_client()
            
            async with client.messages.stream(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            self._handle_error(e)
    
    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Anthropic models.
        
        Note: Anthropic doesn't provide a public tokenizer,
        so this is an approximation.
        """
        # Rough estimate: ~4 characters per token for English
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
            
            response = client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                stop_sequences=stop or [],
                **kwargs
            )
            
            content = ""
            if response.content:
                content = response.content[0].text
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                finish_reason=response.stop_reason or "end_turn",
                metadata={"id": response.id}
            )
        except Exception as e:
            self._handle_error(e)
            raise
