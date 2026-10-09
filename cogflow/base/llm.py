"""
CogFlow LLM Provider Interface

This module defines the abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator, Union
from pydantic import BaseModel, Field

from .types import Message


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    
    content: str = Field(description="Generated text content")
    model: str = Field(description="Model that generated the response")
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics"
    )
    finish_reason: str = Field(
        default="stop",
        description="Reason for completion"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @property
    def prompt_tokens(self) -> int:
        """Get prompt token count."""
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        """Get completion token count."""
        return self.usage.get("completion_tokens", 0)
    
    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.usage.get("total_tokens", 
                              self.prompt_tokens + self.completion_tokens)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers (OpenAI, Anthropic, HuggingFace, Ollama) must
    implement this interface.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the LLM provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            **kwargs: Additional provider-specific options
        """
        self.model_name = model_name
        self.api_key = api_key
        self.options = kwargs
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generate a response from a single prompt.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional generation options
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    async def generate_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generate a response from a chat conversation.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional generation options
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    async def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts.
        
        Default implementation calls generate() sequentially.
        Providers can override for batch optimization.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            **kwargs: Additional options
            
        Returns:
            List of LLMResponse objects
        """
        responses = []
        for prompt in prompts:
            response = await self.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            responses.append(response)
        return responses
    
    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Stream generated tokens.
        
        Default implementation yields the full response.
        Providers can override for true streaming.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional options
            
        Yields:
            Generated tokens as they become available
        """
        response = await self.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        yield response.content
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Number of tokens
        """
        pass
    
    def validate_connection(self) -> bool:
        """
        Validate the provider connection and API key.
        
        Returns:
            True if connection is valid
        """
        return self.api_key is not None or self.provider_name == "ollama"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "options": self.options
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    
    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        status_code: Optional[int] = None
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class RateLimitError(LLMProviderError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(LLMProviderError):
    """Raised when authentication fails."""
    pass


class ModelNotFoundError(LLMProviderError):
    """Raised when the specified model is not found."""
    pass
