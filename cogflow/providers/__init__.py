"""
CogFlow Providers Package

LLM provider implementations for OpenAI, Anthropic, HuggingFace, and Ollama.
"""

# Import providers (with graceful fallback if dependencies missing)
try:
    from .openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None  # type: ignore

try:
    from .anthropic_provider import AnthropicProvider
except ImportError:
    AnthropicProvider = None  # type: ignore

try:
    from .huggingface_provider import HuggingFaceProvider
except ImportError:
    HuggingFaceProvider = None  # type: ignore

try:
    from .ollama_provider import OllamaProvider
except ImportError:
    OllamaProvider = None  # type: ignore


def get_provider(provider_name: str, **kwargs):
    """
    Get a provider instance by name.
    
    Args:
        provider_name: One of 'openai', 'anthropic', 'huggingface', 'ollama'
        **kwargs: Provider-specific arguments
        
    Returns:
        Provider instance
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "ollama": OllamaProvider,
    }
    
    provider_class = providers.get(provider_name.lower())
    if provider_class is None:
        available = [k for k, v in providers.items() if v is not None]
        raise ValueError(
            f"Provider '{provider_name}' not available. "
            f"Available providers: {available}"
        )
    
    return provider_class(**kwargs)


__all__ = [
    "OpenAIProvider",
    "AnthropicProvider",
    "HuggingFaceProvider",
    "OllamaProvider",
    "get_provider",
]
