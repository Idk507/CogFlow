"""
CogFlow Configuration Management

Handles environment variables, API keys, and model configuration settings.
"""

import os
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """Configuration for LLM model parameters."""
    
    model_name: str = Field(default="gpt-4", description="Model name to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=1024, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    timeout: float = Field(default=60.0, gt=0, description="Request timeout in seconds")
    
    @field_validator('temperature', 'top_p')
    @classmethod
    def validate_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Value must be between 0.0 and 2.0")
        return v


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""
    
    max_iterations: int = Field(default=10, gt=0, description="Maximum ReAct loop iterations")
    early_stopping: bool = Field(default=True, description="Stop early if agent finishes")
    return_intermediate_steps: bool = Field(default=True, description="Include reasoning trace")
    handle_parsing_errors: bool = Field(default=True, description="Handle output parsing errors")
    verbose: bool = Field(default=False, description="Enable verbose logging")


class ReasonerConfig(BaseModel):
    """Configuration for reasoning module."""
    
    num_candidates: int = Field(default=1, ge=1, description="Number of candidates to generate")
    use_self_consistency: bool = Field(default=False, description="Enable self-consistency")
    voting_method: Literal["majority", "weighted", "best_score"] = Field(
        default="majority", description="Method for selecting best candidate"
    )
    include_reasoning_trace: bool = Field(default=True, description="Include reasoning steps")


class RerankerConfig(BaseModel):
    """Configuration for reranking module."""
    
    method: Literal["llm", "rule_based", "voting", "ensemble"] = Field(
        default="llm", description="Reranking method to use"
    )
    top_k: int = Field(default=1, ge=1, description="Number of top candidates to return")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")


class CogFlowConfig(BaseSettings):
    """
    Main configuration for CogFlow framework.
    
    Loads settings from environment variables with COGFLOW_ prefix.
    """
    
    model_config = {
        "env_prefix": "COGFLOW_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, alias="HUGGINGFACE_API_KEY")
    
    # Provider settings
    default_provider: Literal["openai", "anthropic", "huggingface", "ollama"] = Field(
        default="openai", description="Default LLM provider"
    )
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    
    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    reasoner: ReasonerConfig = Field(default_factory=ReasonerConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    
    # Logging and debugging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    enable_tracing: bool = Field(default=True, description="Enable execution tracing")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CogFlowConfig":
        """Create configuration from a dictionary."""
        return cls(**config_dict)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "huggingface": self.huggingface_api_key,
        }
        return key_map.get(provider)
    
    def validate_provider(self, provider: str) -> bool:
        """Check if a provider is properly configured."""
        if provider == "ollama":
            return True  # Ollama doesn't require API key
        return self.get_api_key(provider) is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary (excluding sensitive data)."""
        data = self.model_dump()
        # Mask API keys
        for key in ["openai_api_key", "anthropic_api_key", "huggingface_api_key"]:
            if data.get(key):
                data[key] = "***" + data[key][-4:] if len(data[key]) > 4 else "***"
        return data


# Global configuration instance
_config: Optional[CogFlowConfig] = None


def get_config() -> CogFlowConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = CogFlowConfig()
    return _config


def set_config(config: CogFlowConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config_from_env() -> CogFlowConfig:
    """Load configuration from environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    return CogFlowConfig()
