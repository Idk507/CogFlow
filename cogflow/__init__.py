# CogFlow - A Modular Reasoning Framework
"""
CogFlow is a flexible and extensible framework for building reasoning pipelines
using modular components like chain-of-thought reasoning, reranking, and agentic loops.
"""

__version__ = "0.1.0"
__author__ = "CogFlow Team"

# Core types
from cogflow.base.types import (
    ReasoningStep,
    AgentAction,
    AgentFinish,
    Candidate,
    Message,
    MessageRole,
    ToolCall,
)

# Configuration
from cogflow.config import (
    CogFlowConfig,
    LLMConfig,
    AgentConfig as AgentConfigSettings,
    ReasonerConfig,
    RerankerConfig,
    get_config,
    set_config,
)

# Base classes
from cogflow.base.llm import BaseLLMProvider
from cogflow.base.tool import BaseTool
from cogflow.base.callbacks import BaseCallback, CallbackManager

# Modules
from cogflow.modules.tool_registry import ToolRegistry, register_tool
from cogflow.modules.reasoner import LLMReasoner, AlgorithmicReasoner
from cogflow.modules.agent import ReActAgent, SimpleAgent, AgentConfig
from cogflow.modules.reranker import LLMReranker, HybridReranker
from cogflow.modules.candidate_generator import (
    LLMCandidateGenerator,
    SelfConsistencyAggregator,
)
from cogflow.modules.output_formatter import (
    OutputFormat,
    OutputFormatterFactory,
    format_output,
    format_response,
)

# Pipeline
from cogflow.pipeline import (
    Pipeline,
    PipelineStep,
    PipelineContext,
    create_pipeline,
)

# Prompts
from cogflow.prompts.cot_templates import PromptTemplate, TemplateRegistry

# Tools
from cogflow.tools.builtin import (
    CalculatorTool,
    WebSearchTool,
    TextProcessingTool,
    DateTimeTool,
    JSONTool,
    register_builtin_tools,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core types
    "ReasoningStep",
    "AgentAction",
    "AgentFinish",
    "Candidate",
    "Message",
    "MessageRole",
    "ToolCall",
    # Configuration
    "CogFlowConfig",
    "LLMConfig",
    "AgentConfigSettings",
    "ReasonerConfig",
    "RerankerConfig",
    "get_config",
    "set_config",
    # Base classes
    "BaseLLMProvider",
    "BaseTool",
    "BaseCallback",
    "CallbackManager",
    # Modules
    "ToolRegistry",
    "register_tool",
    "LLMReasoner",
    "AlgorithmicReasoner",
    "ReActAgent",
    "SimpleAgent",
    "AgentConfig",
    "LLMReranker",
    "HybridReranker",
    "LLMCandidateGenerator",
    "SelfConsistencyAggregator",
    "OutputFormat",
    "OutputFormatterFactory",
    "format_output",
    "format_response",
    # Pipeline
    "Pipeline",
    "PipelineStep",
    "PipelineContext",
    "create_pipeline",
    # Prompts
    "PromptTemplate",
    "TemplateRegistry",
    # Built-in tools
    "CalculatorTool",
    "WebSearchTool",
    "TextProcessingTool",
    "DateTimeTool",
    "JSONTool",
    "register_builtin_tools",
]
