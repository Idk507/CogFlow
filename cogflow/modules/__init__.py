"""
CogFlow Modules Package

Core module implementations for reasoning, agents, and tools.
"""

from cogflow.modules.tool_registry import ToolRegistry, FunctionTool, register_tool
from cogflow.modules.reasoner import (
    ReasonerOutput,
    BaseReasoner,
    LLMReasoner,
    AlgorithmicReasoner,
)
from cogflow.modules.agent import (
    AgentState,
    AgentConfig,
    ReActAgent,
    SimpleAgent,
)
from cogflow.modules.reranker import (
    RerankResult,
    BaseReranker,
    LLMReranker,
    CrossEncoderReranker,
    HybridReranker,
)
from cogflow.modules.candidate_generator import (
    GenerationConfig,
    BaseCandidateGenerator,
    LLMCandidateGenerator,
    DiversePromptGenerator,
    SelfConsistencyAggregator,
    EnsembleCandidateGenerator,
)
from cogflow.modules.output_formatter import (
    OutputFormat,
    FormattedOutput,
    BaseFormatter,
    PlainTextFormatter,
    MarkdownFormatter,
    JSONFormatter,
    HTMLFormatter,
    OutputFormatterFactory,
    format_output,
    format_reasoning,
    format_response,
)

__all__ = [
    # Tool Registry
    "ToolRegistry",
    "FunctionTool",
    "register_tool",
    # Reasoner
    "ReasonerOutput",
    "BaseReasoner",
    "LLMReasoner",
    "AlgorithmicReasoner",
    # Agent
    "AgentState",
    "AgentConfig",
    "ReActAgent",
    "SimpleAgent",
    # Reranker
    "RerankResult",
    "BaseReranker",
    "LLMReranker",
    "CrossEncoderReranker",
    "HybridReranker",
    # Candidate Generator
    "GenerationConfig",
    "BaseCandidateGenerator",
    "LLMCandidateGenerator",
    "DiversePromptGenerator",
    "SelfConsistencyAggregator",
    "EnsembleCandidateGenerator",
    # Output Formatter
    "OutputFormat",
    "FormattedOutput",
    "BaseFormatter",
    "PlainTextFormatter",
    "MarkdownFormatter",
    "JSONFormatter",
    "HTMLFormatter",
    "OutputFormatterFactory",
    "format_output",
    "format_reasoning",
    "format_response",
]
