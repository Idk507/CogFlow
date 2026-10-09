# Prompts module exports
"""Prompt templates for CogFlow reasoning."""

from cogflow.prompts.cot_templates import (
    PromptTemplate,
    TemplateRegistry,
    get_template_registry,
    get_template,
    format_template,
    COT_BASIC,
    COT_DETAILED,
    COT_MATH,
    COT_ANALYSIS,
    REACT_SYSTEM,
    REACT_STEP,
    REACT_WITH_EXAMPLES,
    RERANK_COMPARE,
    RERANK_SCORE,
    RERANK_SELECT_BEST,
    PLANNING_TEMPLATE,
    SUMMARIZATION_TEMPLATE,
    VERIFICATION_TEMPLATE,
)

__all__ = [
    "PromptTemplate",
    "TemplateRegistry",
    "get_template_registry",
    "get_template",
    "format_template",
    "COT_BASIC",
    "COT_DETAILED",
    "COT_MATH",
    "COT_ANALYSIS",
    "REACT_SYSTEM",
    "REACT_STEP",
    "REACT_WITH_EXAMPLES",
    "RERANK_COMPARE",
    "RERANK_SCORE",
    "RERANK_SELECT_BEST",
    "PLANNING_TEMPLATE",
    "SUMMARIZATION_TEMPLATE",
    "VERIFICATION_TEMPLATE",
]
