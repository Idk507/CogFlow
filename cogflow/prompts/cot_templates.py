"""
CogFlow Prompt Templates

This module provides Chain-of-Thought (CoT) and ReAct prompt templates
for various reasoning tasks.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from string import Template


class PromptTemplate(BaseModel):
    """A reusable prompt template with variable substitution."""
    
    name: str = Field(description="Template name for identification")
    template: str = Field(description="Template string with {variables}")
    input_variables: List[str] = Field(
        default_factory=list,
        description="List of required input variables"
    )
    description: str = Field(
        default="",
        description="Description of what this template is for"
    )
    
    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables."""
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return self.template.format(**kwargs)
    
    def partial(self, **kwargs: Any) -> "PromptTemplate":
        """Create a new template with some variables filled in."""
        new_template = self.template.format(**{
            k: "{" + k + "}" if k not in kwargs else kwargs[k]
            for k in self.input_variables
        })
        remaining_vars = [v for v in self.input_variables if v not in kwargs]
        return PromptTemplate(
            name=f"{self.name}_partial",
            template=new_template,
            input_variables=remaining_vars,
            description=self.description
        )


# =============================================================================
# Chain-of-Thought (CoT) Templates
# =============================================================================

COT_BASIC = PromptTemplate(
    name="cot_basic",
    template="""You are a helpful AI assistant that thinks step by step.

Question: {question}

Let's think through this step by step:
1.""",
    input_variables=["question"],
    description="Basic chain-of-thought prompt for general reasoning"
)

COT_DETAILED = PromptTemplate(
    name="cot_detailed",
    template="""You are an expert problem solver. Approach this systematically.

Problem: {question}

Context: {context}

Instructions:
1. First, understand what is being asked
2. Break down the problem into smaller parts
3. Solve each part step by step
4. Combine the results into a final answer

Let me work through this carefully:

Step 1: Understanding the problem""",
    input_variables=["question", "context"],
    description="Detailed CoT prompt with context support"
)

COT_MATH = PromptTemplate(
    name="cot_math",
    template="""You are a mathematics expert. Solve this problem step by step.

Problem: {question}

Show your work:
- Identify the given information
- Determine what needs to be found
- Apply relevant formulas/methods
- Calculate step by step
- Verify your answer

Solution:""",
    input_variables=["question"],
    description="Math-focused CoT prompt"
)

COT_ANALYSIS = PromptTemplate(
    name="cot_analysis",
    template="""Analyze the following carefully and provide a reasoned response.

Topic: {topic}
Question: {question}

Analysis Framework:
1. Key observations
2. Relevant factors to consider
3. Analysis of each factor
4. Synthesis and conclusion

Begin analysis:""",
    input_variables=["topic", "question"],
    description="Analytical reasoning template"
)

# =============================================================================
# ReAct (Reasoning + Acting) Templates
# =============================================================================

REACT_SYSTEM = PromptTemplate(
    name="react_system",
    template="""You are an AI assistant that solves problems using a 
Thought-Action-Observation loop.

Available Tools:
{tools}

Format your responses as:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [input for the tool]

When you have the final answer:
Thought: [Your final reasoning]
Final Answer: [Your complete response]

Important:
- Always think before acting
- Use tools when you need external information
- Provide clear, step-by-step reasoning""",
    input_variables=["tools"],
    description="System prompt for ReAct agents"
)

REACT_STEP = PromptTemplate(
    name="react_step",
    template="""Task: {task}

{history}

Continue with the next step:""",
    input_variables=["task", "history"],
    description="Template for each ReAct step"
)

REACT_WITH_EXAMPLES = PromptTemplate(
    name="react_with_examples",
    template="""You solve problems step by step using available tools.

Available Tools:
{tools}

Example:
Task: What is the weather in Paris?
Thought: I need to search for current weather in Paris
Action: search
Action Input: current weather Paris
Observation: Paris weather: 15°C, partly cloudy
Thought: I now have the weather information
Final Answer: The current weather in Paris is 15°C and partly cloudy.

Now solve this task:
Task: {task}

Begin:""",
    input_variables=["tools", "task"],
    description="ReAct template with examples"
)

# =============================================================================
# Reranking Templates
# =============================================================================

RERANK_COMPARE = PromptTemplate(
    name="rerank_compare",
    template="""Compare these candidate answers and select the best one.

Question: {question}

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}

Evaluation criteria:
1. Accuracy - Is the answer correct?
2. Completeness - Does it fully address the question?
3. Clarity - Is it well-explained?

Which candidate is better? Respond with just "A" or "B" and a brief reason.""",
    input_variables=["question", "candidate_a", "candidate_b"],
    description="Pairwise comparison template for reranking"
)

RERANK_SCORE = PromptTemplate(
    name="rerank_score",
    template="""Evaluate this answer on a scale of 1-10.

Question: {question}

Answer: {answer}

Scoring criteria:
- Accuracy (1-10): Is the answer factually correct?
- Relevance (1-10): Does it address the question?
- Clarity (1-10): Is it well-explained?

Provide scores and calculate the average:
Accuracy: 
Relevance:
Clarity:
Average:""",
    input_variables=["question", "answer"],
    description="Scoring template for individual candidate evaluation"
)

RERANK_SELECT_BEST = PromptTemplate(
    name="rerank_select_best",
    template="""Select the best answer from these candidates.

Question: {question}

Candidates:
{candidates}

Analyze each candidate and select the best one.
Respond with the number of the best candidate and explain why.""",
    input_variables=["question", "candidates"],
    description="Template for selecting best from multiple candidates"
)

# =============================================================================
# Specialized Templates
# =============================================================================

PLANNING_TEMPLATE = PromptTemplate(
    name="planning",
    template="""Create a detailed plan to accomplish this goal.

Goal: {goal}
Constraints: {constraints}

Requirements:
1. Break down into actionable steps
2. Consider dependencies between steps
3. Estimate effort for each step
4. Identify potential risks

Plan:""",
    input_variables=["goal", "constraints"],
    description="Planning and task decomposition template"
)

SUMMARIZATION_TEMPLATE = PromptTemplate(
    name="summarization",
    template="""Summarize the following content.

Content:
{content}

Requirements:
- Keep the key points
- Maintain accuracy
- Target length: {length}

Summary:""",
    input_variables=["content", "length"],
    description="Content summarization template"
)

VERIFICATION_TEMPLATE = PromptTemplate(
    name="verification",
    template="""Verify the following claim or answer.

Claim: {claim}
Context: {context}

Verification steps:
1. Identify key assertions
2. Check for logical consistency
3. Compare with known facts
4. Assess confidence level

Verification result:""",
    input_variables=["claim", "context"],
    description="Answer/claim verification template"
)


# =============================================================================
# Template Registry
# =============================================================================

class TemplateRegistry:
    """Registry for managing prompt templates."""
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register all default templates."""
        defaults = [
            COT_BASIC, COT_DETAILED, COT_MATH, COT_ANALYSIS,
            REACT_SYSTEM, REACT_STEP, REACT_WITH_EXAMPLES,
            RERANK_COMPARE, RERANK_SCORE, RERANK_SELECT_BEST,
            PLANNING_TEMPLATE, SUMMARIZATION_TEMPLATE, VERIFICATION_TEMPLATE,
        ]
        for template in defaults:
            self.register(template)
    
    def register(self, template: PromptTemplate) -> None:
        """Register a template."""
        self._templates[template.name] = template
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return list(self._templates.keys())
    
    def get_by_category(self, prefix: str) -> List[PromptTemplate]:
        """Get all templates with names starting with prefix."""
        return [
            t for name, t in self._templates.items()
            if name.startswith(prefix)
        ]


# Global template registry
_registry: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """Get the global template registry."""
    global _registry
    if _registry is None:
        _registry = TemplateRegistry()
    return _registry


def get_template(name: str) -> Optional[PromptTemplate]:
    """Get a template by name from the global registry."""
    return get_template_registry().get(name)


def format_template(name: str, **kwargs: Any) -> str:
    """Format a template by name with provided variables."""
    template = get_template(name)
    if template is None:
        raise ValueError(f"Template not found: {name}")
    return template.format(**kwargs)
