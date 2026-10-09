"""
CogFlow Reasoner Module

Provides reasoning capabilities through LLM-based and algorithmic approaches.
Supports chain-of-thought reasoning with step-by-step decomposition.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from cogflow.base.types import Message, MessageRole, ReasoningStep
from cogflow.base.llm import BaseLLMProvider
from cogflow.prompts.cot_templates import TemplateRegistry


class ReasonerOutput(BaseModel):
    """Output from a reasoning process."""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Final answer")
    steps: List[ReasoningStep] = Field(
        default_factory=list,
        description="Reasoning steps taken"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class BaseReasoner(ABC):
    """Abstract base class for reasoning implementations.
    
    Reasoners decompose complex problems into steps and
    produce structured reasoning traces.
    """
    
    @abstractmethod
    async def reason(
        self,
        question: str,
        context: Optional[str] = None,
        max_steps: int = 10,
    ) -> ReasonerOutput:
        """Perform reasoning on a question.
        
        Args:
            question: The question to reason about
            context: Optional additional context
            max_steps: Maximum reasoning steps
        
        Returns:
            ReasonerOutput with answer and steps
        """
        pass
    
    def reason_sync(
        self,
        question: str,
        context: Optional[str] = None,
        max_steps: int = 10,
    ) -> ReasonerOutput:
        """Synchronous version of reason."""
        return asyncio.get_event_loop().run_until_complete(
            self.reason(question, context, max_steps)
        )


class LLMReasoner(BaseReasoner):
    """LLM-based reasoner using chain-of-thought prompting.
    
    Uses prompt templates to guide the LLM through structured
    reasoning, extracting individual steps from the response.
    
    Example:
        ```python
        provider = OpenAIProvider(api_key="...")
        reasoner = LLMReasoner(provider)
        
        result = await reasoner.reason(
            "What is 15% of 80?",
            context="Show your calculation steps."
        )
        print(result.answer)
        for step in result.steps:
            print(f"Step {step.step_number}: {step.content}")
        ```
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        template_name: str = "cot_detailed",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        """Initialize LLM reasoner.
        
        Args:
            llm_provider: LLM provider for generation
            template_name: Name of the template to use
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
        """
        self.llm = llm_provider
        self.template_name = template_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._registry = TemplateRegistry()
    
    async def reason(
        self,
        question: str,
        context: Optional[str] = None,
        max_steps: int = 10,
    ) -> ReasonerOutput:
        """Perform chain-of-thought reasoning.
        
        Args:
            question: The question to reason about
            context: Optional additional context
            max_steps: Maximum reasoning steps (for guidance)
        
        Returns:
            ReasonerOutput with structured reasoning
        """
        # Build prompt from template
        template = self._registry.get(self.template_name)
        if template is None:
            # Fallback to basic prompt
            prompt = f"Question: {question}\n"
            if context:
                prompt += f"Context: {context}\n"
            prompt += "\nThink step by step and provide your reasoning:\n"
        else:
            prompt = template.format(
                question=question,
                context=context or "",
            )
        
        # Generate response
        messages = [Message(role=MessageRole.USER, content=prompt)]
        
        response = await self.llm.generate(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        # Parse steps from response
        steps = self._parse_steps(response)
        
        # Extract final answer
        answer = self._extract_answer(response, steps)
        
        return ReasonerOutput(
            question=question,
            answer=answer,
            steps=steps,
            confidence=self._estimate_confidence(steps),
            metadata={
                "template": self.template_name,
                "raw_response": response,
            }
        )
    
    def _parse_steps(self, response: str) -> List[ReasoningStep]:
        """Parse reasoning steps from LLM response.
        
        Looks for numbered steps, bullet points, or structured output.
        
        Args:
            response: Raw LLM response
        
        Returns:
            List of ReasoningStep objects
        """
        import re
        
        steps = []
        
        # Try to find numbered steps (1. or Step 1:)
        patterns = [
            r"(?:Step\s*)?(\d+)[.:]\s*(.+?)(?=(?:Step\s*)?\d+[.:]|$)",
            r"[-•]\s*(.+?)(?=[-•]|$)",
        ]
        
        # Try numbered pattern first
        matches = re.findall(
            r"(?:Step\s*)?(\d+)[.:]\s*(.+?)(?=(?:Step\s*)?\d+[.:]|Therefore|Final|Answer|$)",
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        if matches:
            for num, content in matches:
                content = content.strip()
                if content:
                    steps.append(ReasoningStep(
                        step_number=int(num),
                        content=content,
                        step_type="reasoning",
                    ))
        else:
            # Split by lines and create steps
            lines = response.strip().split("\n")
            step_num = 0
            for line in lines:
                line = line.strip()
                if line and not line.lower().startswith(("therefore", "final answer", "answer:")):
                    # Remove bullet points
                    if line.startswith(("-", "•", "*")):
                        line = line[1:].strip()
                    step_num += 1
                    steps.append(ReasoningStep(
                        step_number=step_num,
                        content=line,
                        step_type="reasoning",
                    ))
        
        return steps[:10]  # Limit to 10 steps
    
    def _extract_answer(
        self,
        response: str,
        steps: List[ReasoningStep]
    ) -> str:
        """Extract final answer from response.
        
        Args:
            response: Raw LLM response
            steps: Parsed reasoning steps
        
        Returns:
            Final answer string
        """
        import re
        
        # Look for explicit answer markers
        patterns = [
            r"(?:Final\s+)?Answer[:\s]+(.+?)$",
            r"Therefore[,:\s]+(.+?)$",
            r"(?:In\s+)?[Cc]onclusion[,:\s]+(.+?)$",
            r"The\s+answer\s+is[:\s]+(.+?)$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up answer
                answer = re.sub(r"\s+", " ", answer)
                return answer[:500]  # Limit length
        
        # If no explicit answer, use last step
        if steps:
            return steps[-1].content
        
        # Fallback to last line of response
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return lines[-1] if lines else response.strip()[:500]
    
    def _estimate_confidence(self, steps: List[ReasoningStep]) -> float:
        """Estimate confidence based on reasoning quality.
        
        Args:
            steps: List of reasoning steps
        
        Returns:
            Confidence score 0-1
        """
        if not steps:
            return 0.5
        
        # More steps often mean more thorough reasoning
        step_score = min(len(steps) / 5, 1.0) * 0.5
        
        # Longer, more detailed steps are better
        avg_length = sum(len(s.content) for s in steps) / len(steps)
        length_score = min(avg_length / 100, 1.0) * 0.5
        
        return step_score + length_score


class AlgorithmicReasoner(BaseReasoner):
    """Rule-based reasoning for structured problems.
    
    Handles problems that can be solved through deterministic
    algorithms rather than LLM inference.
    
    Example:
        ```python
        reasoner = AlgorithmicReasoner()
        result = await reasoner.reason("Calculate 15% of 80")
        ```
    """
    
    def __init__(self):
        """Initialize algorithmic reasoner."""
        self._handlers = {
            "math": self._handle_math,
            "logic": self._handle_logic,
            "comparison": self._handle_comparison,
        }
    
    async def reason(
        self,
        question: str,
        context: Optional[str] = None,
        max_steps: int = 10,
    ) -> ReasonerOutput:
        """Perform algorithmic reasoning.
        
        Args:
            question: The question to reason about
            context: Optional additional context
            max_steps: Maximum reasoning steps
        
        Returns:
            ReasonerOutput with answer and steps
        """
        steps = []
        
        # Classify the problem type
        problem_type = self._classify_problem(question)
        steps.append(ReasoningStep(
            step_number=1,
            content=f"Identified problem type: {problem_type}",
            step_type="analysis",
        ))
        
        # Apply appropriate handler
        if problem_type in self._handlers:
            handler = self._handlers[problem_type]
            result_steps, answer = await handler(question, context)
            steps.extend(result_steps)
        else:
            answer = "Unable to solve algorithmically. Consider using LLM reasoner."
            steps.append(ReasoningStep(
                step_number=2,
                content="Problem type not supported for algorithmic reasoning",
                step_type="error",
            ))
        
        return ReasonerOutput(
            question=question,
            answer=answer,
            steps=steps,
            confidence=1.0 if problem_type in self._handlers else 0.0,
            metadata={"problem_type": problem_type}
        )
    
    def _classify_problem(self, question: str) -> str:
        """Classify the problem type.
        
        Args:
            question: The question text
        
        Returns:
            Problem type string
        """
        question_lower = question.lower()
        
        # Math patterns
        math_keywords = ["calculate", "compute", "what is", "sum", "product",
                        "difference", "%", "percent", "+", "-", "*", "/", "="]
        if any(kw in question_lower for kw in math_keywords):
            return "math"
        
        # Logic patterns
        logic_keywords = ["if", "then", "and", "or", "not", "implies", "therefore"]
        if any(kw in question_lower for kw in logic_keywords):
            return "logic"
        
        # Comparison patterns
        compare_keywords = ["compare", "greater", "less", "equal", "difference between"]
        if any(kw in question_lower for kw in compare_keywords):
            return "comparison"
        
        return "unknown"
    
    async def _handle_math(
        self,
        question: str,
        context: Optional[str]
    ) -> tuple[List[ReasoningStep], str]:
        """Handle mathematical problems.
        
        Args:
            question: Math question
            context: Optional context
        
        Returns:
            Tuple of (steps, answer)
        """
        import re
        
        steps = []
        
        # Extract numbers and operations
        numbers = re.findall(r'-?\d+\.?\d*', question)
        
        steps.append(ReasoningStep(
            step_number=2,
            content=f"Extracted numbers: {numbers}",
            step_type="extraction",
        ))
        
        # Check for percentage
        if "%" in question or "percent" in question.lower():
            if len(numbers) >= 2:
                percent = float(numbers[0])
                value = float(numbers[1])
                result = (percent / 100) * value
                
                steps.append(ReasoningStep(
                    step_number=3,
                    content=f"Calculating {percent}% of {value}",
                    step_type="calculation",
                ))
                steps.append(ReasoningStep(
                    step_number=4,
                    content=f"Formula: ({percent}/100) × {value} = {result}",
                    step_type="calculation",
                ))
                
                return steps, str(result)
        
        # Try basic arithmetic
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            
            if "+" in question or "sum" in question.lower() or "add" in question.lower():
                result = a + b
                op = "+"
            elif "-" in question or "difference" in question.lower() or "subtract" in question.lower():
                result = a - b
                op = "-"
            elif "*" in question or "×" in question or "product" in question.lower() or "multiply" in question.lower():
                result = a * b
                op = "×"
            elif "/" in question or "÷" in question or "divide" in question.lower():
                result = a / b if b != 0 else float('inf')
                op = "÷"
            else:
                return steps, "Could not determine operation"
            
            steps.append(ReasoningStep(
                step_number=3,
                content=f"Operation: {a} {op} {b} = {result}",
                step_type="calculation",
            ))
            
            return steps, str(result)
        
        return steps, "Could not solve: insufficient information"
    
    async def _handle_logic(
        self,
        question: str,
        context: Optional[str]
    ) -> tuple[List[ReasoningStep], str]:
        """Handle logical problems.
        
        Args:
            question: Logic question
            context: Optional context
        
        Returns:
            Tuple of (steps, answer)
        """
        steps = [
            ReasoningStep(
                step_number=2,
                content="Analyzing logical structure",
                step_type="analysis",
            ),
            ReasoningStep(
                step_number=3,
                content="Logic problems require more complex parsing. Consider LLM reasoner.",
                step_type="limitation",
            ),
        ]
        return steps, "Logic reasoning not fully implemented. Use LLM reasoner."
    
    async def _handle_comparison(
        self,
        question: str,
        context: Optional[str]
    ) -> tuple[List[ReasoningStep], str]:
        """Handle comparison problems.
        
        Args:
            question: Comparison question
            context: Optional context
        
        Returns:
            Tuple of (steps, answer)
        """
        import re
        
        steps = []
        numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', question)]
        
        if len(numbers) >= 2:
            a, b = numbers[0], numbers[1]
            
            steps.append(ReasoningStep(
                step_number=2,
                content=f"Comparing {a} and {b}",
                step_type="comparison",
            ))
            
            if a > b:
                answer = f"{a} is greater than {b}"
            elif a < b:
                answer = f"{a} is less than {b}"
            else:
                answer = f"{a} is equal to {b}"
            
            steps.append(ReasoningStep(
                step_number=3,
                content=answer,
                step_type="conclusion",
            ))
            
            return steps, answer
        
        return steps, "Could not compare: need at least two numbers"


# Export all
__all__ = [
    "ReasonerOutput",
    "BaseReasoner",
    "LLMReasoner",
    "AlgorithmicReasoner",
]
