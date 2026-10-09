"""
CogFlow Core Type Definitions

This module defines the core data structures used throughout CogFlow:
- ReasoningStep: A single step in the reasoning process
- AgentAction: An action to be taken by an agent
- AgentFinish: The final output of an agent
- Candidate: A candidate response with metadata
- ToolCall: A tool invocation request
- Message: A message in a conversation
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class MessageRole(str, Enum):
    """Role of a message in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A message in a conversation."""
    
    role: MessageRole = Field(description="Role of the message sender")
    content: str = Field(description="Content of the message")
    name: Optional[str] = Field(default=None, description="Optional name")
    tool_call_id: Optional[str] = Field(
        default=None, 
        description="ID for tool response messages"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Message timestamp"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


class ReasoningStep(BaseModel):
    """
    A single step in the reasoning process.
    
    Represents one iteration of the Thought → Action → Observation cycle.
    """
    
    step_number: int = Field(default=0, description="Step sequence number")
    thought: str = Field(description="The reasoning thought")
    action: Optional[str] = Field(
        default=None, 
        description="Action to take (tool name)"
    )
    action_input: Optional[Any] = Field(
        default=None, 
        description="Input for the action"
    )
    observation: Optional[str] = Field(
        default=None, 
        description="Result of the action"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Step timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional step metadata"
    )
    
    def is_complete(self) -> bool:
        """Check if this step has observation (completed)."""
        return self.observation is not None
    
    def to_string(self) -> str:
        """Format step as readable string."""
        parts = [f"Thought: {self.thought}"]
        if self.action:
            parts.append(f"Action: {self.action}")
            if self.action_input:
                parts.append(f"Action Input: {self.action_input}")
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return "\n".join(parts)


class ToolCall(BaseModel):
    """A request to invoke a tool."""
    
    id: str = Field(description="Unique identifier for this tool call")
    name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments
        }


class AgentAction(BaseModel):
    """
    Represents an action to be taken by an agent.
    
    This is returned when the agent decides to use a tool.
    """
    
    tool: str = Field(description="Name of the tool to use")
    tool_input: Any = Field(description="Input to pass to the tool")
    log: str = Field(
        default="",
        description="Log of agent's reasoning for this action"
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Unique ID for tracking this tool call"
    )
    
    def to_reasoning_step(self) -> ReasoningStep:
        """Convert to a ReasoningStep."""
        return ReasoningStep(
            thought=self.log,
            action=self.tool,
            action_input=self.tool_input
        )


class AgentFinish(BaseModel):
    """
    Represents the final output of an agent.
    
    This is returned when the agent decides it has completed the task.
    """
    
    return_values: Dict[str, Any] = Field(
        description="Dictionary of return values"
    )
    log: str = Field(
        default="",
        description="Log of agent's final reasoning"
    )
    
    @property
    def output(self) -> str:
        """Get the main output value."""
        return self.return_values.get("output", str(self.return_values))


class Candidate(BaseModel):
    """
    A candidate response with metadata.
    
    Used for self-consistency and reranking - represents one possible
    answer from the reasoning process.
    """
    
    content: str = Field(description="The candidate response content")
    reasoning_trace: List[ReasoningStep] = Field(
        default_factory=list,
        description="List of reasoning steps that led to this candidate"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)"
    )
    score: Optional[float] = Field(
        default=None,
        description="Optional reranking score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    source: str = Field(
        default="unknown",
        description="Source of this candidate (model name, method, etc.)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Generation timestamp"
    )
    
    def get_reasoning_summary(self) -> str:
        """Get a summary of the reasoning trace."""
        if not self.reasoning_trace:
            return "No reasoning trace available."
        steps = [step.to_string() for step in self.reasoning_trace]
        return "\n---\n".join(steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "confidence": self.confidence,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
            "reasoning_steps": len(self.reasoning_trace)
        }


# Type aliases for clarity
AgentOutput = Union[AgentAction, AgentFinish]
IntermediateStep = tuple[AgentAction, str]  # (action, observation)
