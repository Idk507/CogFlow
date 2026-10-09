"""
Unit Tests for CogFlow Types

Tests for the core data types in cogflow.base.types
"""

import pytest
from cogflow.base.types import (
    ReasoningStep,
    AgentAction,
    AgentFinish,
    Candidate,
    ToolCall,
    Message,
    MessageRole,
)


class TestReasoningStep:
    """Tests for ReasoningStep model."""
    
    def test_create_reasoning_step(self):
        """Test basic ReasoningStep creation."""
        step = ReasoningStep(
            step_number=1,
            thought="This is a reasoning step",
            action="calculator",
            action_input={"expression": "2+2"},
        )
        assert step.step_number == 1
        assert step.thought == "This is a reasoning step"
        assert step.action == "calculator"
    
    def test_reasoning_step_minimal(self):
        """Test ReasoningStep with just thought."""
        step = ReasoningStep(
            thought="Just thinking",
        )
        assert step.step_number == 0  # default
        assert step.action is None
        assert step.observation is None
    
    def test_reasoning_step_with_metadata(self):
        """Test ReasoningStep with metadata."""
        step = ReasoningStep(
            step_number=2,
            thought="Step with metadata",
            metadata={"confidence": 0.95},
        )
        assert step.metadata["confidence"] == 0.95
    
    def test_reasoning_step_is_complete(self):
        """Test is_complete method."""
        incomplete = ReasoningStep(thought="Thinking")
        complete = ReasoningStep(thought="Thinking", observation="Result")
        
        assert not incomplete.is_complete()
        assert complete.is_complete()
    
    def test_reasoning_step_to_string(self):
        """Test to_string formatting."""
        step = ReasoningStep(
            thought="I should calculate",
            action="calculator",
            action_input="2+2",
            observation="4",
        )
        s = step.to_string()
        assert "Thought: I should calculate" in s
        assert "Action: calculator" in s
        assert "Observation: 4" in s


class TestAgentAction:
    """Tests for AgentAction model."""
    
    def test_create_agent_action(self):
        """Test AgentAction creation."""
        action = AgentAction(
            tool="calculator",
            tool_input={"expression": "2 + 2"},
            log="I need to calculate this",
        )
        assert action.tool == "calculator"
        assert action.tool_input["expression"] == "2 + 2"
        assert action.log == "I need to calculate this"
    
    def test_agent_action_default_log(self):
        """Test AgentAction with default log."""
        action = AgentAction(
            tool="search",
            tool_input={"query": "test"},
        )
        assert action.log == ""
    
    def test_agent_action_to_reasoning_step(self):
        """Test conversion to ReasoningStep."""
        action = AgentAction(
            tool="search",
            tool_input={"query": "weather"},
            log="Need to find weather info",
        )
        step = action.to_reasoning_step()
        assert step.thought == "Need to find weather info"
        assert step.action == "search"


class TestAgentFinish:
    """Tests for AgentFinish model."""
    
    def test_create_agent_finish(self):
        """Test AgentFinish creation."""
        finish = AgentFinish(
            return_values={"output": "The answer is 42"},
            log="I have found the answer",
        )
        assert finish.output == "The answer is 42"
        assert finish.log == "I have found the answer"
    
    def test_agent_finish_default_log(self):
        """Test AgentFinish with default log."""
        finish = AgentFinish(return_values={"output": "Done"})
        assert finish.output == "Done"
        assert finish.log == ""
    
    def test_agent_finish_output_property(self):
        """Test output property extraction."""
        finish = AgentFinish(return_values={"output": "Result", "extra": "data"})
        assert finish.output == "Result"


class TestCandidate:
    """Tests for Candidate model."""
    
    def test_create_candidate(self):
        """Test Candidate creation."""
        candidate = Candidate(
            content="This is a candidate response",
            confidence=0.85,
        )
        assert candidate.content == "This is a candidate response"
        assert candidate.confidence == 0.85
    
    def test_candidate_confidence_bounds(self):
        """Test that confidence is properly bounded."""
        candidate = Candidate(content="test", confidence=0.5)
        assert 0.0 <= candidate.confidence <= 1.0
    
    def test_candidate_with_score(self):
        """Test Candidate with score."""
        candidate = Candidate(
            content="Response",
            confidence=0.9,
            score=0.95,
        )
        assert candidate.score == 0.95
    
    def test_candidate_with_metadata(self):
        """Test Candidate with metadata."""
        candidate = Candidate(
            content="Response",
            confidence=0.9,
            metadata={"source": "model_a"},
        )
        assert candidate.metadata["source"] == "model_a"
    
    def test_candidate_to_dict(self):
        """Test to_dict conversion."""
        candidate = Candidate(
            content="test",
            confidence=0.8,
            source="gpt-4",
        )
        d = candidate.to_dict()
        assert d["content"] == "test"
        assert d["confidence"] == 0.8
        assert d["source"] == "gpt-4"


class TestToolCall:
    """Tests for ToolCall model."""
    
    def test_create_tool_call(self):
        """Test ToolCall creation."""
        call = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "weather"},
        )
        assert call.id == "call_123"
        assert call.name == "search"
        assert call.arguments["query"] == "weather"
    
    def test_tool_call_to_dict(self):
        """Test to_dict conversion."""
        call = ToolCall(
            id="call_1",
            name="calc",
            arguments={"expr": "1+1"},
        )
        d = call.to_dict()
        assert d["id"] == "call_1"
        assert d["name"] == "calc"
        assert d["arguments"]["expr"] == "1+1"


class TestMessage:
    """Tests for Message model."""
    
    def test_create_user_message(self):
        """Test user message creation."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello, world!",
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
    
    def test_create_assistant_message(self):
        """Test assistant message creation."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi there!",
        )
        assert msg.role == MessageRole.ASSISTANT
    
    def test_create_system_message(self):
        """Test system message creation."""
        msg = Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant.",
        )
        assert msg.role == MessageRole.SYSTEM
    
    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Response",
            metadata={"tokens": 50},
        )
        assert msg.metadata["tokens"] == 50
    
    def test_message_to_dict(self):
        """Test message dict conversion."""
        msg = Message(
            role=MessageRole.USER,
            content="Test",
        )
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Test"
    
    def test_message_with_name(self):
        """Test message with name field."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello",
            name="alice",
        )
        d = msg.to_dict()
        assert d["name"] == "alice"


class TestMessageRole:
    """Tests for MessageRole enum."""
    
    def test_role_values(self):
        """Test MessageRole enum values."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"
