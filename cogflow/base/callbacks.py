"""
CogFlow Callback System

This module provides a callback/event interface for monitoring and 
hooking into the execution lifecycle of CogFlow components.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio


class EventType(str, Enum):
    """Types of events that can occur during execution."""
    
    # Agent events
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_STEP_START = "agent_step_start"
    AGENT_STEP_END = "agent_step_end"
    AGENT_ERROR = "agent_error"
    
    # Tool events
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    
    # Reasoning events
    REASONING_START = "reasoning_start"
    REASONING_STEP = "reasoning_step"
    REASONING_END = "reasoning_end"
    
    # LLM events
    LLM_REQUEST_START = "llm_request_start"
    LLM_REQUEST_END = "llm_request_end"
    LLM_STREAM_TOKEN = "llm_stream_token"
    LLM_ERROR = "llm_error"
    
    # Reranking events
    RERANK_START = "rerank_start"
    RERANK_END = "rerank_end"
    
    # Pipeline events
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    PIPELINE_STEP = "pipeline_step"


@dataclass
class Event:
    """
    Represents an event in the execution lifecycle.
    
    Attributes:
        type: The type of event
        timestamp: When the event occurred
        data: Event-specific data
        metadata: Additional metadata
        trace_id: ID linking related events
        parent_id: ID of parent event (for nested events)
    """
    
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
        }


class BaseCallback(ABC):
    """
    Abstract base class for callbacks.
    
    Callbacks receive events and can react to them. They can be used
    for logging, monitoring, debugging, or custom integrations.
    """
    
    @abstractmethod
    async def on_event(self, event: Event) -> None:
        """
        Handle an event.
        
        Args:
            event: The event to handle
        """
        pass
    
    async def on_agent_start(
        self,
        agent_name: str,
        input_text: str,
        **kwargs: Any
    ) -> None:
        """Called when an agent starts processing."""
        pass
    
    async def on_agent_end(
        self,
        agent_name: str,
        output: str,
        **kwargs: Any
    ) -> None:
        """Called when an agent finishes processing."""
        pass
    
    async def on_agent_error(
        self,
        agent_name: str,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when an agent encounters an error."""
        pass
    
    async def on_tool_start(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when a tool starts execution."""
        pass
    
    async def on_tool_end(
        self,
        tool_name: str,
        tool_output: Any,
        **kwargs: Any
    ) -> None:
        """Called when a tool finishes execution."""
        pass
    
    async def on_tool_error(
        self,
        tool_name: str,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when a tool encounters an error."""
        pass
    
    async def on_llm_start(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any
    ) -> None:
        """Called when an LLM request starts."""
        pass
    
    async def on_llm_end(
        self,
        model: str,
        response: str,
        **kwargs: Any
    ) -> None:
        """Called when an LLM request completes."""
        pass
    
    async def on_llm_token(
        self,
        token: str,
        **kwargs: Any
    ) -> None:
        """Called for each streaming token from LLM."""
        pass


class LoggingCallback(BaseCallback):
    """
    A callback that logs all events.
    
    Useful for debugging and monitoring execution.
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        include_data: bool = True,
        logger: Optional[Any] = None
    ):
        """
        Initialize logging callback.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            include_data: Whether to include event data in logs
            logger: Optional custom logger instance
        """
        self.log_level = log_level
        self.include_data = include_data
        self.logger = logger
        self._events: List[Event] = []
    
    async def on_event(self, event: Event) -> None:
        """Log the event."""
        self._events.append(event)
        
        message = f"[{event.type.value}]"
        if event.trace_id:
            message += f" trace={event.trace_id}"
        if self.include_data and event.data:
            message += f" data={event.data}"
        
        if self.logger:
            log_method = getattr(
                self.logger,
                self.log_level.lower(),
                self.logger.info
            )
            log_method(message)
        else:
            print(f"{event.timestamp.isoformat()} {message}")
    
    async def on_agent_start(
        self,
        agent_name: str,
        input_text: str,
        **kwargs: Any
    ) -> None:
        event = Event(
            type=EventType.AGENT_START,
            data={"agent": agent_name, "input": input_text[:100]},
        )
        await self.on_event(event)
    
    async def on_agent_end(
        self,
        agent_name: str,
        output: str,
        **kwargs: Any
    ) -> None:
        event = Event(
            type=EventType.AGENT_END,
            data={"agent": agent_name, "output": output[:100]},
        )
        await self.on_event(event)
    
    async def on_tool_start(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        event = Event(
            type=EventType.TOOL_START,
            data={"tool": tool_name, "input": tool_input},
        )
        await self.on_event(event)
    
    async def on_tool_end(
        self,
        tool_name: str,
        tool_output: Any,
        **kwargs: Any
    ) -> None:
        event = Event(
            type=EventType.TOOL_END,
            data={"tool": tool_name, "output": str(tool_output)[:100]},
        )
        await self.on_event(event)
    
    def get_events(self) -> List[Event]:
        """Get all recorded events."""
        return list(self._events)
    
    def clear(self) -> None:
        """Clear recorded events."""
        self._events.clear()


class CompositeCallback(BaseCallback):
    """
    A callback that delegates to multiple other callbacks.
    
    Useful for combining multiple callback handlers.
    """
    
    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        """
        Initialize composite callback.
        
        Args:
            callbacks: List of callbacks to delegate to
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: BaseCallback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: BaseCallback) -> None:
        """Remove a callback from the list."""
        self.callbacks.remove(callback)
    
    async def on_event(self, event: Event) -> None:
        """Delegate event to all callbacks."""
        await asyncio.gather(*[
            cb.on_event(event) for cb in self.callbacks
        ])
    
    async def on_agent_start(
        self,
        agent_name: str,
        input_text: str,
        **kwargs: Any
    ) -> None:
        await asyncio.gather(*[
            cb.on_agent_start(agent_name, input_text, **kwargs)
            for cb in self.callbacks
        ])
    
    async def on_agent_end(
        self,
        agent_name: str,
        output: str,
        **kwargs: Any
    ) -> None:
        await asyncio.gather(*[
            cb.on_agent_end(agent_name, output, **kwargs)
            for cb in self.callbacks
        ])
    
    async def on_tool_start(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        await asyncio.gather(*[
            cb.on_tool_start(tool_name, tool_input, **kwargs)
            for cb in self.callbacks
        ])
    
    async def on_tool_end(
        self,
        tool_name: str,
        tool_output: Any,
        **kwargs: Any
    ) -> None:
        await asyncio.gather(*[
            cb.on_tool_end(tool_name, tool_output, **kwargs)
            for cb in self.callbacks
        ])


class TracingCallback(BaseCallback):
    """
    A callback that builds execution traces.
    
    Collects events with trace IDs to build a hierarchical
    view of execution.
    """
    
    def __init__(self):
        """Initialize tracing callback."""
        self.traces: Dict[str, List[Event]] = {}
        self._current_trace: Optional[str] = None
    
    def start_trace(self, trace_id: str) -> None:
        """Start a new trace."""
        self._current_trace = trace_id
        self.traces[trace_id] = []
    
    def end_trace(self) -> Optional[str]:
        """End the current trace and return its ID."""
        trace_id = self._current_trace
        self._current_trace = None
        return trace_id
    
    async def on_event(self, event: Event) -> None:
        """Add event to current trace."""
        if self._current_trace:
            event.trace_id = self._current_trace
            self.traces[self._current_trace].append(event)
    
    def get_trace(self, trace_id: str) -> List[Event]:
        """Get events for a specific trace."""
        return self.traces.get(trace_id, [])
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get a summary of a trace."""
        events = self.get_trace(trace_id)
        if not events:
            return {}
        
        return {
            "trace_id": trace_id,
            "event_count": len(events),
            "start_time": events[0].timestamp.isoformat(),
            "end_time": events[-1].timestamp.isoformat(),
            "event_types": list(set(e.type.value for e in events)),
        }


class CallbackManager:
    """
    Manages callbacks for CogFlow components.
    
    Provides a central place to register and invoke callbacks.
    """
    
    def __init__(self):
        """Initialize callback manager."""
        self._callbacks: List[BaseCallback] = []
    
    def add_callback(self, callback: BaseCallback) -> None:
        """Register a callback."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: BaseCallback) -> None:
        """Remove a registered callback."""
        self._callbacks.remove(callback)
    
    def clear_callbacks(self) -> None:
        """Remove all callbacks."""
        self._callbacks.clear()
    
    async def emit(self, event: Event) -> None:
        """Emit an event to all callbacks."""
        if not self._callbacks:
            return
        await asyncio.gather(*[
            cb.on_event(event) for cb in self._callbacks
        ])
    
    async def emit_agent_start(
        self,
        agent_name: str,
        input_text: str,
        **kwargs: Any
    ) -> None:
        """Emit agent start event."""
        event = Event(
            type=EventType.AGENT_START,
            data={"agent": agent_name, "input": input_text},
            metadata=kwargs,
        )
        await self.emit(event)
        await asyncio.gather(*[
            cb.on_agent_start(agent_name, input_text, **kwargs)
            for cb in self._callbacks
        ])
    
    async def emit_agent_end(
        self,
        agent_name: str,
        output: str,
        **kwargs: Any
    ) -> None:
        """Emit agent end event."""
        event = Event(
            type=EventType.AGENT_END,
            data={"agent": agent_name, "output": output},
            metadata=kwargs,
        )
        await self.emit(event)
        await asyncio.gather(*[
            cb.on_agent_end(agent_name, output, **kwargs)
            for cb in self._callbacks
        ])
    
    async def emit_tool_start(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Emit tool start event."""
        event = Event(
            type=EventType.TOOL_START,
            data={"tool": tool_name, "input": tool_input},
            metadata=kwargs,
        )
        await self.emit(event)
        await asyncio.gather(*[
            cb.on_tool_start(tool_name, tool_input, **kwargs)
            for cb in self._callbacks
        ])
    
    async def emit_tool_end(
        self,
        tool_name: str,
        tool_output: Any,
        **kwargs: Any
    ) -> None:
        """Emit tool end event."""
        event = Event(
            type=EventType.TOOL_END,
            data={"tool": tool_name, "output": tool_output},
            metadata=kwargs,
        )
        await self.emit(event)
        await asyncio.gather(*[
            cb.on_tool_end(tool_name, tool_output, **kwargs)
            for cb in self._callbacks
        ])


# Global callback manager instance
_global_callback_manager: Optional[CallbackManager] = None


def get_callback_manager() -> CallbackManager:
    """Get the global callback manager."""
    global _global_callback_manager
    if _global_callback_manager is None:
        _global_callback_manager = CallbackManager()
    return _global_callback_manager


def set_callback_manager(manager: CallbackManager) -> None:
    """Set the global callback manager."""
    global _global_callback_manager
    _global_callback_manager = manager


__all__ = [
    "EventType",
    "Event",
    "BaseCallback",
    "LoggingCallback",
    "CompositeCallback",
    "TracingCallback",
    "CallbackManager",
    "get_callback_manager",
    "set_callback_manager",
]
