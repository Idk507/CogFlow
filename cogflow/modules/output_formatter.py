"""
Output Formatter Module for CogFlow Framework.

Provides response formatting capabilities for different output formats
including Markdown, JSON, Plain text, and structured formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json
import re
from dataclasses import dataclass

from cogflow.base.types import (
    ReasoningStep,
    AgentAction,
    AgentFinish,
    ToolCall,
    Candidate,
)


class OutputFormat(Enum):
    """Enumeration of supported output formats."""
    
    PLAIN = "plain"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    STRUCTURED = "structured"


@dataclass
class FormattedOutput:
    """Container for formatted output with metadata."""
    
    content: str
    format_type: OutputFormat
    metadata: Dict[str, Any]
    
    def __str__(self) -> str:
        return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "format_type": self.format_type.value,
            "metadata": self.metadata,
        }


class BaseFormatter(ABC):
    """Abstract base class for output formatters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the formatter with optional configuration."""
        self.config = config or {}
    
    @property
    @abstractmethod
    def format_type(self) -> OutputFormat:
        """Return the format type this formatter produces."""
        pass
    
    @abstractmethod
    def format(self, data: Any) -> FormattedOutput:
        """Format the given data into the target format."""
        pass
    
    @abstractmethod
    def format_reasoning_steps(
        self, steps: List[ReasoningStep]
    ) -> FormattedOutput:
        """Format a list of reasoning steps."""
        pass
    
    @abstractmethod
    def format_agent_response(
        self,
        response: Union[AgentAction, AgentFinish],
        include_trace: bool = False,
    ) -> FormattedOutput:
        """Format an agent's response."""
        pass
    
    @abstractmethod
    def format_tool_result(
        self, tool_call: ToolCall, result: Any
    ) -> FormattedOutput:
        """Format a tool call and its result."""
        pass
    
    def _create_output(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FormattedOutput:
        """Helper to create FormattedOutput with this formatter's type."""
        return FormattedOutput(
            content=content,
            format_type=self.format_type,
            metadata=metadata or {},
        )


class PlainTextFormatter(BaseFormatter):
    """Formatter for plain text output."""
    
    @property
    def format_type(self) -> OutputFormat:
        return OutputFormat.PLAIN
    
    def format(self, data: Any) -> FormattedOutput:
        """Format any data as plain text."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = self._format_dict(data)
        elif isinstance(data, list):
            content = self._format_list(data)
        else:
            content = str(data)
        
        return self._create_output(content)
    
    def _format_dict(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Format a dictionary as indented plain text."""
        lines = []
        prefix = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_list(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)
    
    def _format_list(self, data: List[Any], indent: int = 0) -> str:
        """Format a list as plain text."""
        lines = []
        prefix = "  " * indent
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                lines.append(f"{prefix}{i}.")
                lines.append(self._format_dict(item, indent + 1))
            else:
                lines.append(f"{prefix}{i}. {item}")
        return "\n".join(lines)
    
    def format_reasoning_steps(
        self, steps: List[ReasoningStep]
    ) -> FormattedOutput:
        """Format reasoning steps as plain text."""
        lines = ["Reasoning Steps:", "=" * 40]
        
        for i, step in enumerate(steps, 1):
            lines.append(f"\nStep {i}:")
            lines.append("-" * 20)
            
            if step.thought:
                lines.append(f"Thought: {step.thought}")
            if step.action:
                lines.append(f"Action: {step.action}")
            if step.observation:
                lines.append(f"Observation: {step.observation}")
            if step.confidence is not None:
                lines.append(f"Confidence: {step.confidence:.2%}")
        
        return self._create_output(
            "\n".join(lines),
            {"step_count": len(steps)},
        )
    
    def format_agent_response(
        self,
        response: Union[AgentAction, AgentFinish],
        include_trace: bool = False,
    ) -> FormattedOutput:
        """Format agent response as plain text."""
        lines = ["Agent Response:", "=" * 40]
        
        if isinstance(response, AgentAction):
            lines.append(f"Action Type: Tool Call")
            lines.append(f"Tool: {response.tool}")
            lines.append(f"Input: {response.tool_input}")
            if include_trace and response.log:
                lines.append(f"\nTrace:\n{response.log}")
        else:
            lines.append("Action Type: Final Answer")
            if isinstance(response.return_values, dict):
                for key, value in response.return_values.items():
                    lines.append(f"{key}: {value}")
            else:
                lines.append(f"Result: {response.return_values}")
            if include_trace and response.log:
                lines.append(f"\nTrace:\n{response.log}")
        
        return self._create_output("\n".join(lines))
    
    def format_tool_result(
        self, tool_call: ToolCall, result: Any
    ) -> FormattedOutput:
        """Format tool call and result as plain text."""
        lines = [
            f"Tool: {tool_call.name}",
            f"Arguments: {tool_call.arguments}",
            f"Result: {result}",
        ]
        return self._create_output("\n".join(lines))


class MarkdownFormatter(BaseFormatter):
    """Formatter for Markdown output."""
    
    @property
    def format_type(self) -> OutputFormat:
        return OutputFormat.MARKDOWN
    
    def format(self, data: Any) -> FormattedOutput:
        """Format any data as Markdown."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = self._format_dict(data)
        elif isinstance(data, list):
            content = self._format_list(data)
        else:
            content = str(data)
        
        return self._create_output(content)
    
    def _format_dict(self, data: Dict[str, Any], level: int = 0) -> str:
        """Format a dictionary as Markdown."""
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"**{key}:**")
                lines.append(self._format_dict(value, level + 1))
            elif isinstance(value, list):
                lines.append(f"**{key}:**")
                lines.append(self._format_list(value, level + 1))
            else:
                lines.append(f"- **{key}:** {value}")
        return "\n".join(lines)
    
    def _format_list(self, data: List[Any], level: int = 0) -> str:
        """Format a list as Markdown."""
        lines = []
        prefix = "  " * level
        for item in data:
            if isinstance(item, dict):
                # Format as sub-list
                for key, value in item.items():
                    lines.append(f"{prefix}- **{key}:** {value}")
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)
    
    def format_reasoning_steps(
        self, steps: List[ReasoningStep]
    ) -> FormattedOutput:
        """Format reasoning steps as Markdown."""
        lines = ["# Reasoning Steps", ""]
        
        for i, step in enumerate(steps, 1):
            lines.append(f"## Step {i}")
            lines.append("")
            
            if step.thought:
                lines.append(f"**Thought:** {step.thought}")
                lines.append("")
            if step.action:
                lines.append(f"**Action:** `{step.action}`")
                lines.append("")
            if step.observation:
                lines.append(f"**Observation:**")
                lines.append(f"> {step.observation}")
                lines.append("")
            if step.confidence is not None:
                lines.append(f"*Confidence: {step.confidence:.2%}*")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return self._create_output(
            "\n".join(lines),
            {"step_count": len(steps)},
        )
    
    def format_agent_response(
        self,
        response: Union[AgentAction, AgentFinish],
        include_trace: bool = False,
    ) -> FormattedOutput:
        """Format agent response as Markdown."""
        lines = ["# Agent Response", ""]
        
        if isinstance(response, AgentAction):
            lines.extend([
                "## Tool Call",
                "",
                f"- **Tool:** `{response.tool}`",
                f"- **Input:** `{response.tool_input}`",
            ])
            if include_trace and response.log:
                lines.extend([
                    "",
                    "### Trace",
                    "```",
                    response.log,
                    "```",
                ])
        else:
            lines.extend([
                "## Final Answer",
                "",
            ])
            if isinstance(response.return_values, dict):
                for key, value in response.return_values.items():
                    lines.append(f"- **{key}:** {value}")
            else:
                lines.append(f"{response.return_values}")
            
            if include_trace and response.log:
                lines.extend([
                    "",
                    "### Trace",
                    "```",
                    response.log,
                    "```",
                ])
        
        return self._create_output("\n".join(lines))
    
    def format_tool_result(
        self, tool_call: ToolCall, result: Any
    ) -> FormattedOutput:
        """Format tool call and result as Markdown."""
        result_str = (
            json.dumps(result, indent=2)
            if isinstance(result, (dict, list))
            else str(result)
        )
        
        lines = [
            f"## Tool: `{tool_call.name}`",
            "",
            "**Arguments:**",
            "```json",
            json.dumps(tool_call.arguments, indent=2),
            "```",
            "",
            "**Result:**",
            "```",
            result_str,
            "```",
        ]
        return self._create_output("\n".join(lines))
    
    def format_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
    ) -> FormattedOutput:
        """Format data as a Markdown table."""
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return self._create_output(
            "\n".join(lines),
            {"rows": len(rows), "columns": len(headers)},
        )
    
    def format_code_block(
        self,
        code: str,
        language: str = "",
    ) -> FormattedOutput:
        """Format code with syntax highlighting."""
        content = f"```{language}\n{code}\n```"
        return self._create_output(content, {"language": language})


class JSONFormatter(BaseFormatter):
    """Formatter for JSON output."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        indent: int = 2,
        ensure_ascii: bool = False,
    ):
        super().__init__(config)
        self.indent = indent
        self.ensure_ascii = ensure_ascii
    
    @property
    def format_type(self) -> OutputFormat:
        return OutputFormat.JSON
    
    def format(self, data: Any) -> FormattedOutput:
        """Format any data as JSON."""
        if isinstance(data, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(data)
                content = json.dumps(
                    parsed,
                    indent=self.indent,
                    ensure_ascii=self.ensure_ascii,
                )
            except json.JSONDecodeError:
                content = json.dumps(
                    {"text": data},
                    indent=self.indent,
                    ensure_ascii=self.ensure_ascii,
                )
        elif hasattr(data, "to_dict"):
            content = json.dumps(
                data.to_dict(),
                indent=self.indent,
                ensure_ascii=self.ensure_ascii,
            )
        elif hasattr(data, "__dict__"):
            content = json.dumps(
                data.__dict__,
                indent=self.indent,
                ensure_ascii=self.ensure_ascii,
                default=str,
            )
        else:
            content = json.dumps(
                data,
                indent=self.indent,
                ensure_ascii=self.ensure_ascii,
                default=str,
            )
        
        return self._create_output(content)
    
    def format_reasoning_steps(
        self, steps: List[ReasoningStep]
    ) -> FormattedOutput:
        """Format reasoning steps as JSON."""
        data = {
            "reasoning_steps": [
                {
                    "step_number": i,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                    "confidence": step.confidence,
                    "metadata": step.metadata,
                }
                for i, step in enumerate(steps, 1)
            ],
            "total_steps": len(steps),
        }
        
        content = json.dumps(
            data,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
        )
        return self._create_output(content, {"step_count": len(steps)})
    
    def format_agent_response(
        self,
        response: Union[AgentAction, AgentFinish],
        include_trace: bool = False,
    ) -> FormattedOutput:
        """Format agent response as JSON."""
        if isinstance(response, AgentAction):
            data = {
                "type": "action",
                "tool": response.tool,
                "tool_input": response.tool_input,
            }
            if include_trace:
                data["trace"] = response.log
        else:
            data = {
                "type": "finish",
                "return_values": response.return_values,
            }
            if include_trace:
                data["trace"] = response.log
        
        content = json.dumps(
            data,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
        )
        return self._create_output(content)
    
    def format_tool_result(
        self, tool_call: ToolCall, result: Any
    ) -> FormattedOutput:
        """Format tool call and result as JSON."""
        data = {
            "tool_call": {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            "result": result,
        }
        
        content = json.dumps(
            data,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
            default=str,
        )
        return self._create_output(content)


class HTMLFormatter(BaseFormatter):
    """Formatter for HTML output."""
    
    @property
    def format_type(self) -> OutputFormat:
        return OutputFormat.HTML
    
    def format(self, data: Any) -> FormattedOutput:
        """Format any data as HTML."""
        if isinstance(data, str):
            content = f"<p>{self._escape_html(data)}</p>"
        elif isinstance(data, dict):
            content = self._format_dict_html(data)
        elif isinstance(data, list):
            content = self._format_list_html(data)
        else:
            content = f"<p>{self._escape_html(str(data))}</p>"
        
        return self._create_output(content)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
    
    def _format_dict_html(self, data: Dict[str, Any]) -> str:
        """Format a dictionary as HTML definition list."""
        items = []
        for key, value in data.items():
            if isinstance(value, dict):
                value_html = self._format_dict_html(value)
            elif isinstance(value, list):
                value_html = self._format_list_html(value)
            else:
                value_html = self._escape_html(str(value))
            items.append(f"<dt><strong>{self._escape_html(key)}</strong></dt>")
            items.append(f"<dd>{value_html}</dd>")
        return f"<dl>{''.join(items)}</dl>"
    
    def _format_list_html(self, data: List[Any]) -> str:
        """Format a list as HTML unordered list."""
        items = []
        for item in data:
            if isinstance(item, dict):
                item_html = self._format_dict_html(item)
            elif isinstance(item, list):
                item_html = self._format_list_html(item)
            else:
                item_html = self._escape_html(str(item))
            items.append(f"<li>{item_html}</li>")
        return f"<ul>{''.join(items)}</ul>"
    
    def format_reasoning_steps(
        self, steps: List[ReasoningStep]
    ) -> FormattedOutput:
        """Format reasoning steps as HTML."""
        html_parts = ['<div class="reasoning-steps">']
        html_parts.append("<h2>Reasoning Steps</h2>")
        
        for i, step in enumerate(steps, 1):
            html_parts.append(f'<div class="step" id="step-{i}">')
            html_parts.append(f"<h3>Step {i}</h3>")
            
            if step.thought:
                html_parts.append(
                    f'<p class="thought"><strong>Thought:</strong> '
                    f'{self._escape_html(step.thought)}</p>'
                )
            if step.action:
                html_parts.append(
                    f'<p class="action"><strong>Action:</strong> '
                    f'<code>{self._escape_html(step.action)}</code></p>'
                )
            if step.observation:
                html_parts.append(
                    f'<blockquote class="observation">'
                    f'{self._escape_html(step.observation)}</blockquote>'
                )
            if step.confidence is not None:
                html_parts.append(
                    f'<p class="confidence"><em>Confidence: '
                    f'{step.confidence:.2%}</em></p>'
                )
            
            html_parts.append("</div>")
        
        html_parts.append("</div>")
        
        return self._create_output(
            "\n".join(html_parts),
            {"step_count": len(steps)},
        )
    
    def format_agent_response(
        self,
        response: Union[AgentAction, AgentFinish],
        include_trace: bool = False,
    ) -> FormattedOutput:
        """Format agent response as HTML."""
        html_parts = ['<div class="agent-response">']
        html_parts.append("<h2>Agent Response</h2>")
        
        if isinstance(response, AgentAction):
            html_parts.extend([
                '<div class="tool-call">',
                "<h3>Tool Call</h3>",
                f"<p><strong>Tool:</strong> <code>{self._escape_html(response.tool)}</code></p>",
                f"<p><strong>Input:</strong> <code>{self._escape_html(str(response.tool_input))}</code></p>",
            ])
            if include_trace and response.log:
                html_parts.extend([
                    "<details>",
                    "<summary>Trace</summary>",
                    f"<pre>{self._escape_html(response.log)}</pre>",
                    "</details>",
                ])
            html_parts.append("</div>")
        else:
            html_parts.extend([
                '<div class="final-answer">',
                "<h3>Final Answer</h3>",
            ])
            if isinstance(response.return_values, dict):
                html_parts.append(self._format_dict_html(response.return_values))
            else:
                html_parts.append(
                    f"<p>{self._escape_html(str(response.return_values))}</p>"
                )
            if include_trace and response.log:
                html_parts.extend([
                    "<details>",
                    "<summary>Trace</summary>",
                    f"<pre>{self._escape_html(response.log)}</pre>",
                    "</details>",
                ])
            html_parts.append("</div>")
        
        html_parts.append("</div>")
        
        return self._create_output("\n".join(html_parts))
    
    def format_tool_result(
        self, tool_call: ToolCall, result: Any
    ) -> FormattedOutput:
        """Format tool call and result as HTML."""
        result_html = (
            f"<pre>{self._escape_html(json.dumps(result, indent=2))}</pre>"
            if isinstance(result, (dict, list))
            else f"<p>{self._escape_html(str(result))}</p>"
        )
        
        html_parts = [
            '<div class="tool-result">',
            f"<h3>Tool: <code>{self._escape_html(tool_call.name)}</code></h3>",
            "<h4>Arguments:</h4>",
            f"<pre>{self._escape_html(json.dumps(tool_call.arguments, indent=2))}</pre>",
            "<h4>Result:</h4>",
            result_html,
            "</div>",
        ]
        
        return self._create_output("\n".join(html_parts))


class StructuredFormatter(BaseFormatter):
    """Formatter that preserves structured data while adding metadata."""
    
    @property
    def format_type(self) -> OutputFormat:
        return OutputFormat.STRUCTURED
    
    def format(self, data: Any) -> FormattedOutput:
        """Wrap data in structured format with metadata."""
        if hasattr(data, "to_dict"):
            content = json.dumps(data.to_dict(), default=str)
        elif hasattr(data, "__dict__"):
            content = json.dumps(data.__dict__, default=str)
        else:
            content = json.dumps(data, default=str)
        
        return self._create_output(
            content,
            {"type": type(data).__name__},
        )
    
    def format_reasoning_steps(
        self, steps: List[ReasoningStep]
    ) -> FormattedOutput:
        """Format reasoning steps preserving structure."""
        data = {
            "steps": [
                {
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                    "confidence": step.confidence,
                    "metadata": step.metadata,
                }
                for step in steps
            ],
        }
        return self._create_output(
            json.dumps(data, default=str),
            {"step_count": len(steps), "type": "reasoning_chain"},
        )
    
    def format_agent_response(
        self,
        response: Union[AgentAction, AgentFinish],
        include_trace: bool = False,
    ) -> FormattedOutput:
        """Format agent response preserving structure."""
        if isinstance(response, AgentAction):
            data = {
                "action_type": "tool_call",
                "tool": response.tool,
                "tool_input": response.tool_input,
                "log": response.log if include_trace else None,
            }
        else:
            data = {
                "action_type": "finish",
                "return_values": response.return_values,
                "log": response.log if include_trace else None,
            }
        
        return self._create_output(
            json.dumps(data, default=str),
            {"type": "agent_response"},
        )
    
    def format_tool_result(
        self, tool_call: ToolCall, result: Any
    ) -> FormattedOutput:
        """Format tool result preserving structure."""
        data = {
            "tool_call": {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            "result": result,
        }
        return self._create_output(
            json.dumps(data, default=str),
            {"type": "tool_result"},
        )


class OutputFormatterFactory:
    """Factory for creating output formatters."""
    
    _formatters: Dict[OutputFormat, type] = {
        OutputFormat.PLAIN: PlainTextFormatter,
        OutputFormat.MARKDOWN: MarkdownFormatter,
        OutputFormat.JSON: JSONFormatter,
        OutputFormat.HTML: HTMLFormatter,
        OutputFormat.STRUCTURED: StructuredFormatter,
    }
    
    @classmethod
    def create(
        cls,
        format_type: Union[OutputFormat, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseFormatter:
        """Create a formatter for the specified format type."""
        if isinstance(format_type, str):
            format_type = OutputFormat(format_type.lower())
        
        formatter_class = cls._formatters.get(format_type)
        if formatter_class is None:
            raise ValueError(f"Unknown format type: {format_type}")
        
        return formatter_class(config=config)
    
    @classmethod
    def register(
        cls,
        format_type: OutputFormat,
        formatter_class: type,
    ) -> None:
        """Register a custom formatter for a format type."""
        if not issubclass(formatter_class, BaseFormatter):
            raise TypeError(
                f"Formatter class must inherit from BaseFormatter"
            )
        cls._formatters[format_type] = formatter_class
    
    @classmethod
    def available_formats(cls) -> List[str]:
        """Return list of available format types."""
        return [fmt.value for fmt in cls._formatters.keys()]


# Convenience functions
def format_output(
    data: Any,
    format_type: Union[OutputFormat, str] = OutputFormat.MARKDOWN,
    config: Optional[Dict[str, Any]] = None,
) -> FormattedOutput:
    """Format data using the specified formatter."""
    formatter = OutputFormatterFactory.create(format_type, config)
    return formatter.format(data)


def format_reasoning(
    steps: List[ReasoningStep],
    format_type: Union[OutputFormat, str] = OutputFormat.MARKDOWN,
) -> FormattedOutput:
    """Format reasoning steps using the specified formatter."""
    formatter = OutputFormatterFactory.create(format_type)
    return formatter.format_reasoning_steps(steps)


def format_response(
    response: Union[AgentAction, AgentFinish],
    format_type: Union[OutputFormat, str] = OutputFormat.MARKDOWN,
    include_trace: bool = False,
) -> FormattedOutput:
    """Format agent response using the specified formatter."""
    formatter = OutputFormatterFactory.create(format_type)
    return formatter.format_agent_response(response, include_trace)


# Export public API
__all__ = [
    "OutputFormat",
    "FormattedOutput",
    "BaseFormatter",
    "PlainTextFormatter",
    "MarkdownFormatter",
    "JSONFormatter",
    "HTMLFormatter",
    "StructuredFormatter",
    "OutputFormatterFactory",
    "format_output",
    "format_reasoning",
    "format_response",
]
