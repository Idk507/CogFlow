"""
CogFlow Tool Interface

This module defines the abstract base class for all tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable
from pydantic import BaseModel, Field
import json


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    
    name: str = Field(description="Parameter name")
    type: str = Field(default="string", description="Parameter type")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether required")
    default: Optional[Any] = Field(default=None, description="Default value")


class ToolSchema(BaseModel):
    """JSON Schema definition for a tool."""
    
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for parameters"
    )
    
    @classmethod
    def from_parameters(
        cls,
        name: str,
        description: str,
        params: List[ToolParameter]
    ) -> "ToolSchema":
        """Create schema from parameter list."""
        properties = {}
        required = []
        
        for param in params:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)
        
        return cls(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required
            }
        )
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }


class ToolResult(BaseModel):
    """Result from a tool execution."""
    
    success: bool = Field(description="Whether execution succeeded")
    output: str = Field(description="Tool output as string")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @classmethod
    def success_result(cls, output: Any) -> "ToolResult":
        """Create a successful result."""
        if not isinstance(output, str):
            output = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
        return cls(success=True, output=output)
    
    @classmethod
    def error_result(cls, error: str) -> "ToolResult":
        """Create an error result."""
        return cls(success=False, output="", error=error)


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools are callable components that agents can use to interact
    with external systems or perform computations.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this tool does."""
        pass
    
    @property
    def parameters(self) -> List[ToolParameter]:
        """Return the list of parameters this tool accepts."""
        return []
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with given inputs.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult with output or error
        """
        pass
    
    def get_schema(self) -> ToolSchema:
        """Get the JSON schema for this tool."""
        return ToolSchema.from_parameters(
            name=self.name,
            description=self.description,
            params=self.parameters
        )
    
    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Make the tool callable."""
        return await self.execute(**kwargs)
    
    def validate_input(self, **kwargs: Any) -> bool:
        """
        Validate input arguments.
        
        Args:
            **kwargs: Arguments to validate
            
        Returns:
            True if valid
        """
        required = [p.name for p in self.parameters if p.required]
        for name in required:
            if name not in kwargs:
                return False
        return True
    
    def __repr__(self) -> str:
        return f"Tool({self.name})"


class FunctionTool(BaseTool):
    """
    A tool created from a function.
    
    Allows wrapping any async function as a tool.
    """
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        params: Optional[List[ToolParameter]] = None
    ):
        """
        Create a tool from a function.
        
        Args:
            func: The function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            params: Parameter definitions
        """
        self._func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or ""
        self._parameters = params or []
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return self._parameters
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function."""
        try:
            import asyncio
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(**kwargs)
            else:
                result = self._func(**kwargs)
            return ToolResult.success_result(result)
        except Exception as e:
            return ToolResult.error_result(str(e))


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    params: Optional[List[ToolParameter]] = None
) -> Callable:
    """
    Decorator to create a tool from a function.
    
    Usage:
        @tool(name="my_tool", description="Does something")
        async def my_tool(x: int, y: int) -> str:
            return str(x + y)
    """
    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(
            func=func,
            name=name,
            description=description,
            params=params
        )
    return decorator


class ToolError(Exception):
    """Base exception for tool errors."""
    
    def __init__(self, tool_name: str, message: str):
        super().__init__(f"Tool '{tool_name}': {message}")
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Raised when a tool is not found in the registry."""
    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""
    pass
