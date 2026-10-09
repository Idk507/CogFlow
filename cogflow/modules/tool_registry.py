"""
CogFlow Tool Registry

Provides tool registration, discovery, and execution functionality.
Supports decorator-based registration and schema validation.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from cogflow.base.tool import BaseTool
from cogflow.base.types import ToolCall


class ToolMetadata(BaseModel):
    """Metadata for a registered tool."""
    
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for parameters"
    )
    is_async: bool = Field(default=False, description="Whether tool is async")
    tags: List[str] = Field(default_factory=list, description="Tool tags for categorization")


class FunctionTool(BaseTool):
    """Wrapper to convert a function into a BaseTool."""
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameter_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Initialize function tool.
        
        Args:
            func: The function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            parameter_schema: JSON schema for parameters
            tags: Optional tags for categorization
        """
        self._func = func
        self._name = name or func.__name__
        self._description = description or (func.__doc__ or "No description available").strip()
        self._parameter_schema = parameter_schema or self._infer_schema(func)
        self._tags = tags or []
        self._is_async = inspect.iscoroutinefunction(func)
    
    def _infer_schema(self, func: Callable) -> Dict[str, Any]:
        """Infer JSON schema from function signature."""
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            
            # Get type annotation
            if param.annotation != inspect.Parameter.empty:
                param_type = self._python_type_to_json(param.annotation)
            else:
                param_type = {"type": "string"}
            
            properties[param_name] = param_type
            
            # Check if required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    
    def _python_type_to_json(self, python_type: type) -> Dict[str, Any]:
        """Convert Python type to JSON schema type."""
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }
        
        # Handle Optional and Union types
        origin = getattr(python_type, "__origin__", None)
        if origin is Union:
            args = python_type.__args__
            # Handle Optional[X] (Union[X, None])
            non_none_types = [a for a in args if a is not type(None)]
            if len(non_none_types) == 1:
                return self._python_type_to_json(non_none_types[0])
        
        return type_map.get(python_type, {"type": "string"})
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def schema(self) -> Dict[str, Any]:
        return self._parameter_schema
    
    @property
    def tags(self) -> List[str]:
        return self._tags
    
    @property
    def is_async(self) -> bool:
        return self._is_async
    
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped function."""
        if self._is_async:
            return await self._func(**kwargs)
        else:
            return self._func(**kwargs)
    
    def execute_sync(self, **kwargs: Any) -> Any:
        """Execute synchronously (for non-async functions only)."""
        if self._is_async:
            import asyncio
            return asyncio.get_event_loop().run_until_complete(self._func(**kwargs))
        return self._func(**kwargs)


class ToolRegistry:
    """Registry for managing tools.
    
    Provides:
    - Decorator-based registration
    - Tool discovery and listing
    - Schema validation
    - Tool execution
    
    Example:
        ```python
        registry = ToolRegistry()
        
        @registry.register(description="Calculate sum")
        def add(a: int, b: int) -> int:
            return a + b
        
        result = await registry.execute("add", a=1, b=2)
        ```
    """
    
    _global_instance: Optional["ToolRegistry"] = None
    
    def __init__(self, name: str = "default"):
        """Initialize tool registry.
        
        Args:
            name: Registry name for identification
        """
        self.name = name
        self._tools: Dict[str, BaseTool] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
    
    @classmethod
    def get_global(cls) -> "ToolRegistry":
        """Get or create the global tool registry."""
        if cls._global_instance is None:
            cls._global_instance = cls("global")
        return cls._global_instance
    
    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameter_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Callable:
        """Decorator to register a function as a tool.
        
        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            parameter_schema: JSON schema for parameters
            tags: Optional tags for categorization
        
        Returns:
            Decorator function
        
        Example:
            ```python
            @registry.register(description="Search the web")
            def search(query: str) -> str:
                ...
            ```
        """
        def decorator(func: Callable) -> Callable:
            tool = FunctionTool(
                func=func,
                name=name,
                description=description,
                parameter_schema=parameter_schema,
                tags=tags,
            )
            self.add_tool(tool)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the registry.
        
        Args:
            tool: Tool instance to register
        
        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        
        self._tools[tool.name] = tool
        
        # Create metadata
        is_async = getattr(tool, "is_async", False)
        tags = getattr(tool, "tags", [])
        
        self._metadata[tool.name] = ToolMetadata(
            name=tool.name,
            description=tool.description,
            parameters=tool.schema,
            is_async=is_async,
            tags=tags,
        )
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the registry.
        
        Args:
            name: Tool name to remove
        
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            del self._metadata[name]
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.
        
        Args:
            name: Tool name
        
        Returns:
            True if tool exists
        """
        return name in self._tools
    
    def list_tools(self, tags: Optional[List[str]] = None) -> List[ToolMetadata]:
        """List all registered tools.
        
        Args:
            tags: Optional filter by tags
        
        Returns:
            List of tool metadata
        """
        if tags is None:
            return list(self._metadata.values())
        
        return [
            meta for meta in self._metadata.values()
            if any(tag in meta.tags for tag in tags)
        ]
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (for LLM function calling).
        
        Returns:
            List of tool schemas in OpenAI function format
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                }
            })
        return schemas
    
    async def execute(
        self,
        name: str,
        validate: bool = True,
        **kwargs: Any
    ) -> Any:
        """Execute a tool by name.
        
        Args:
            name: Tool name
            validate: Whether to validate input against schema
            **kwargs: Tool arguments
        
        Returns:
            Tool execution result
        
        Raises:
            ValueError: If tool not found
            ValidationError: If validation fails
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        
        if validate:
            tool.validate_input(**kwargs)
        
        return await tool.execute(**kwargs)
    
    async def execute_tool_call(
        self,
        tool_call: ToolCall,
        validate: bool = True,
    ) -> Any:
        """Execute a ToolCall object.
        
        Args:
            tool_call: ToolCall instance with name and arguments
            validate: Whether to validate input
        
        Returns:
            Tool execution result
        """
        return await self.execute(
            name=tool_call.name,
            validate=validate,
            **tool_call.arguments,
        )
    
    def execute_sync(
        self,
        name: str,
        validate: bool = True,
        **kwargs: Any
    ) -> Any:
        """Execute a tool synchronously.
        
        Args:
            name: Tool name
            validate: Whether to validate input
            **kwargs: Tool arguments
        
        Returns:
            Tool execution result
        """
        import asyncio
        try:
            # Try to get running loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create new one
            return asyncio.run(self.execute(name, validate=validate, **kwargs))
        else:
            # There's already a running loop, use run_until_complete
            return loop.run_until_complete(
                self.execute(name, validate=validate, **kwargs)
            )
    
    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()
        self._metadata.clear()
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools
    
    def __repr__(self) -> str:
        return f"ToolRegistry(name='{self.name}', tools={list(self._tools.keys())})"


# Convenience decorator for global registry
def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameter_schema: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Callable:
    """Register a function as a tool in the global registry.
    
    Example:
        ```python
        @register_tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b
        ```
    """
    return ToolRegistry.get_global().register(
        name=name,
        description=description,
        parameter_schema=parameter_schema,
        tags=tags,
    )
