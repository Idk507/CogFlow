"""
CogFlow Built-in Tools

Provides a collection of commonly used tools that can be registered
with the tool registry for agent use.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Union

from cogflow.base.tool import BaseTool


class CalculatorTool(BaseTool):
    """Calculator tool for mathematical operations.
    
    Supports basic arithmetic, scientific functions, and expression evaluation.
    """
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return (
            "A calculator for mathematical operations. "
            "Supports basic arithmetic (+, -, *, /, ^), "
            "functions (sqrt, sin, cos, tan, log, exp), "
            "and constants (pi, e)."
        )
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)')"
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, expression: str) -> Dict[str, Any]:
        """Evaluate a mathematical expression.
        
        Args:
            expression: Math expression string
        
        Returns:
            Dict with result or error
        """
        try:
            # Clean and prepare expression
            expr = expression.strip()
            
            # Replace common math functions and constants
            replacements = {
                "^": "**",
                "sqrt": "math.sqrt",
                "sin": "math.sin",
                "cos": "math.cos",
                "tan": "math.tan",
                "log": "math.log",
                "log10": "math.log10",
                "exp": "math.exp",
                "abs": "abs",
                "pi": "math.pi",
                "e": "math.e",
            }
            
            for old, new in replacements.items():
                expr = expr.replace(old, new)
            
            # Validate expression (only allow safe characters)
            allowed = set("0123456789+-*/.() ,mathlsqrincogeabspx")
            if not all(c in allowed for c in expr.lower()):
                return {
                    "success": False,
                    "error": "Expression contains invalid characters",
                    "expression": expression,
                }
            
            # Evaluate expression
            result = eval(expr, {"__builtins__": {}, "math": math, "abs": abs})
            
            return {
                "success": True,
                "result": result,
                "expression": expression,
            }
        except ZeroDivisionError:
            return {
                "success": False,
                "error": "Division by zero",
                "expression": expression,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expression": expression,
            }


class WebSearchTool(BaseTool):
    """Web search tool (placeholder implementation).
    
    In production, this would integrate with a search API like
    Google, Bing, or DuckDuckGo.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize search tool.
        
        Args:
            api_key: Optional API key for search service
        """
        self._api_key = api_key
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return (
            "Search the web for information. "
            "Returns relevant search results for a query."
        )
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    async def execute(
        self,
        query: str,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """Perform a web search.
        
        Args:
            query: Search query string
            num_results: Number of results to return
        
        Returns:
            Dict with search results or error
        """
        # Placeholder implementation
        # In production, integrate with actual search API
        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "title": f"Result {i+1} for: {query}",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a placeholder result for the query '{query}'.",
                }
                for i in range(min(num_results, 5))
            ],
            "note": "This is a placeholder. Integrate with a real search API for production use."
        }


class TextProcessingTool(BaseTool):
    """Text processing utilities."""
    
    @property
    def name(self) -> str:
        return "text_processor"
    
    @property
    def description(self) -> str:
        return (
            "Process and analyze text. Operations include: "
            "word_count, char_count, summarize, extract_keywords, uppercase, lowercase"
        )
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to process"
                },
                "operation": {
                    "type": "string",
                    "enum": ["word_count", "char_count", "uppercase", "lowercase", "reverse", "extract_numbers"],
                    "description": "Operation to perform"
                }
            },
            "required": ["text", "operation"]
        }
    
    async def execute(
        self,
        text: str,
        operation: str
    ) -> Dict[str, Any]:
        """Process text with specified operation.
        
        Args:
            text: Input text
            operation: Operation to perform
        
        Returns:
            Dict with operation result
        """
        operations = {
            "word_count": lambda t: {"count": len(t.split())},
            "char_count": lambda t: {"count": len(t)},
            "uppercase": lambda t: {"text": t.upper()},
            "lowercase": lambda t: {"text": t.lower()},
            "reverse": lambda t: {"text": t[::-1]},
            "extract_numbers": lambda t: {"numbers": re.findall(r'-?\d+\.?\d*', t)},
        }
        
        if operation not in operations:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}",
                "available_operations": list(operations.keys()),
            }
        
        try:
            result = operations[operation](text)
            return {
                "success": True,
                "operation": operation,
                **result,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


class DateTimeTool(BaseTool):
    """Date and time utilities."""
    
    @property
    def name(self) -> str:
        return "datetime"
    
    @property
    def description(self) -> str:
        return (
            "Get current date/time or perform date calculations. "
            "Operations: now, today, add_days, diff_days, format"
        )
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["now", "today", "add_days", "format"],
                    "description": "Operation to perform"
                },
                "date": {
                    "type": "string",
                    "description": "Date string (ISO format) for operations that need it"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days for add_days operation"
                },
                "format": {
                    "type": "string",
                    "description": "Output format (e.g., '%Y-%m-%d', '%B %d, %Y')"
                }
            },
            "required": ["operation"]
        }
    
    async def execute(
        self,
        operation: str,
        date: Optional[str] = None,
        days: Optional[int] = None,
        format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform date/time operation.
        
        Args:
            operation: Operation to perform
            date: Optional date string
            days: Optional days for calculations
            format: Optional output format
        
        Returns:
            Dict with result
        """
        from datetime import datetime, timedelta
        
        try:
            if operation == "now":
                now = datetime.now()
                return {
                    "success": True,
                    "datetime": now.isoformat(),
                    "formatted": now.strftime(format or "%Y-%m-%d %H:%M:%S"),
                }
            
            elif operation == "today":
                today = datetime.now().date()
                return {
                    "success": True,
                    "date": today.isoformat(),
                }
            
            elif operation == "add_days":
                if date is None:
                    base = datetime.now()
                else:
                    base = datetime.fromisoformat(date)
                
                result = base + timedelta(days=days or 0)
                return {
                    "success": True,
                    "result": result.isoformat(),
                    "original": base.isoformat(),
                    "days_added": days or 0,
                }
            
            elif operation == "format":
                if date is None:
                    dt = datetime.now()
                else:
                    dt = datetime.fromisoformat(date)
                
                return {
                    "success": True,
                    "formatted": dt.strftime(format or "%Y-%m-%d"),
                    "original": dt.isoformat(),
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


class JSONTool(BaseTool):
    """JSON parsing and manipulation tool."""
    
    @property
    def name(self) -> str:
        return "json_processor"
    
    @property
    def description(self) -> str:
        return (
            "Parse, validate, and manipulate JSON data. "
            "Operations: parse, validate, get_value, format"
        )
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["parse", "validate", "get_value", "format"],
                    "description": "Operation to perform"
                },
                "data": {
                    "type": "string",
                    "description": "JSON string to process"
                },
                "path": {
                    "type": "string",
                    "description": "JSON path for get_value (e.g., 'user.name', 'items[0]')"
                }
            },
            "required": ["operation", "data"]
        }
    
    async def execute(
        self,
        operation: str,
        data: str,
        path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process JSON data.
        
        Args:
            operation: Operation to perform
            data: JSON string
            path: Optional path for value extraction
        
        Returns:
            Dict with result
        """
        import json
        
        try:
            if operation == "validate":
                try:
                    json.loads(data)
                    return {"success": True, "valid": True}
                except json.JSONDecodeError as e:
                    return {
                        "success": True,
                        "valid": False,
                        "error": str(e),
                    }
            
            elif operation == "parse":
                parsed = json.loads(data)
                return {
                    "success": True,
                    "parsed": parsed,
                    "type": type(parsed).__name__,
                }
            
            elif operation == "format":
                parsed = json.loads(data)
                formatted = json.dumps(parsed, indent=2)
                return {
                    "success": True,
                    "formatted": formatted,
                }
            
            elif operation == "get_value":
                parsed = json.loads(data)
                
                if path is None:
                    return {
                        "success": False,
                        "error": "Path required for get_value operation",
                    }
                
                # Navigate path
                value = parsed
                for key in path.replace("[", ".").replace("]", "").split("."):
                    if key.isdigit():
                        value = value[int(key)]
                    else:
                        value = value[key]
                
                return {
                    "success": True,
                    "path": path,
                    "value": value,
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


def register_builtin_tools(registry: "ToolRegistry") -> None:
    """Register all built-in tools with a registry.
    
    Args:
        registry: ToolRegistry instance to register tools with
    """
    from cogflow.modules.tool_registry import ToolRegistry
    
    tools = [
        CalculatorTool(),
        WebSearchTool(),
        TextProcessingTool(),
        DateTimeTool(),
        JSONTool(),
    ]
    
    for tool in tools:
        try:
            registry.add_tool(tool)
        except ValueError:
            # Tool already registered, skip
            pass


# Export all built-in tools
__all__ = [
    "CalculatorTool",
    "WebSearchTool",
    "TextProcessingTool",
    "DateTimeTool",
    "JSONTool",
    "register_builtin_tools",
]
