"""
Unit Tests for CogFlow Tool System

Tests for BaseTool, ToolRegistry, and built-in tools
"""

import pytest
import asyncio
from typing import Dict, Any

from cogflow.base.tool import BaseTool, ToolResult
from cogflow.modules.tool_registry import ToolRegistry, FunctionTool
from cogflow.tools.builtin import (
    CalculatorTool,
    TextProcessingTool,
    DateTimeTool,
    JSONTool,
)


class TestBaseTool:
    """Tests for BaseTool abstract base class."""
    
    def test_cannot_instantiate_abstract(self):
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool()
    
    def test_concrete_tool_implementation(self):
        """Test that concrete tool can be created."""
        class SimpleTool(BaseTool):
            @property
            def name(self) -> str:
                return "simple"
            
            @property
            def description(self) -> str:
                return "A simple tool"
            
            @property
            def schema(self) -> Dict[str, Any]:
                return {"type": "object", "properties": {}}
            
            async def execute(self, **kwargs) -> Dict[str, Any]:
                return {"result": "success"}
        
        tool = SimpleTool()
        assert tool.name == "simple"
        assert tool.description == "A simple tool"


class TestFunctionTool:
    """Tests for FunctionTool wrapper."""
    
    def test_create_function_tool(self):
        """Test creating a FunctionTool from a function."""
        def my_func(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y
        
        tool = FunctionTool(
            func=my_func,
            name="add",
            description="Add two numbers",
        )
        assert tool.name == "add"
        assert tool.description == "Add two numbers"
    
    @pytest.mark.asyncio
    async def test_function_tool_execution(self):
        """Test executing a FunctionTool."""
        def multiply(a: int, b: int) -> int:
            return a * b
        
        tool = FunctionTool(
            func=multiply,
            name="multiply",
            description="Multiply two numbers",
        )
        result = await tool.execute(a=3, b=4)
        assert result == 12
    
    @pytest.mark.asyncio
    async def test_async_function_tool(self):
        """Test FunctionTool with async function."""
        async def async_greet(name: str) -> str:
            await asyncio.sleep(0.01)
            return f"Hello, {name}!"
        
        tool = FunctionTool(
            func=async_greet,
            name="greet",
            description="Greet someone",
        )
        result = await tool.execute(name="World")
        assert result == "Hello, World!"


class TestRegisterToolDecorator:
    """Tests for registry.register decorator."""
    
    def test_register_simple_function(self):
        """Test registering a simple function as a tool."""
        registry = ToolRegistry()
        
        @registry.register(
            name="subtract",
            description="Subtract two numbers",
        )
        def subtract(a: int, b: int) -> int:
            return a - b
        
        # list_tools returns ToolMetadata objects
        tool_names = [meta.name for meta in registry.list_tools()]
        assert "subtract" in tool_names
    
    def test_registered_function_still_callable(self):
        """Test that decorated function remains callable."""
        registry = ToolRegistry()
        
        @registry.register(name="double", description="Double a number")
        def double(x: int) -> int:
            return x * 2
        
        result = double(5)
        assert result == 10


class TestToolRegistry:
    """Tests for ToolRegistry class."""
    
    def setup_method(self):
        """Set up fresh registry for each test."""
        self.registry = ToolRegistry()
    
    def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        tool = FunctionTool(
            func=lambda x: x,
            name="test_tool",
            description="A test tool",
        )
        self.registry.add_tool(tool)
        
        retrieved = self.registry.get_tool("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"
    
    def test_list_tools(self):
        """Test listing all registered tools."""
        for i in range(3):
            tool = FunctionTool(
                func=lambda: None,
                name=f"tool_{i}",
                description=f"Tool {i}",
            )
            self.registry.add_tool(tool)
        
        tools = self.registry.list_tools()
        assert len(tools) == 3
    
    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        result = self.registry.get_tool("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool through the registry."""
        def concat(a: str, b: str) -> str:
            return a + b
        
        tool = FunctionTool(
            func=concat,
            name="concat",
            description="Concatenate strings",
        )
        self.registry.add_tool(tool)
        
        result = await self.registry.execute("concat", a="Hello", b="World")
        assert result == "HelloWorld"


class TestCalculatorTool:
    """Tests for CalculatorTool."""
    
    def setup_method(self):
        """Set up calculator tool."""
        self.calc = CalculatorTool()
    
    def test_calculator_properties(self):
        """Test calculator tool properties."""
        assert self.calc.name == "calculator"
        assert "math" in self.calc.description.lower() or "calculator" in self.calc.description.lower()
    
    @pytest.mark.asyncio
    async def test_simple_addition(self):
        """Test simple addition."""
        result = await self.calc.execute(expression="2 + 2")
        assert result.get("success") is True
        assert result.get("result") == 4
    
    @pytest.mark.asyncio
    async def test_complex_expression(self):
        """Test more complex expression."""
        result = await self.calc.execute(expression="(10 + 5) * 2")
        assert result.get("success") is True
        assert result.get("result") == 30
    
    @pytest.mark.asyncio
    async def test_division(self):
        """Test division."""
        result = await self.calc.execute(expression="100 / 4")
        assert result.get("success") is True
        assert result.get("result") == 25.0


class TestTextProcessingTool:
    """Tests for TextProcessingTool."""
    
    def setup_method(self):
        """Set up text processing tool."""
        self.processor = TextProcessingTool()
    
    def test_processor_properties(self):
        """Test text processor properties."""
        assert self.processor.name == "text_processor"
    
    @pytest.mark.asyncio
    async def test_count_words(self):
        """Test word counting."""
        result = await self.processor.execute(
            text="Hello world this is a test",
            operation="word_count",
        )
        assert result.get("success") is True
        assert result.get("count") == 6
    
    @pytest.mark.asyncio
    async def test_uppercase(self):
        """Test uppercase conversion."""
        result = await self.processor.execute(
            text="hello world",
            operation="uppercase",
        )
        assert result.get("success") is True
        assert result.get("text") == "HELLO WORLD"
    
    @pytest.mark.asyncio
    async def test_lowercase(self):
        """Test lowercase conversion."""
        result = await self.processor.execute(
            text="HELLO WORLD",
            operation="lowercase",
        )
        assert result.get("success") is True
        assert result.get("text") == "hello world"


class TestDateTimeTool:
    """Tests for DateTimeTool."""
    
    def setup_method(self):
        """Set up datetime tool."""
        self.dt_tool = DateTimeTool()
    
    def test_datetime_properties(self):
        """Test datetime tool properties."""
        assert self.dt_tool.name == "datetime"
    
    @pytest.mark.asyncio
    async def test_get_current_time(self):
        """Test getting current time."""
        result = await self.dt_tool.execute(operation="now")
        assert result is not None
        # Result should contain success flag
        assert result.get("success") is True


class TestJSONTool:
    """Tests for JSONTool."""
    
    def setup_method(self):
        """Set up JSON tool."""
        self.json_tool = JSONTool()
    
    def test_json_properties(self):
        """Test JSON tool properties."""
        assert self.json_tool.name == "json_processor"
    
    @pytest.mark.asyncio
    async def test_parse_json(self):
        """Test JSON parsing."""
        result = await self.json_tool.execute(
            data='{"name": "test", "value": 42}',
            operation="parse",
        )
        assert result is not None
        assert result.get("success") is True
    
    @pytest.mark.asyncio
    async def test_stringify_json(self):
        """Test JSON formatting (pretty print)."""
        result = await self.json_tool.execute(
            data='{"name": "test", "value": 42}',
            operation="format",
        )
        assert result is not None
        assert result.get("success") is True
        assert "formatted" in result
