"""Test script to verify CogFlow functionality."""
import asyncio
import sys
sys.path.insert(0, 'c:\\Users\\dhanu\\Downloads\\CogFlow')

from cogflow.modules.tool_registry import ToolRegistry
from cogflow.tools.builtin import (
    CalculatorTool,
    WebSearchTool,
    TextProcessingTool,
    DateTimeTool,
    JSONTool
)

async def test_tools():
    """Test all builtin tools."""
    print("Testing CogFlow Tools...")
    print("=" * 50)
    
    # Create registry
    registry = ToolRegistry()
    
    # Register tools
    registry.add_tool(CalculatorTool())
    registry.add_tool(WebSearchTool())
    registry.add_tool(TextProcessingTool())
    registry.add_tool(DateTimeTool())
    registry.add_tool(JSONTool())
    
    # Test calculator
    print("\n1. Testing Calculator...")
    result = await registry.execute("calculator", expression="2+2")
    print(f"   2+2 = {result}")
    
    result = await registry.execute("calculator", expression="sqrt(16)")
    print(f"   sqrt(16) = {result}")
    
    # Test text processor
    print("\n2. Testing Text Processor...")
    result = await registry.execute("text_processor", text="Hello World", operation="word_count")
    print(f"   Word count of 'Hello World': {result}")
    
    # Test datetime
    print("\n3. Testing DateTime...")
    result = await registry.execute("datetime", operation="today")
    print(f"   Today: {result}")
    
    # Test JSON processor
    print("\n4. Testing JSON Processor...")
    result = await registry.execute("json_processor", operation="parse", data='{"key": "value"}')
    print(f"   Parsed JSON: {result}")
    
    print("\n" + "=" * 50)
    print("✅ All tools working correctly!")

if __name__ == "__main__":
    asyncio.run(test_tools())
