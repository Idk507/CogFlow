#!/usr/bin/env python3
"""
CogFlow CLI - Command Line Interface for the CogFlow Framework.

Provides commands for running agents, pipelines, and tools from the command line.
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogflow import __version__
from cogflow.config import (
    CogFlowConfig,
    LLMConfig,
    ReasonerConfig,
    AgentConfig,
)
from cogflow.modules.tool_registry import ToolRegistry
from cogflow.modules.agent import ReActAgent, SimpleAgent
from cogflow.modules.output_formatter import (
    OutputFormatterFactory,
    OutputFormat,
)
from cogflow.tools.builtin import (
    CalculatorTool,
    WebSearchTool,
    TextProcessingTool,
    DateTimeTool,
    JSONTool,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("cogflow.cli")


def setup_default_tools() -> ToolRegistry:
    """Set up the default tool registry with built-in tools."""
    registry = ToolRegistry()
    
    # Register built-in tools
    tools = [
        CalculatorTool(),
        WebSearchTool(),
        TextProcessingTool(),
        DateTimeTool(),
        JSONTool(),
    ]
    
    for tool in tools:
        registry.add_tool(tool)
    
    return registry


def load_config(config_path: Optional[str] = None) -> CogFlowConfig:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)
        return CogFlowConfig(**config_data)
    return CogFlowConfig()


def cmd_version(args: argparse.Namespace) -> int:
    """Handle the version command."""
    print(f"CogFlow version {__version__}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """Handle the tools command - list available tools."""
    registry = setup_default_tools()
    
    formatter = OutputFormatterFactory.create(
        getattr(args, "format", "plain") or "plain"
    )
    
    tools_data = []
    for tool_meta in registry.list_tools():
        # Extract parameter names from the schema
        param_names = []
        if isinstance(tool_meta.parameters, dict):
            properties = tool_meta.parameters.get("properties", {})
            param_names = list(properties.keys())
        
        tools_data.append({
            "name": tool_meta.name,
            "description": tool_meta.description,
            "parameters": param_names,
        })
    
    if args.format == "json":
        print(json.dumps(tools_data, indent=2))
    else:
        print("\n=== Available Tools ===\n")
        for tool in tools_data:
            print(f"📦 {tool['name']}")
            print(f"   {tool['description']}")
            if tool['parameters']:
                print(f"   Parameters: {', '.join(tool['parameters'])}")
            print()
    
    return 0


def cmd_execute(args: argparse.Namespace) -> int:
    """Handle the execute command - run a specific tool."""
    import asyncio
    
    registry = setup_default_tools()
    
    tool_name = args.tool
    
    # Parse arguments
    tool_args = {}
    if args.args:
        for arg in args.args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                # Try to parse as JSON for complex values
                try:
                    tool_args[key] = json.loads(value)
                except json.JSONDecodeError:
                    tool_args[key] = value
    
    try:
        # Execute the tool (handle async)
        result = registry.execute(tool_name, **tool_args)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        
        if args.format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"\n=== Result from {tool_name} ===\n")
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"{key}: {value}")
            else:
                print(result)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return 1


async def cmd_run_async(args: argparse.Namespace) -> int:
    """Handle the run command - run an agent with a query (async)."""
    config = load_config(args.config)
    registry = setup_default_tools()
    
    # Create agent based on type
    agent_type = getattr(args, "agent_type", "react")
    
    if agent_type == "simple":
        agent = SimpleAgent(
            name="CLI Agent",
            tools=registry.list_tools(),
            tool_registry=registry,
        )
    else:
        agent = ReActAgent(
            name="CLI ReAct Agent",
            tools=registry.list_tools(),
            tool_registry=registry,
            max_iterations=args.max_iterations or 10,
        )
    
    query = args.query
    
    try:
        print(f"\n🤖 Running agent with query: {query}\n")
        print("-" * 50)
        
        # Run the agent
        result = await agent.run(query)
        
        # Format output
        formatter = OutputFormatterFactory.create(
            args.format or "markdown"
        )
        
        if hasattr(result, "return_values"):
            print("\n✅ Final Answer:")
            if isinstance(result.return_values, dict):
                for key, value in result.return_values.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   {result.return_values}")
        else:
            print(f"\n📤 Result: {result}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Synchronous wrapper for run command."""
    return asyncio.run(cmd_run_async(args))


def cmd_interactive(args: argparse.Namespace) -> int:
    """Handle the interactive command - start interactive REPL."""
    print(f"\n🧠 CogFlow Interactive Mode v{__version__}")
    print("=" * 50)
    print("Type 'help' for available commands, 'quit' to exit\n")
    
    registry = setup_default_tools()
    history: List[Dict[str, Any]] = []
    
    while True:
        try:
            user_input = input("cogflow> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == "help":
                print("\nAvailable commands:")
                print("  tools          - List available tools")
                print("  run <tool>     - Execute a tool")
                print("  history        - Show command history")
                print("  clear          - Clear screen")
                print("  quit           - Exit interactive mode")
                print("\nOr just type a natural language query to run the agent.\n")
                continue
            
            if user_input.lower() == "tools":
                for tool_meta in registry.list_tools():
                    print(f"  • {tool_meta.name}: {tool_meta.description}")
                print()
                continue
            
            if user_input.lower() == "history":
                for i, entry in enumerate(history, 1):
                    print(f"  {i}. {entry['input']}")
                print()
                continue
            
            if user_input.lower() == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue
            
            if user_input.lower().startswith("run "):
                parts = user_input[4:].split()
                tool_name = parts[0]
                tool_args = {}
                
                for part in parts[1:]:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        try:
                            tool_args[k] = json.loads(v)
                        except json.JSONDecodeError:
                            tool_args[k] = v
                
                try:
                    result = registry.execute(tool_name, **tool_args)
                    print(f"Result: {result}\n")
                    history.append({
                        "input": user_input,
                        "result": result,
                    })
                except Exception as e:
                    print(f"Error: {e}\n")
                continue
            
            # Treat as agent query
            print("Running agent query...")
            
            # Create a simple namespace for the run command
            run_args = argparse.Namespace(
                query=user_input,
                config=None,
                format="plain",
                max_iterations=10,
                verbose=False,
                agent_type="react",
            )
            
            result = asyncio.run(cmd_run_async(run_args))
            history.append({
                "input": user_input,
                "success": result == 0,
            })
            print()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except EOFError:
            print("\nGoodbye! 👋")
            break
    
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Handle the init command - initialize a new CogFlow project."""
    project_dir = Path(args.directory or ".")
    
    print(f"Initializing CogFlow project in {project_dir.absolute()}")
    
    # Create directory structure
    dirs = [
        "agents",
        "tools",
        "prompts",
        "pipelines",
        "tests",
    ]
    
    for d in dirs:
        (project_dir / d).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created {d}/")
    
    # Create default configuration file
    config_path = project_dir / "cogflow.json"
    if not config_path.exists():
        default_config = {
            "version": __version__,
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
            },
            "agent": {
                "max_iterations": 10,
                "verbose": True,
            },
            "tools": {
                "enabled": ["calculator", "text_processing", "datetime", "json_processor"],
            },
        }
        
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        print(f"  ✅ Created cogflow.json")
    
    # Create example agent file
    example_agent = project_dir / "agents" / "example_agent.py"
    if not example_agent.exists():
        example_code = '''"""Example CogFlow Agent."""

from cogflow.modules.agent import ReActAgent
from cogflow.modules.tool_registry import ToolRegistry
from cogflow.tools.builtin import CalculatorTool, TextProcessingTool


def create_agent():
    """Create and configure an example agent."""
    registry = ToolRegistry()
    registry.add_tool(CalculatorTool())
    registry.add_tool(TextProcessingTool())
    
    agent = ReActAgent(
        name="Example Agent",
        tools=registry.list_tools(),
        tool_registry=registry,
        max_iterations=10,
    )
    
    return agent


if __name__ == "__main__":
    import asyncio
    
    agent = create_agent()
    result = asyncio.run(agent.run("Calculate 25 * 4 and then convert the result to uppercase text"))
    print(f"Result: {result}")
'''
        
        with open(example_agent, "w") as f:
            f.write(example_code)
        print(f"  ✅ Created agents/example_agent.py")
    
    # Create example test file
    example_test = project_dir / "tests" / "test_example.py"
    if not example_test.exists():
        test_code = '''"""Example tests for CogFlow project."""

import pytest
from cogflow.tools.builtin import CalculatorTool


def test_calculator():
    """Test the calculator tool."""
    calc = CalculatorTool()
    result = calc.execute(expression="2 + 2")
    assert result["result"] == 4


def test_calculator_complex():
    """Test complex calculations."""
    calc = CalculatorTool()
    result = calc.execute(expression="(10 + 5) * 2")
    assert result["result"] == 30
'''
        
        with open(example_test, "w") as f:
            f.write(test_code)
        print(f"  ✅ Created tests/test_example.py")
    
    print(f"\n✨ Project initialized successfully!")
    print(f"\nNext steps:")
    print(f"  1. Edit cogflow.json to configure your LLM provider")
    print(f"  2. Create agents in the agents/ directory")
    print(f"  3. Run 'cogflow run \"your query\"' to test")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cogflow",
        description="CogFlow - A Modular Reasoning Framework for AI Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cogflow version                     Show version
  cogflow tools                       List available tools
  cogflow execute calculator expression="2+2"
  cogflow run "What is 25 times 4?"
  cogflow interactive                 Start interactive mode
  cogflow init                        Initialize a new project
        """,
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to configuration file",
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )
    version_parser.set_defaults(func=cmd_version)
    
    # Tools command
    tools_parser = subparsers.add_parser(
        "tools",
        help="List available tools",
    )
    tools_parser.add_argument(
        "-f", "--format",
        choices=["plain", "json"],
        default="plain",
        help="Output format",
    )
    tools_parser.set_defaults(func=cmd_tools)
    
    # Execute command
    execute_parser = subparsers.add_parser(
        "execute",
        help="Execute a specific tool",
    )
    execute_parser.add_argument(
        "tool",
        help="Name of the tool to execute",
    )
    execute_parser.add_argument(
        "args",
        nargs="*",
        help="Tool arguments in key=value format",
    )
    execute_parser.add_argument(
        "-f", "--format",
        choices=["plain", "json"],
        default="plain",
        help="Output format",
    )
    execute_parser.set_defaults(func=cmd_execute)
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run an agent with a query",
    )
    run_parser.add_argument(
        "query",
        help="The query or task for the agent",
    )
    run_parser.add_argument(
        "-t", "--agent-type",
        choices=["simple", "react"],
        default="react",
        help="Type of agent to use",
    )
    run_parser.add_argument(
        "-m", "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent iterations",
    )
    run_parser.add_argument(
        "-f", "--format",
        choices=["plain", "markdown", "json"],
        default="markdown",
        help="Output format",
    )
    run_parser.set_defaults(func=cmd_run)
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive",
        aliases=["i", "repl"],
        help="Start interactive mode",
    )
    interactive_parser.set_defaults(func=cmd_interactive)
    
    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new CogFlow project",
    )
    init_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to initialize (default: current)",
    )
    init_parser.set_defaults(func=cmd_init)
    
    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
