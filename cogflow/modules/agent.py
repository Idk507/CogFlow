"""
CogFlow Agent Module

Provides the ReAct (Reasoning + Acting) agent implementation
that combines reasoning with tool execution.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from cogflow.base.types import (
    Message,
    MessageRole,
    ReasoningStep,
    AgentAction,
    AgentFinish,
    ToolCall,
)
from cogflow.base.llm import BaseLLMProvider
from cogflow.modules.tool_registry import ToolRegistry
from cogflow.prompts.cot_templates import TemplateRegistry


class AgentState(BaseModel):
    """State of an agent execution."""
    
    messages: List[Message] = Field(
        default_factory=list,
        description="Conversation history"
    )
    steps: List[ReasoningStep] = Field(
        default_factory=list,
        description="Reasoning steps taken"
    )
    actions: List[AgentAction] = Field(
        default_factory=list,
        description="Actions executed"
    )
    observations: List[str] = Field(
        default_factory=list,
        description="Observations from tool executions"
    )
    current_step: int = Field(default=0, description="Current step number")
    is_complete: bool = Field(default=False, description="Whether execution is complete")
    final_answer: Optional[str] = Field(default=None, description="Final answer if complete")


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    
    max_steps: int = Field(default=10, description="Maximum reasoning steps")
    max_tokens: int = Field(default=2048, description="Maximum tokens per generation")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    stop_sequences: List[str] = Field(
        default_factory=lambda: ["Observation:", "\nObservation"],
        description="Stop sequences for generation"
    )
    verbose: bool = Field(default=False, description="Enable verbose logging")


class ReActAgent:
    """ReAct Agent: Reasoning + Acting.
    
    Implements the ReAct pattern for tool-augmented reasoning:
    1. Think: Generate a thought about what to do
    2. Act: Decide on an action (tool call) or finish
    3. Observe: Execute the action and observe the result
    4. Repeat until the task is complete
    
    Example:
        ```python
        provider = OpenAIProvider(api_key="...")
        registry = ToolRegistry()
        
        @registry.register(description="Calculate math")
        def calculator(expression: str) -> float:
            return eval(expression)
        
        agent = ReActAgent(
            llm_provider=provider,
            tool_registry=registry,
        )
        
        result = await agent.run("What is 25 * 4?")
        print(result.final_answer)
        ```
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize ReAct agent.
        
        Args:
            llm_provider: LLM provider for generation
            tool_registry: Registry of available tools
            config: Agent configuration
            system_prompt: Optional custom system prompt
        """
        self.llm = llm_provider
        self.tools = tool_registry or ToolRegistry()
        self.config = config or AgentConfig()
        self._template_registry = TemplateRegistry()
        
        # Build system prompt
        if system_prompt:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        # Get ReAct template
        template = self._template_registry.get("react_system")
        
        # Build tool descriptions
        tool_descriptions = []
        for meta in self.tools.list_tools():
            desc = f"- {meta.name}: {meta.description}"
            tool_descriptions.append(desc)
        
        if not tool_descriptions:
            tool_descriptions = ["- No tools available"]
        
        tools_str = "\n".join(tool_descriptions)
        
        if template:
            return template.format(tools=tools_str)
        
        # Fallback prompt
        return f"""You are a helpful AI assistant that can use tools to answer questions.

Available tools:
{tools_str}

To use a tool, respond with:
Thought: [your reasoning about what to do]
Action: [tool_name]
Action Input: [input for the tool as JSON]

After receiving the observation, continue reasoning.
When you have the final answer, respond with:
Thought: [final reasoning]
Final Answer: [your answer]

Always think step by step."""
    
    async def run(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> AgentState:
        """Run the agent on a query.
        
        Args:
            query: User query to process
            context: Optional additional context
        
        Returns:
            AgentState with final result
        """
        state = AgentState()
        
        # Initialize conversation
        state.messages.append(Message(
            role=MessageRole.SYSTEM,
            content=self._system_prompt,
        ))
        
        # Add user query
        user_content = query
        if context:
            user_content = f"Context: {context}\n\nQuestion: {query}"
        
        state.messages.append(Message(
            role=MessageRole.USER,
            content=user_content,
        ))
        
        # ReAct loop
        while state.current_step < self.config.max_steps and not state.is_complete:
            state.current_step += 1
            
            if self.config.verbose:
                print(f"\n--- Step {state.current_step} ---")
            
            # Generate thought/action
            response = await self._generate_step(state)
            
            # Parse response
            parsed = self._parse_response(response)
            
            if isinstance(parsed, AgentFinish):
                # Agent decided to finish
                state.is_complete = True
                state.final_answer = parsed.output
                
                state.steps.append(ReasoningStep(
                    step_number=state.current_step,
                    content=parsed.thought or "Completed reasoning",
                    step_type="conclusion",
                ))
                
                if self.config.verbose:
                    print(f"Final Answer: {state.final_answer}")
            
            elif isinstance(parsed, AgentAction):
                # Execute tool
                state.actions.append(parsed)
                
                state.steps.append(ReasoningStep(
                    step_number=state.current_step,
                    content=f"Thought: {parsed.thought}\nAction: {parsed.tool}",
                    step_type="action",
                ))
                
                if self.config.verbose:
                    print(f"Thought: {parsed.thought}")
                    print(f"Action: {parsed.tool}({parsed.tool_input})")
                
                # Execute action
                observation = await self._execute_action(parsed)
                state.observations.append(observation)
                
                if self.config.verbose:
                    print(f"Observation: {observation}")
                
                # Add observation to conversation
                state.messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=response,
                ))
                state.messages.append(Message(
                    role=MessageRole.USER,
                    content=f"Observation: {observation}",
                ))
            
            else:
                # Couldn't parse, add as reasoning step
                state.steps.append(ReasoningStep(
                    step_number=state.current_step,
                    content=response,
                    step_type="reasoning",
                ))
                
                state.messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=response,
                ))
        
        # If max steps reached without answer
        if not state.is_complete:
            state.final_answer = self._extract_best_answer(state)
            state.is_complete = True
        
        return state
    
    async def _generate_step(self, state: AgentState) -> str:
        """Generate the next step.
        
        Args:
            state: Current agent state
        
        Returns:
            LLM response string
        """
        response = await self.llm.generate(
            messages=state.messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=self.config.stop_sequences,
        )
        
        return response.strip()
    
    def _parse_response(
        self,
        response: str
    ) -> Union[AgentAction, AgentFinish, None]:
        """Parse LLM response into action or finish.
        
        Args:
            response: Raw LLM response
        
        Returns:
            AgentAction, AgentFinish, or None if can't parse
        """
        # Check for Final Answer
        final_match = re.search(
            r"Final\s*Answer[:\s]+(.+?)$",
            response,
            re.IGNORECASE | re.DOTALL
        )
        
        if final_match:
            thought_match = re.search(
                r"Thought[:\s]+(.+?)(?=Final\s*Answer)",
                response,
                re.IGNORECASE | re.DOTALL
            )
            thought = thought_match.group(1).strip() if thought_match else None
            
            return AgentFinish(
                output=final_match.group(1).strip(),
                thought=thought,
            )
        
        # Check for Action
        action_match = re.search(
            r"Action[:\s]+(\w+)\s*(?:\n|Action\s*Input[:\s]+)(.+?)$",
            response,
            re.IGNORECASE | re.DOTALL
        )
        
        if action_match:
            tool_name = action_match.group(1).strip()
            action_input = action_match.group(2).strip()
            
            # Try to parse as JSON
            tool_input = self._parse_action_input(action_input)
            
            thought_match = re.search(
                r"Thought[:\s]+(.+?)(?=Action[:\s])",
                response,
                re.IGNORECASE | re.DOTALL
            )
            thought = thought_match.group(1).strip() if thought_match else ""
            
            return AgentAction(
                tool=tool_name,
                tool_input=tool_input,
                thought=thought,
            )
        
        return None
    
    def _parse_action_input(self, input_str: str) -> Dict[str, Any]:
        """Parse action input string to dictionary.
        
        Args:
            input_str: Input string (might be JSON or plain text)
        
        Returns:
            Dictionary of arguments
        """
        import json
        
        input_str = input_str.strip()
        
        # Try JSON first
        try:
            if input_str.startswith("{"):
                return json.loads(input_str)
        except json.JSONDecodeError:
            pass
        
        # Try key=value format
        if "=" in input_str:
            result = {}
            pairs = input_str.split(",")
            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    result[key.strip()] = value.strip().strip('"\'')
            if result:
                return result
        
        # Treat as single argument
        return {"input": input_str}
    
    async def _execute_action(self, action: AgentAction) -> str:
        """Execute an action and return observation.
        
        Args:
            action: Action to execute
        
        Returns:
            Observation string
        """
        tool_name = action.tool.lower()
        
        if not self.tools.has_tool(tool_name):
            return f"Error: Tool '{action.tool}' not found. Available tools: {[t.name for t in self.tools.list_tools()]}"
        
        try:
            result = await self.tools.execute(
                name=tool_name,
                validate=True,
                **action.tool_input,
            )
            
            # Convert result to string
            if isinstance(result, dict):
                import json
                return json.dumps(result, indent=2)
            return str(result)
        
        except Exception as e:
            return f"Error executing {action.tool}: {str(e)}"
    
    def _extract_best_answer(self, state: AgentState) -> str:
        """Extract best answer when max steps reached.
        
        Args:
            state: Current agent state
        
        Returns:
            Best available answer
        """
        # Look through observations for useful data
        if state.observations:
            return f"Based on observations: {state.observations[-1]}"
        
        # Use last reasoning step
        if state.steps:
            return state.steps[-1].content
        
        return "Unable to complete reasoning within step limit."
    
    def run_sync(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> AgentState:
        """Synchronous version of run.
        
        Args:
            query: User query
            context: Optional context
        
        Returns:
            AgentState with result
        """
        return asyncio.get_event_loop().run_until_complete(
            self.run(query, context)
        )


class SimpleAgent:
    """Simple agent without ReAct loop.
    
    Directly calls tools based on user intent without
    explicit reasoning traces.
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        tool_registry: Optional[ToolRegistry] = None,
        max_tokens: int = 1024,
    ):
        """Initialize simple agent.
        
        Args:
            llm_provider: LLM provider
            tool_registry: Tool registry
            max_tokens: Max generation tokens
        """
        self.llm = llm_provider
        self.tools = tool_registry or ToolRegistry()
        self.max_tokens = max_tokens
    
    async def run(
        self,
        query: str,
        use_tools: bool = True,
    ) -> str:
        """Process a query.
        
        Args:
            query: User query
            use_tools: Whether to use tools
        
        Returns:
            Response string
        """
        messages = [Message(role=MessageRole.USER, content=query)]
        
        # Get tool schemas if using tools
        tools = self.tools.get_schemas() if use_tools else None
        
        # Generate response
        response = await self.llm.generate(
            messages=messages,
            max_tokens=self.max_tokens,
            tools=tools,
        )
        
        return response


# Export all
__all__ = [
    "AgentState",
    "AgentConfig",
    "ReActAgent",
    "SimpleAgent",
]
