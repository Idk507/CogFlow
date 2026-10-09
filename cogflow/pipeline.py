"""
Pipeline Module for CogFlow Framework.

Provides a composable pipeline builder with fluent API and | operator support
for chaining components like agents, tools, and formatters.
"""

from abc import ABC, abstractmethod
from typing import (
    Dict,
    Any,
    List,
    Optional,
    Union,
    Callable,
    TypeVar,
    Generic,
    Awaitable,
)
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import uuid
import logging
from functools import wraps

from cogflow.base.types import ReasoningStep, AgentAction, AgentFinish
from cogflow.base.callbacks import CallbackManager, BaseCallback


logger = logging.getLogger(__name__)


T = TypeVar("T")
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class PipelineStatus(Enum):
    """Status of a pipeline execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineContext:
    """Context object passed through pipeline stages."""
    
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: PipelineStatus = PipelineStatus.PENDING
    errors: List[Exception] = field(default_factory=list)
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def set(self, key: str, value: Any) -> "PipelineContext":
        """Set a value in the context data."""
        self.data[key] = value
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context data."""
        return self.data.get(key, default)
    
    def update(self, data: Dict[str, Any]) -> "PipelineContext":
        """Update context data with multiple values."""
        self.data.update(data)
        return self
    
    def add_result(
        self, step_name: str, result: Any, duration: float
    ) -> None:
        """Add a step result to the execution history."""
        self.step_results.append({
            "step": step_name,
            "result": result,
            "duration": duration,
            "timestamp": time.time(),
        })
    
    @property
    def duration(self) -> Optional[float]:
        """Get the total execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "data": self.data,
            "metadata": self.metadata,
            "duration": self.duration,
            "step_count": len(self.step_results),
            "errors": [str(e) for e in self.errors],
        }


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the pipeline step."""
        self._name = name or self.__class__.__name__
        self._next: Optional["PipelineStep"] = None
    
    @property
    def name(self) -> str:
        """Return the step name."""
        return self._name
    
    @abstractmethod
    async def execute(
        self, context: PipelineContext
    ) -> PipelineContext:
        """Execute this step on the context."""
        pass
    
    def __or__(self, other: "PipelineStep") -> "Pipeline":
        """Enable chaining with | operator."""
        pipeline = Pipeline()
        pipeline.add_step(self)
        pipeline.add_step(other)
        return pipeline
    
    def __rshift__(self, other: "PipelineStep") -> "Pipeline":
        """Enable chaining with >> operator."""
        return self | other


class FunctionStep(PipelineStep):
    """Pipeline step that wraps a function."""
    
    def __init__(
        self,
        func: Callable[[PipelineContext], Union[PipelineContext, Awaitable[PipelineContext]]],
        name: Optional[str] = None,
    ):
        """Initialize with a function."""
        super().__init__(name or func.__name__)
        self._func = func
        self._is_async = asyncio.iscoroutinefunction(func)
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the wrapped function."""
        if self._is_async:
            return await self._func(context)
        else:
            return self._func(context)


class TransformStep(PipelineStep):
    """Pipeline step that transforms data."""
    
    def __init__(
        self,
        transform_func: Callable[[Any], Any],
        input_key: str = "input",
        output_key: str = "output",
        name: Optional[str] = None,
    ):
        """Initialize the transform step."""
        super().__init__(name or "transform")
        self._transform = transform_func
        self._input_key = input_key
        self._output_key = output_key
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Apply transformation to input data."""
        input_data = context.get(self._input_key)
        
        if asyncio.iscoroutinefunction(self._transform):
            output = await self._transform(input_data)
        else:
            output = self._transform(input_data)
        
        context.set(self._output_key, output)
        return context


class ConditionalStep(PipelineStep):
    """Pipeline step that conditionally executes one of two branches."""
    
    def __init__(
        self,
        condition: Callable[[PipelineContext], bool],
        if_true: PipelineStep,
        if_false: Optional[PipelineStep] = None,
        name: Optional[str] = None,
    ):
        """Initialize the conditional step."""
        super().__init__(name or "conditional")
        self._condition = condition
        self._if_true = if_true
        self._if_false = if_false
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the appropriate branch based on condition."""
        if self._condition(context):
            return await self._if_true.execute(context)
        elif self._if_false:
            return await self._if_false.execute(context)
        return context


class ParallelStep(PipelineStep):
    """Pipeline step that executes multiple steps in parallel."""
    
    def __init__(
        self,
        steps: List[PipelineStep],
        merge_strategy: str = "dict",
        name: Optional[str] = None,
    ):
        """Initialize the parallel step."""
        super().__init__(name or "parallel")
        self._steps = steps
        self._merge_strategy = merge_strategy
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute all steps in parallel and merge results."""
        # Create copies of context for each step
        contexts = [
            PipelineContext(
                data=context.data.copy(),
                metadata=context.metadata.copy(),
                execution_id=context.execution_id,
            )
            for _ in self._steps
        ]
        
        # Execute all steps in parallel
        results = await asyncio.gather(
            *[step.execute(ctx) for step, ctx in zip(self._steps, contexts)],
            return_exceptions=True,
        )
        
        # Merge results based on strategy
        if self._merge_strategy == "dict":
            for i, (step, result) in enumerate(zip(self._steps, results)):
                if isinstance(result, Exception):
                    context.errors.append(result)
                else:
                    context.set(f"{step.name}_result", result.data)
        elif self._merge_strategy == "list":
            merged_results = []
            for result in results:
                if isinstance(result, Exception):
                    context.errors.append(result)
                else:
                    merged_results.append(result.data)
            context.set("parallel_results", merged_results)
        
        return context


class RetryStep(PipelineStep):
    """Pipeline step that retries on failure."""
    
    def __init__(
        self,
        step: PipelineStep,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        name: Optional[str] = None,
    ):
        """Initialize the retry step."""
        super().__init__(name or f"retry_{step.name}")
        self._step = step
        self._max_retries = max_retries
        self._delay = delay
        self._backoff = backoff
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute step with retry logic."""
        last_error = None
        delay = self._delay
        
        for attempt in range(self._max_retries + 1):
            try:
                return await self._step.execute(context)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Step {self._step.name} failed (attempt {attempt + 1}/"
                    f"{self._max_retries + 1}): {e}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)
                    delay *= self._backoff
        
        context.errors.append(last_error)
        raise last_error


class LoopStep(PipelineStep):
    """Pipeline step that loops until a condition is met."""
    
    def __init__(
        self,
        step: PipelineStep,
        condition: Callable[[PipelineContext], bool],
        max_iterations: int = 100,
        name: Optional[str] = None,
    ):
        """Initialize the loop step."""
        super().__init__(name or f"loop_{step.name}")
        self._step = step
        self._condition = condition
        self._max_iterations = max_iterations
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute step in a loop until condition is false."""
        iteration = 0
        
        while self._condition(context) and iteration < self._max_iterations:
            context = await self._step.execute(context)
            iteration += 1
            context.metadata["loop_iteration"] = iteration
        
        context.metadata["total_iterations"] = iteration
        return context


class Pipeline(PipelineStep):
    """
    Composable pipeline for chaining processing steps.
    
    Supports:
    - Fluent API with .add_step()
    - | operator for chaining
    - >> operator as alternative to |
    - Conditional branching
    - Parallel execution
    - Retry with backoff
    - Callbacks for monitoring
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        """Initialize the pipeline."""
        super().__init__(name or "pipeline")
        self._steps: List[PipelineStep] = []
        self._callback_manager = callback_manager or CallbackManager()
        self._on_error: Optional[Callable[[Exception, PipelineContext], None]] = None
    
    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Add a step to the pipeline."""
        self._steps.append(step)
        return self
    
    def add_function(
        self,
        func: Callable[[PipelineContext], PipelineContext],
        name: Optional[str] = None,
    ) -> "Pipeline":
        """Add a function as a pipeline step."""
        return self.add_step(FunctionStep(func, name))
    
    def add_transform(
        self,
        transform: Callable[[Any], Any],
        input_key: str = "input",
        output_key: str = "output",
        name: Optional[str] = None,
    ) -> "Pipeline":
        """Add a data transformation step."""
        return self.add_step(
            TransformStep(transform, input_key, output_key, name)
        )
    
    def add_conditional(
        self,
        condition: Callable[[PipelineContext], bool],
        if_true: PipelineStep,
        if_false: Optional[PipelineStep] = None,
        name: Optional[str] = None,
    ) -> "Pipeline":
        """Add a conditional branch."""
        return self.add_step(
            ConditionalStep(condition, if_true, if_false, name)
        )
    
    def add_parallel(
        self,
        steps: List[PipelineStep],
        merge_strategy: str = "dict",
        name: Optional[str] = None,
    ) -> "Pipeline":
        """Add parallel execution of multiple steps."""
        return self.add_step(ParallelStep(steps, merge_strategy, name))
    
    def add_retry(
        self,
        step: PipelineStep,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
    ) -> "Pipeline":
        """Add a step with retry logic."""
        return self.add_step(
            RetryStep(step, max_retries, delay, backoff)
        )
    
    def add_loop(
        self,
        step: PipelineStep,
        condition: Callable[[PipelineContext], bool],
        max_iterations: int = 100,
    ) -> "Pipeline":
        """Add a looping step."""
        return self.add_step(LoopStep(step, condition, max_iterations))
    
    def on_error(
        self,
        handler: Callable[[Exception, PipelineContext], None],
    ) -> "Pipeline":
        """Set error handler for the pipeline."""
        self._on_error = handler
        return self
    
    def add_callback(self, callback: BaseCallback) -> "Pipeline":
        """Add a callback handler to the pipeline."""
        self._callback_manager.add_handler(callback)
        return self
    
    def __or__(self, other: PipelineStep) -> "Pipeline":
        """Enable chaining with | operator."""
        if isinstance(other, Pipeline):
            for step in other._steps:
                self.add_step(step)
        else:
            self.add_step(other)
        return self
    
    def __ror__(self, other: PipelineStep) -> "Pipeline":
        """Enable chaining when Pipeline is on the right side."""
        new_pipeline = Pipeline(self.name, self._callback_manager)
        new_pipeline.add_step(other)
        for step in self._steps:
            new_pipeline.add_step(step)
        return new_pipeline
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute all steps in sequence."""
        context.status = PipelineStatus.RUNNING
        context.start_time = time.time()
        
        try:
            await self._callback_manager.on_agent_start(
                {"pipeline": self.name, "steps": [s.name for s in self._steps]}
            )
            
            for step in self._steps:
                step_start = time.time()
                
                await self._callback_manager.on_tool_start(
                    step.name, context.data
                )
                
                try:
                    context = await step.execute(context)
                    step_duration = time.time() - step_start
                    context.add_result(step.name, "success", step_duration)
                    
                    await self._callback_manager.on_tool_end(
                        step.name, {"status": "success", "duration": step_duration}
                    )
                except Exception as e:
                    step_duration = time.time() - step_start
                    context.add_result(step.name, f"error: {e}", step_duration)
                    context.errors.append(e)
                    
                    await self._callback_manager.on_error(e)
                    
                    if self._on_error:
                        self._on_error(e, context)
                    else:
                        raise
            
            context.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            context.status = PipelineStatus.FAILED
            logger.error(f"Pipeline {self.name} failed: {e}")
            raise
        
        finally:
            context.end_time = time.time()
            await self._callback_manager.on_agent_end(context.to_dict())
        
        return context
    
    async def run(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PipelineContext:
        """
        Run the pipeline with the given input.
        
        Args:
            input_data: Initial data for the context
            **kwargs: Additional context data
            
        Returns:
            The final pipeline context
        """
        context = PipelineContext(
            data=input_data or {},
            metadata=kwargs,
        )
        return await self.execute(context)
    
    def run_sync(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PipelineContext:
        """Synchronous version of run()."""
        return asyncio.run(self.run(input_data, **kwargs))
    
    @property
    def steps(self) -> List[PipelineStep]:
        """Return the list of steps."""
        return self._steps.copy()
    
    def __len__(self) -> int:
        """Return the number of steps."""
        return len(self._steps)
    
    def __repr__(self) -> str:
        """Return string representation."""
        step_names = " | ".join(s.name for s in self._steps)
        return f"Pipeline({step_names})"


# Decorator for creating pipeline steps from functions
def pipeline_step(name: Optional[str] = None):
    """Decorator to convert a function into a PipelineStep."""
    def decorator(func: Callable) -> FunctionStep:
        return FunctionStep(func, name or func.__name__)
    return decorator


# Pre-built utility steps
class LogStep(PipelineStep):
    """Step that logs context data."""
    
    def __init__(
        self,
        message: str = "Pipeline step",
        log_level: int = logging.INFO,
        keys: Optional[List[str]] = None,
    ):
        """Initialize the log step."""
        super().__init__("log")
        self._message = message
        self._log_level = log_level
        self._keys = keys
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Log the specified message and data."""
        data_to_log = (
            {k: context.get(k) for k in self._keys}
            if self._keys
            else context.data
        )
        logger.log(self._log_level, f"{self._message}: {data_to_log}")
        return context


class DelayStep(PipelineStep):
    """Step that adds a delay."""
    
    def __init__(self, seconds: float):
        """Initialize the delay step."""
        super().__init__("delay")
        self._seconds = seconds
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Wait for the specified duration."""
        await asyncio.sleep(self._seconds)
        return context


class ValidateStep(PipelineStep):
    """Step that validates context data."""
    
    def __init__(
        self,
        validator: Callable[[PipelineContext], bool],
        error_message: str = "Validation failed",
    ):
        """Initialize the validation step."""
        super().__init__("validate")
        self._validator = validator
        self._error_message = error_message
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Validate the context data."""
        if not self._validator(context):
            raise ValueError(self._error_message)
        return context


# Convenience functions
def create_pipeline(*steps: PipelineStep, name: str = "pipeline") -> Pipeline:
    """Create a pipeline from a sequence of steps."""
    pipeline = Pipeline(name)
    for step in steps:
        pipeline.add_step(step)
    return pipeline


def chain(*funcs: Callable) -> Pipeline:
    """Create a pipeline from a sequence of functions."""
    pipeline = Pipeline("function_chain")
    for func in funcs:
        pipeline.add_function(func)
    return pipeline


# Export public API
__all__ = [
    "PipelineStatus",
    "PipelineContext",
    "PipelineStep",
    "FunctionStep",
    "TransformStep",
    "ConditionalStep",
    "ParallelStep",
    "RetryStep",
    "LoopStep",
    "Pipeline",
    "LogStep",
    "DelayStep",
    "ValidateStep",
    "pipeline_step",
    "create_pipeline",
    "chain",
]
