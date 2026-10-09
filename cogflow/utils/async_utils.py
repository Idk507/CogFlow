"""
CogFlow Async Utilities

Helper functions for asynchronous operations.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Awaitable, Callable, List, Optional, TypeVar

T = TypeVar('T')


def run_async(coro: Awaitable[T]) -> T:
    """Run an async coroutine synchronously.
    
    Args:
        coro: Coroutine to run
    
    Returns:
        Result of the coroutine
    
    Example:
        ```python
        result = run_async(some_async_function())
        ```
    """
    try:
        loop = asyncio.get_running_loop()
        # Already in async context, create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(coro)


async def gather_with_concurrency(
    tasks: List[Awaitable[T]],
    max_concurrency: int = 10,
) -> List[T]:
    """Run tasks with limited concurrency.
    
    Args:
        tasks: List of coroutines to run
        max_concurrency: Maximum concurrent tasks
    
    Returns:
        List of results in order
    
    Example:
        ```python
        async def fetch(url):
            ...
        
        urls = [...]
        tasks = [fetch(url) for url in urls]
        results = await gather_with_concurrency(tasks, max_concurrency=5)
        ```
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def bounded_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task
    
    bounded = [bounded_task(task) for task in tasks]
    return await asyncio.gather(*bounded)


async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    **kwargs: Any,
) -> T:
    """Retry an async function with exponential backoff.
    
    Args:
        func: Async function to call
        max_retries: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry
        **kwargs: Arguments to pass to func
    
    Returns:
        Result from successful call
    
    Raises:
        Last exception if all retries fail
    
    Example:
        ```python
        result = await retry_async(
            api_call,
            max_retries=3,
            delay=1.0,
            param="value"
        )
        ```
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            return await func(**kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                raise
    
    raise last_exception  # type: ignore


def async_to_sync(func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """Decorator to convert async function to sync.
    
    Args:
        func: Async function
    
    Returns:
        Sync wrapper function
    
    Example:
        ```python
        @async_to_sync
        async def fetch_data():
            ...
        
        # Now can be called synchronously
        result = fetch_data()
        ```
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return run_async(func(*args, **kwargs))
    
    return wrapper


class AsyncPool:
    """Async task pool with limited concurrency.
    
    Example:
        ```python
        pool = AsyncPool(max_workers=5)
        
        async with pool:
            for url in urls:
                await pool.submit(fetch, url)
            
            results = await pool.results()
        ```
    """
    
    def __init__(self, max_workers: int = 10):
        """Initialize pool.
        
        Args:
            max_workers: Maximum concurrent workers
        """
        self.max_workers = max_workers
        self.semaphore: Optional[asyncio.Semaphore] = None
        self._tasks: List[asyncio.Task] = []
        self._results: List[Any] = []
    
    async def __aenter__(self):
        self.semaphore = asyncio.Semaphore(self.max_workers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Wait for all tasks
        if self._tasks:
            self._results = await asyncio.gather(*self._tasks, return_exceptions=True)
        return False
    
    async def submit(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Submit a task to the pool.
        
        Args:
            func: Async function
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        async def bounded():
            async with self.semaphore:  # type: ignore
                return await func(*args, **kwargs)
        
        task = asyncio.create_task(bounded())
        self._tasks.append(task)
    
    async def results(self) -> List[Any]:
        """Get all results.
        
        Returns:
            List of results or exceptions
        """
        if self._tasks and not self._results:
            self._results = await asyncio.gather(*self._tasks, return_exceptions=True)
        return self._results


__all__ = [
    "run_async",
    "gather_with_concurrency",
    "retry_async",
    "async_to_sync",
    "AsyncPool",
]
