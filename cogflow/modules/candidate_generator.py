"""
CogFlow Candidate Generator

This module provides multi-sampling capabilities for self-consistency
and diverse response generation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import asyncio
import random
from collections import Counter

from cogflow.base.types import Candidate, ReasoningStep
from cogflow.base.llm import BaseLLMProvider, LLMResponse


@dataclass
class GenerationConfig:
    """Configuration for candidate generation."""
    
    num_samples: int = 5
    temperature: float = 0.7
    max_tokens: int = 1024
    diverse_prompts: bool = False
    timeout: float = 60.0
    concurrent_limit: int = 5


class BaseCandidateGenerator(ABC):
    """
    Abstract base class for candidate generators.
    
    Candidate generators produce multiple responses that can be
    aggregated or reranked for better quality.
    """
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> List[Candidate]:
        """
        Generate multiple candidate responses.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
        
        Returns:
            List of candidate responses
        """
        pass


class LLMCandidateGenerator(BaseCandidateGenerator):
    """
    Generates candidates using an LLM provider.
    
    Supports:
    - Temperature-based sampling
    - Parallel generation
    - Diverse prompt variations
    """
    
    def __init__(
        self,
        provider: BaseLLMProvider,
        default_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize LLM candidate generator.
        
        Args:
            provider: LLM provider for generation
            default_config: Default generation configuration
        """
        self.provider = provider
        self.default_config = default_config or GenerationConfig()
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> List[Candidate]:
        """Generate multiple candidates using the LLM."""
        cfg = config or self.default_config
        
        # Create generation tasks
        tasks = []
        semaphore = asyncio.Semaphore(cfg.concurrent_limit)
        
        for i in range(cfg.num_samples):
            # Vary temperature slightly for diversity
            temp_variation = 0.05 * (i - cfg.num_samples // 2)
            temperature = max(0.1, min(1.5, cfg.temperature + temp_variation))
            
            task = self._generate_single(
                prompt=prompt,
                temperature=temperature,
                max_tokens=cfg.max_tokens,
                semaphore=semaphore,
                sample_index=i,
                **kwargs
            )
            tasks.append(task)
        
        # Run generation with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=cfg.timeout
            )
        except asyncio.TimeoutError:
            results = []
        
        # Convert to candidates
        candidates = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            if isinstance(result, LLMResponse):
                candidate = Candidate(
                    content=result.content,
                    confidence=1.0 - (i * 0.05),  # Slight preference for earlier
                    source=f"sample_{i}",
                    metadata={
                        "temperature": cfg.temperature,
                        "model": result.model,
                        "tokens": result.usage,
                    }
                )
                candidates.append(candidate)
        
        return candidates
    
    async def _generate_single(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        semaphore: asyncio.Semaphore,
        sample_index: int,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a single candidate."""
        async with semaphore:
            return await self.provider.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )


class DiversePromptGenerator(BaseCandidateGenerator):
    """
    Generates candidates using prompt variations.
    
    Creates diverse outputs by rephrasing the prompt in
    different ways before generating.
    """
    
    def __init__(
        self,
        provider: BaseLLMProvider,
        prompt_variations: Optional[List[str]] = None
    ):
        """
        Initialize diverse prompt generator.
        
        Args:
            provider: LLM provider for generation
            prompt_variations: Template variations for prompts
        """
        self.provider = provider
        self.prompt_variations = prompt_variations or [
            "Please answer the following question: {prompt}",
            "Think step by step to solve: {prompt}",
            "Let's work through this carefully: {prompt}",
            "Consider this problem: {prompt}",
            "Here's a question for you: {prompt}",
        ]
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> List[Candidate]:
        """Generate candidates using prompt variations."""
        cfg = config or GenerationConfig()
        
        # Select prompt variations
        variations = self.prompt_variations[: cfg.num_samples]
        if len(variations) < cfg.num_samples:
            variations = variations * (cfg.num_samples // len(variations) + 1)
            variations = variations[: cfg.num_samples]
        
        # Generate with each variation
        tasks = []
        for i, variation in enumerate(variations):
            varied_prompt = variation.format(prompt=prompt)
            task = self.provider.generate(
                messages=[{"role": "user", "content": varied_prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                **kwargs
            )
            tasks.append((i, variation, task))
        
        # Gather results
        candidates = []
        for i, variation, task in tasks:
            try:
                result = await task
                candidate = Candidate(
                    content=result.content,
                    confidence=0.8,
                    source=f"variation_{i}",
                    metadata={
                        "prompt_variation": variation,
                    }
                )
                candidates.append(candidate)
            except Exception:
                continue
        
        return candidates


class SelfConsistencyAggregator:
    """
    Aggregates candidates using self-consistency voting.
    
    Self-consistency generates multiple reasoning paths and
    selects the most common answer.
    """
    
    def __init__(
        self,
        answer_extractor: Optional[Callable[[str], str]] = None,
        normalize: bool = True
    ):
        """
        Initialize self-consistency aggregator.
        
        Args:
            answer_extractor: Function to extract answer from response
            normalize: Whether to normalize answers before comparison
        """
        self.answer_extractor = answer_extractor or self._default_extractor
        self.normalize = normalize
    
    def _default_extractor(self, response: str) -> str:
        """Extract the final answer from a response."""
        # Look for common answer patterns
        patterns = [
            "the answer is",
            "the result is",
            "therefore,",
            "in conclusion,",
            "so,",
            "final answer:",
        ]
        
        response_lower = response.lower()
        for pattern in patterns:
            if pattern in response_lower:
                idx = response_lower.find(pattern)
                answer_part = response[idx + len(pattern):].strip()
                # Take first sentence
                for delim in [".", "\n", "!"]:
                    if delim in answer_part:
                        answer_part = answer_part.split(delim)[0]
                        break
                return answer_part.strip()
        
        # Fallback: return last line
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return lines[-1] if lines else response
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        if not self.normalize:
            return answer
        
        # Basic normalization
        normalized = answer.lower().strip()
        # Remove punctuation
        normalized = "".join(
            c for c in normalized
            if c.isalnum() or c.isspace()
        )
        # Collapse whitespace
        normalized = " ".join(normalized.split())
        return normalized
    
    def aggregate(
        self,
        candidates: List[Candidate],
        return_all_votes: bool = False
    ) -> Candidate:
        """
        Aggregate candidates using majority voting.
        
        Args:
            candidates: List of candidate responses
            return_all_votes: Include vote counts in metadata
        
        Returns:
            The winning candidate
        """
        if not candidates:
            raise ValueError("No candidates to aggregate")
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Extract and normalize answers
        answers = []
        for candidate in candidates:
            answer = self.answer_extractor(candidate.content)
            normalized = self._normalize_answer(answer)
            answers.append((normalized, candidate))
        
        # Count votes
        votes = Counter(a[0] for a in answers)
        
        # Find winning answer
        winning_answer, vote_count = votes.most_common(1)[0]
        
        # Get the first candidate with the winning answer
        for normalized, candidate in answers:
            if normalized == winning_answer:
                # Update with voting information
                winner = Candidate(
                    content=candidate.content,
                    reasoning_trace=candidate.reasoning_trace,
                    confidence=vote_count / len(candidates),
                    score=candidate.score,
                    metadata={
                        **candidate.metadata,
                        "vote_count": vote_count,
                        "total_candidates": len(candidates),
                        "consensus_rate": vote_count / len(candidates),
                    },
                    source=candidate.source,
                )
                if return_all_votes:
                    winner.metadata["all_votes"] = dict(votes)
                return winner
        
        # Fallback (shouldn't happen)
        return candidates[0]
    
    def get_consensus_rate(self, candidates: List[Candidate]) -> float:
        """Calculate the consensus rate among candidates."""
        if len(candidates) <= 1:
            return 1.0
        
        answers = []
        for candidate in candidates:
            answer = self.answer_extractor(candidate.content)
            normalized = self._normalize_answer(answer)
            answers.append(normalized)
        
        votes = Counter(answers)
        _, max_votes = votes.most_common(1)[0]
        return max_votes / len(candidates)


class WeightedVotingAggregator:
    """
    Aggregates candidates using weighted voting.
    
    Uses candidate confidence/scores as weights.
    """
    
    def __init__(
        self,
        answer_extractor: Optional[Callable[[str], str]] = None,
        weight_key: str = "confidence"
    ):
        """
        Initialize weighted voting aggregator.
        
        Args:
            answer_extractor: Function to extract answer
            weight_key: Which attribute to use as weight
        """
        self.answer_extractor = answer_extractor or (lambda x: x.strip())
        self.weight_key = weight_key
    
    def aggregate(self, candidates: List[Candidate]) -> Candidate:
        """Aggregate candidates using weighted voting."""
        if not candidates:
            raise ValueError("No candidates to aggregate")
        
        # Extract answers and weights
        answer_weights: Dict[str, float] = {}
        answer_candidates: Dict[str, Candidate] = {}
        
        for candidate in candidates:
            answer = self.answer_extractor(candidate.content)
            weight = getattr(candidate, self.weight_key, 1.0) or 1.0
            
            if answer not in answer_weights:
                answer_weights[answer] = 0.0
                answer_candidates[answer] = candidate
            answer_weights[answer] += weight
        
        # Find winner
        winning_answer = max(answer_weights.keys(), key=lambda a: answer_weights[a])
        winner = answer_candidates[winning_answer]
        
        return Candidate(
            content=winner.content,
            reasoning_trace=winner.reasoning_trace,
            confidence=answer_weights[winning_answer] / sum(answer_weights.values()),
            score=winner.score,
            metadata={
                **winner.metadata,
                "weighted_vote": answer_weights[winning_answer],
                "total_weight": sum(answer_weights.values()),
            },
            source=winner.source,
        )


class EnsembleCandidateGenerator(BaseCandidateGenerator):
    """
    Combines multiple generators for ensemble generation.
    """
    
    def __init__(self, generators: List[BaseCandidateGenerator]):
        """
        Initialize ensemble generator.
        
        Args:
            generators: List of candidate generators to combine
        """
        self.generators = generators
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> List[Candidate]:
        """Generate candidates from all generators."""
        cfg = config or GenerationConfig()
        samples_per_generator = max(1, cfg.num_samples // len(self.generators))
        
        generator_config = GenerationConfig(
            num_samples=samples_per_generator,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )
        
        # Generate from all generators
        tasks = [
            gen.generate(prompt, generator_config, **kwargs)
            for gen in self.generators
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine candidates
        all_candidates = []
        for i, result in enumerate(results):
            if isinstance(result, list):
                for candidate in result:
                    candidate.metadata["generator_index"] = i
                    all_candidates.append(candidate)
        
        return all_candidates


__all__ = [
    "GenerationConfig",
    "BaseCandidateGenerator",
    "LLMCandidateGenerator",
    "DiversePromptGenerator",
    "SelfConsistencyAggregator",
    "WeightedVotingAggregator",
    "EnsembleCandidateGenerator",
]
