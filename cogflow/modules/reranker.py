"""
CogFlow Reranker Module

Provides candidate reranking functionality using LLM-based
comparison and scoring.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from cogflow.base.types import Candidate, Message, MessageRole
from cogflow.base.llm import BaseLLMProvider
from cogflow.prompts.cot_templates import TemplateRegistry


class RerankResult(BaseModel):
    """Result of a reranking operation."""
    
    query: str = Field(..., description="Original query")
    candidates: List[Candidate] = Field(
        default_factory=list,
        description="Reranked candidates with scores"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @property
    def top_candidate(self) -> Optional[Candidate]:
        """Get the top-ranked candidate."""
        if self.candidates:
            return self.candidates[0]
        return None
    
    def top_n(self, n: int = 5) -> List[Candidate]:
        """Get top N candidates."""
        return self.candidates[:n]


class BaseReranker(ABC):
    """Abstract base class for rerankers.
    
    Rerankers take a list of candidates and reorder them
    based on relevance to a query.
    """
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Rerank candidates based on relevance to query.
        
        Args:
            query: The query to rank against
            candidates: List of candidate texts
            top_k: Return only top K candidates
        
        Returns:
            RerankResult with scored and ordered candidates
        """
        pass
    
    def rerank_sync(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Synchronous version of rerank."""
        return asyncio.get_event_loop().run_until_complete(
            self.rerank(query, candidates, top_k)
        )


class LLMReranker(BaseReranker):
    """LLM-based reranker using pairwise comparison.
    
    Uses an LLM to compare candidates pairwise and determine
    the best ordering based on relevance scores.
    
    Example:
        ```python
        provider = OpenAIProvider(api_key="...")
        reranker = LLMReranker(provider)
        
        result = await reranker.rerank(
            query="Best programming language for web development",
            candidates=[
                "Python is great for backend development.",
                "JavaScript is essential for web development.",
                "C++ is used for system programming.",
            ]
        )
        
        print(result.top_candidate.content)
        ```
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        scoring_method: str = "pointwise",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        """Initialize LLM reranker.
        
        Args:
            llm_provider: LLM provider for scoring
            scoring_method: 'pointwise' or 'pairwise'
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature (0 for deterministic)
        """
        self.llm = llm_provider
        self.scoring_method = scoring_method
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._template_registry = TemplateRegistry()
    
    async def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Rerank candidates using LLM.
        
        Args:
            query: The query to rank against
            candidates: List of candidate texts
            top_k: Return only top K candidates
        
        Returns:
            RerankResult with scored candidates
        """
        if not candidates:
            return RerankResult(query=query, candidates=[])
        
        if self.scoring_method == "pairwise":
            scored = await self._pairwise_ranking(query, candidates)
        else:
            scored = await self._pointwise_scoring(query, candidates)
        
        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        
        # Apply top_k
        if top_k is not None:
            scored = scored[:top_k]
        
        return RerankResult(
            query=query,
            candidates=scored,
            metadata={
                "method": self.scoring_method,
                "original_count": len(candidates),
            }
        )
    
    async def _pointwise_scoring(
        self,
        query: str,
        candidates: List[str]
    ) -> List[Candidate]:
        """Score each candidate independently.
        
        Args:
            query: Query to score against
            candidates: List of candidates
        
        Returns:
            List of scored Candidate objects
        """
        template = self._template_registry.get("rerank_score")
        
        scored = []
        
        # Score candidates in parallel (batch)
        tasks = []
        for i, candidate in enumerate(candidates):
            tasks.append(self._score_single(query, candidate, i, template))
        
        results = await asyncio.gather(*tasks)
        
        for result in results:
            if result:
                scored.append(result)
        
        return scored
    
    async def _score_single(
        self,
        query: str,
        candidate: str,
        index: int,
        template: Optional[Any] = None,
    ) -> Optional[Candidate]:
        """Score a single candidate.
        
        Args:
            query: Query to score against
            candidate: Candidate text
            index: Original index
            template: Optional template to use
        
        Returns:
            Scored Candidate or None on error
        """
        if template:
            prompt = template.format(query=query, candidate=candidate)
        else:
            prompt = f"""Rate how relevant this candidate is to the query.

Query: {query}

Candidate: {candidate}

On a scale of 1-10, how relevant is this candidate? 
Respond with ONLY a number between 1 and 10."""
        
        try:
            messages = [Message(role=MessageRole.USER, content=prompt)]
            
            response = await self.llm.generate(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            # Parse score
            score = self._parse_score(response)
            
            return Candidate(
                content=candidate,
                score=score,
                metadata={"original_index": index}
            )
        except Exception as e:
            # Return with low score on error
            return Candidate(
                content=candidate,
                score=0.0,
                metadata={"original_index": index, "error": str(e)}
            )
    
    async def _pairwise_ranking(
        self,
        query: str,
        candidates: List[str]
    ) -> List[Candidate]:
        """Rank candidates using pairwise comparison.
        
        Uses tournament-style comparison to determine ranking.
        
        Args:
            query: Query to rank against
            candidates: List of candidates
        
        Returns:
            List of scored Candidate objects
        """
        template = self._template_registry.get("rerank_compare")
        
        n = len(candidates)
        
        # Initialize win counts
        wins = [0] * n
        
        # Compare all pairs (for small lists) or sample pairs (for large lists)
        pairs_to_compare = []
        if n <= 5:
            # Compare all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    pairs_to_compare.append((i, j))
        else:
            # Sample comparisons (each item compared ~3 times)
            import random
            for i in range(n):
                opponents = [j for j in range(n) if j != i]
                for j in random.sample(opponents, min(3, len(opponents))):
                    if (i, j) not in pairs_to_compare and (j, i) not in pairs_to_compare:
                        pairs_to_compare.append((i, j))
        
        # Run comparisons in parallel
        tasks = []
        for i, j in pairs_to_compare:
            tasks.append(self._compare_pair(
                query, candidates[i], candidates[j], i, j, template
            ))
        
        results = await asyncio.gather(*tasks)
        
        # Tally wins
        for winner_idx in results:
            if winner_idx is not None:
                wins[winner_idx] += 1
        
        # Create scored candidates
        max_wins = max(wins) if wins else 1
        scored = []
        for i, (candidate, win_count) in enumerate(zip(candidates, wins)):
            # Normalize score to 0-1
            score = win_count / max_wins if max_wins > 0 else 0.5
            scored.append(Candidate(
                content=candidate,
                score=score,
                metadata={"original_index": i, "wins": win_count}
            ))
        
        return scored
    
    async def _compare_pair(
        self,
        query: str,
        candidate_a: str,
        candidate_b: str,
        index_a: int,
        index_b: int,
        template: Optional[Any] = None,
    ) -> Optional[int]:
        """Compare two candidates.
        
        Args:
            query: Query to compare against
            candidate_a: First candidate
            candidate_b: Second candidate
            index_a: Index of first candidate
            index_b: Index of second candidate
            template: Optional template
        
        Returns:
            Index of winner, or None on error
        """
        if template:
            prompt = template.format(
                query=query,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )
        else:
            prompt = f"""Compare these two candidates for relevance to the query.

Query: {query}

Candidate A: {candidate_a}

Candidate B: {candidate_b}

Which candidate is MORE relevant to the query?
Answer with ONLY 'A' or 'B'."""
        
        try:
            messages = [Message(role=MessageRole.USER, content=prompt)]
            
            response = await self.llm.generate(
                messages=messages,
                max_tokens=10,
                temperature=self.temperature,
            )
            
            # Parse winner
            response = response.strip().upper()
            if "A" in response and "B" not in response:
                return index_a
            elif "B" in response and "A" not in response:
                return index_b
            elif response.startswith("A"):
                return index_a
            elif response.startswith("B"):
                return index_b
            
            return None
        except Exception:
            return None
    
    def _parse_score(self, response: str) -> float:
        """Parse score from LLM response.
        
        Args:
            response: LLM response
        
        Returns:
            Normalized score (0-1)
        """
        import re
        
        # Try to find a number
        matches = re.findall(r'(\d+(?:\.\d+)?)', response)
        
        if matches:
            score = float(matches[0])
            # Normalize to 0-1 (assuming 1-10 scale)
            if score > 1:
                score = score / 10
            return min(max(score, 0.0), 1.0)
        
        # Default score if can't parse
        return 0.5


class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder models.
    
    Uses sentence transformer cross-encoder models for
    efficient and accurate reranking.
    
    Requires: sentence-transformers library
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                )
    
    async def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Rerank using cross-encoder.
        
        Args:
            query: Query to rank against
            candidates: List of candidate texts
            top_k: Return only top K candidates
        
        Returns:
            RerankResult with scored candidates
        """
        if not candidates:
            return RerankResult(query=query, candidates=[])
        
        # Load model
        self._load_model()
        
        # Create query-candidate pairs
        pairs = [(query, candidate) for candidate in candidates]
        
        # Score all pairs
        scores = self._model.predict(pairs)
        
        # Create scored candidates
        scored = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            scored.append(Candidate(
                content=candidate,
                score=float(score),
                metadata={"original_index": i}
            ))
        
        # Sort by score
        scored.sort(key=lambda x: x.score, reverse=True)
        
        # Apply top_k
        if top_k is not None:
            scored = scored[:top_k]
        
        return RerankResult(
            query=query,
            candidates=scored,
            metadata={
                "model": self.model_name,
                "original_count": len(candidates),
            }
        )


class HybridReranker(BaseReranker):
    """Hybrid reranker combining multiple strategies.
    
    Uses a fast initial filtering followed by accurate
    LLM-based reranking of top candidates.
    """
    
    def __init__(
        self,
        llm_reranker: LLMReranker,
        initial_top_k: int = 10,
        final_top_k: int = 5,
    ):
        """Initialize hybrid reranker.
        
        Args:
            llm_reranker: LLM reranker for final ranking
            initial_top_k: Number of candidates for initial filter
            final_top_k: Final number of candidates to return
        """
        self.llm_reranker = llm_reranker
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
    
    async def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Rerank using hybrid approach.
        
        Args:
            query: Query to rank against
            candidates: List of candidate texts
            top_k: Return only top K candidates
        
        Returns:
            RerankResult with scored candidates
        """
        if len(candidates) <= self.initial_top_k:
            # Small list, use LLM directly
            return await self.llm_reranker.rerank(
                query, candidates, top_k or self.final_top_k
            )
        
        # Initial keyword-based filtering
        initial_filtered = self._keyword_filter(query, candidates, self.initial_top_k)
        
        # LLM reranking on filtered set
        result = await self.llm_reranker.rerank(
            query,
            initial_filtered,
            top_k or self.final_top_k,
        )
        
        result.metadata["hybrid"] = True
        result.metadata["initial_filtered"] = len(initial_filtered)
        
        return result
    
    def _keyword_filter(
        self,
        query: str,
        candidates: List[str],
        top_k: int
    ) -> List[str]:
        """Filter candidates by keyword overlap.
        
        Args:
            query: Query text
            candidates: Candidate texts
            top_k: Number to keep
        
        Returns:
            Filtered list of candidates
        """
        query_words = set(query.lower().split())
        
        scored = []
        for candidate in candidates:
            candidate_words = set(candidate.lower().split())
            overlap = len(query_words & candidate_words)
            scored.append((candidate, overlap))
        
        # Sort by overlap
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [c for c, _ in scored[:top_k]]


# Export all
__all__ = [
    "RerankResult",
    "BaseReranker",
    "LLMReranker",
    "CrossEncoderReranker",
    "HybridReranker",
]
