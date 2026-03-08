"""
API Schemas Module

This module defines Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict


class QueryRequest(BaseModel):
    """Request schema for query endpoint."""
    
    query: str = Field(
        ...,
        description="Natural language query text",
        min_length=1,
        max_length=1000,
        examples=["What are the best graphics cards for gaming?"]
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use semantic cache"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to return",
        ge=1,
        le=50
    )


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""
    
    query: str = Field(..., description="Original query text")
    cache_hit: bool = Field(..., description="Whether result was from cache")
    matched_query: Optional[str] = Field(
        default=None,
        description="Matched cached query (if cache hit)"
    )
    similarity_score: float = Field(
        ...,
        description="Similarity to cached query"
    )
    result: str = Field(..., description="Query result")
    dominant_cluster: int = Field(..., description="Query's primary cluster ID")
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )


class CacheStatsResponse(BaseModel):
    """Schema for cache statistics."""
    
    total_entries: int = Field(..., description="Total cached entries")
    total_queries: int = Field(..., description="Total queries processed")
    hit_count: int = Field(..., description="Number of cache hits")
    miss_count: int = Field(..., description="Number of cache misses")
    hit_rate: float = Field(..., description="Cache hit rate (percentage)")
    miss_rate: float = Field(..., description="Cache miss rate (percentage)")
    similarity_threshold: float = Field(
        ...,
        description="Similarity threshold for cache hits"
    )
    use_clustering: bool = Field(
        ...,
        description="Whether cluster optimization is enabled"
    )


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component status")


class MessageResponse(BaseModel):
    """Generic message response."""
    
    message: str = Field(..., description="Response message")
    success: bool = Field(default=True, description="Operation success status")
