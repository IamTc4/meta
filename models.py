from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from datetime import datetime

# ==========================================
# OBSERVATION MODELS
# ==========================================

class AccountNode(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(..., description="Unique identifier for the account")
    creation_date: str = Field(..., description="ISO 8601 timestamp of account creation")
    verified: bool = Field(..., description="Whether the account is verified")
    follower_count: int = Field(..., description="Number of followers")
    following_count: int = Field(..., description="Number of accounts this user follows")
    feature_vector: List[float] = Field(
        default_factory=list,
        description="Arbitrary feature vector (e.g., embeddings of user bio/metadata)"
    )

class EdgeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_id: str = Field(..., description="ID of the account initiating the interaction")
    target_id: str = Field(..., description="ID of the account receiving the interaction")
    interaction_type: str = Field(..., description="Type of interaction: follow, retweet, mention, like")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the interaction")
    weight: float = Field(1.0, description="Interaction weight")

class PostRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    post_id: str = Field(..., description="Unique post identifier")
    author_id: str = Field(..., description="Account ID of the author")
    content_hash: str = Field(..., description="Hash representing the content to detect dupes")
    topics: List[str] = Field(..., description="List of topics detected in the post")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the post")
    engagement_score: float = Field(..., description="Aggregate engagement metric")

class TimeWindow(BaseModel):
    model_config = ConfigDict(frozen=True)
    start_time: str = Field(..., description="ISO 8601 start time")
    end_time: str = Field(..., description="ISO 8601 end time")
    active_account_ids: List[str] = Field(..., description="Accounts active in this window")
    interaction_count: int = Field(..., description="Total interactions during window")

class GraphStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    total_nodes: int
    total_edges: int
    density: float
    clustering_coefficient: float
    community_count: int

class GraphObservation(BaseModel):
    """
    Typed GraphObservation.
    Nodes, edges, posts may be a subset of the actual graph (discovered so far) or global representations.
    """
    model_config = ConfigDict(frozen=True)
    nodes: List[AccountNode] = Field(..., description="Accounts visible in the current graph snapshot")
    edges: List[EdgeRecord] = Field(..., description="Connections visible in the current graph snapshot")
    posts: List[PostRecord] = Field(default_factory=list, description="Sampled posts")
    temporal_windows: List[TimeWindow] = Field(default_factory=list, description="Aggregated activity bursts")
    graph_stats: GraphStats = Field(..., description="Global metrics")
    step_budget: int = Field(..., description="Remaining investigation steps")

# ==========================================
# ACTION MODELS
# ==========================================

class ActionType(str, Enum):
    FLAG_ACCOUNT = "FLAG_ACCOUNT"
    QUERY_NEIGHBORHOOD = "QUERY_NEIGHBORHOOD"
    REQUEST_TIMESERIES = "REQUEST_TIMESERIES"
    SUBMIT_REPORT = "SUBMIT_REPORT"

class InvestigationAction(BaseModel):
    action_type: ActionType
    target_ids: List[str] = Field(default_factory=list, description="Account IDs to flag or investigate")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in this flag (for FLAG_ACCOUNT)")
    reasoning: str = Field("", description="Free-text justification (optional but scored for hard task)")
    query_params: Optional[Dict[str, Any]] = Field(default=None, description="Additional query parameters")
