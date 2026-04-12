from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

# ==========================================
# SUB-MODELS (nested within GraphObservation)
# ==========================================

class AccountNode(BaseModel):
    id: str = Field(..., description="Unique identifier for the account")
    creation_date: str = Field("", description="ISO 8601 timestamp of account creation")
    verified: bool = Field(False, description="Whether the account is verified")
    follower_count: int = Field(0, description="Number of followers")
    following_count: int = Field(0, description="Number of accounts this user follows")
    feature_vector: List[float] = Field(
        default_factory=list,
        description="Arbitrary feature vector (e.g., embeddings of user bio/metadata)"
    )


class EdgeRecord(BaseModel):
    source_id: str = Field(..., description="ID of the account initiating the interaction")
    target_id: str = Field(..., description="ID of the account receiving the interaction")
    interaction_type: str = Field("follow", description="Type of interaction: follow, retweet, mention, like")
    timestamp: str = Field("", description="ISO 8601 timestamp of the interaction")
    weight: float = Field(1.0, description="Interaction weight")


class PostRecord(BaseModel):
    post_id: str = Field(..., description="Unique post identifier")
    author_id: str = Field(..., description="Account ID of the author")
    content_hash: str = Field(..., description="Hash representing the content to detect dupes")
    topics: List[str] = Field(default_factory=list, description="List of topics detected in the post")
    timestamp: str = Field("", description="ISO 8601 timestamp of the post")
    engagement_score: float = Field(0.0, description="Aggregate engagement metric")


class TimeWindow(BaseModel):
    start_time: str = Field(..., description="ISO 8601 start time")
    end_time: str = Field(..., description="ISO 8601 end time")
    active_account_ids: List[str] = Field(default_factory=list, description="Accounts active in this window")
    interaction_count: int = Field(0, description="Total interactions during window")


class GraphStats(BaseModel):
    total_nodes: int = Field(0)
    total_edges: int = Field(0)
    density: float = Field(0.0)
    clustering_coefficient: float = Field(0.0)
    community_count: int = Field(0)


# ==========================================
# OBSERVATION MODEL
# ==========================================

class GraphObservation(BaseModel):
    """
    Full observation emitted by the Social Graph environment each step.

    Nodes, edges, and posts represent the *currently visible* subgraph that
    the agent has explored via QUERY_NEIGHBORHOOD actions. Global graph
    statistics remain visible at all times for strategic planning.
    """
    nodes: List[AccountNode] = Field(default_factory=list, description="Accounts visible in the current graph snapshot")
    edges: List[EdgeRecord] = Field(default_factory=list, description="Connections visible in the current graph snapshot")
    posts: List[PostRecord] = Field(default_factory=list, description="Sampled posts from visible accounts")
    temporal_windows: List[TimeWindow] = Field(default_factory=list, description="Aggregated activity bursts")
    graph_stats: GraphStats = Field(default_factory=GraphStats, description="Global graph metrics (always visible)")
    step_budget: int = Field(0, description="Remaining investigation steps")
    reward: float = Field(0.0, description="Reward earned on the last step")
    done: bool = Field(False, description="Whether the episode is finished")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional environment info")


# ==========================================
# ACTION MODELS
# ==========================================

class ActionType(str, Enum):
    FLAG_ACCOUNT = "FLAG_ACCOUNT"
    QUERY_NEIGHBORHOOD = "QUERY_NEIGHBORHOOD"
    REQUEST_TIMESERIES = "REQUEST_TIMESERIES"
    SUBMIT_REPORT = "SUBMIT_REPORT"


class InvestigationAction(BaseModel):
    """
    Action issued by the agent each step.

    - FLAG_ACCOUNT: Label one or more accounts as inauthentic. Precision matters.
    - QUERY_NEIGHBORHOOD: Expand the visible subgraph around given account IDs.
    - REQUEST_TIMESERIES: Request a temporal activity breakdown (signals burst patterns).
    - SUBMIT_REPORT: Finalise the episode and trigger graded F1 scoring.
    """
    action_type: ActionType = Field(..., description="The type of investigation action to perform")
    target_ids: List[str] = Field(default_factory=list, description="Account IDs to flag or investigate")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Agent confidence in this action (used for FLAG_ACCOUNT scoring)")
    reasoning: str = Field("", description="Free-text justification (optional; scored for the hard task)")
    query_params: Optional[Dict[str, Any]] = Field(default=None, description="Additional query parameters")
