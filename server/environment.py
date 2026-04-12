"""
server/environment.py – OpenEnv-compliant SocialGraphEnv.

Inherits from openenv.core.env_server.Environment and wires up:
- reset()  → GraphObservation  (initialises episode state, no side effects)
- step()   → (GraphObservation, float, bool, dict)  (ALL code paths)
- state    → GraphObservation  (pure read, zero side effects)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Set, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Make root package importable when run from the server/ sub-directory
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openenv.core.env_server import Environment  # type: ignore

from graph_generator import GraphGenerator
from grader import Grader
from models import (
    AccountNode,
    ActionType,
    EdgeRecord,
    GraphObservation,
    GraphStats,
    InvestigationAction,
    PostRecord,
    TimeWindow,
)

# ---------------------------------------------------------------------------
# Fixed temporal windows used across all tasks (non-random, deterministic)
# ---------------------------------------------------------------------------
_TEMPORAL_WINDOWS = [
    TimeWindow(
        start_time="2024-01-01T00:00:00",
        end_time="2024-01-01T06:00:00",
        active_account_ids=[],
        interaction_count=0,
    ),
    TimeWindow(
        start_time="2024-01-01T06:00:00",
        end_time="2024-01-01T12:00:00",
        active_account_ids=[],
        interaction_count=0,
    ),
    TimeWindow(
        start_time="2024-01-01T12:00:00",
        end_time="2024-01-01T18:00:00",
        active_account_ids=[],
        interaction_count=0,
    ),
    TimeWindow(
        start_time="2024-01-01T18:00:00",
        end_time="2024-01-02T00:00:00",
        active_account_ids=[],
        interaction_count=0,
    ),
]

# Number of initial nodes revealed without any query action
_INITIAL_REVEAL = 10

# Max steps per task (budget)
_MAX_STEPS: Dict[str, int] = {
    "task_01": 20,
    "task_02": 40,
    "task_03": 80,
}


class SocialGraphEnv(Environment):
    """
    Social Graph Manipulation Detection environment.

    The agent investigates a synthetic social graph to identify coordinated
    inauthentic behaviour (CIB).  Three tasks of increasing difficulty are
    supported via the `task_id` constructor parameter.

    Episode lifecycle
    -----------------
    1. reset() → reveals 10 seed accounts, returns initial observation.
    2. step(action) → processes the action, returns (obs, reward, done, info).
    3. Episode ends when:
       - Agent calls SUBMIT_REPORT, or
       - Step budget is exhausted.

    Observation / Action types:  GraphObservation / InvestigationAction
    """

    def __init__(self, task_id: str = "task_01", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self.max_steps: int = _MAX_STEPS.get(task_id, 20)

        self._generator = GraphGenerator(seed=seed)

        # Mutable episode state (initialised by reset)
        self._graph: Optional[nx.DiGraph] = None
        self._ground_truth: Dict[str, bool] = {}
        self._grader: Optional[Grader] = None
        self._queried_nodes: Set[str] = set()
        self._post_cache: Dict[str, PostRecord] = {}

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(self) -> GraphObservation:
        """
        Initialise a new episode for the current task.

        Generates a fresh graph, resets all tracking state, reveals the
        first `_INITIAL_REVEAL` seed accounts, and returns the opening
        observation (no reward, not done).
        """
        self._graph, self._ground_truth = self._generator.generate_task(self.task_id)
        self._generator.populate_attributes(self._graph)
        self._grader = Grader(self._ground_truth, self.max_steps)
        self._queried_nodes = set()
        self._post_cache = {}

        # Deterministically reveal the first N nodes from the sorted node list
        initial_nodes = sorted(self._graph.nodes())[:_INITIAL_REVEAL]
        self._queried_nodes.update(initial_nodes)

        # Pre-build the post cache for deterministic posts
        self._build_post_cache()

        return self.state

    def step(
        self, action: InvestigationAction
    ) -> Tuple[GraphObservation, float, bool, Dict[str, Any]]:
        """
        Process one action and return (observation, reward, done, info).

        All code paths MUST return the full 4-tuple – no early exit that
        returns only `self.state`.
        """
        # Auto-reset if called before reset() (defensive)
        if self._grader is None:
            self.reset()

        # Already done – return terminal state with zero reward
        if self._grader.is_done:
            obs = self.state
            return obs, 0.0, True, {"msg": "Episode already finished", "step": self._grader.steps_taken}

        # Advance step counter
        self._grader.step()
        reward = 0.0

        # ── Process action ──────────────────────────────────────────────
        if action.action_type == ActionType.FLAG_ACCOUNT:
            if action.target_ids:
                reward += self._grader.score_flag_action(action.target_ids)

        elif action.action_type == ActionType.QUERY_NEIGHBORHOOD:
            for tid in action.target_ids:
                if tid in self._graph:
                    reward += self._grader.score_query_action(tid)
                    # Expand the visible subgraph
                    neighbors = (
                        list(self._graph.successors(tid))
                        + list(self._graph.predecessors(tid))
                    )
                    self._queried_nodes.update(neighbors)
                    self._queried_nodes.add(tid)

        elif action.action_type == ActionType.REQUEST_TIMESERIES:
            reward += self._grader.score_timeseries_action()

        elif action.action_type == ActionType.SUBMIT_REPORT:
            reward += self._grader.score_submit_action()

        # ── Check step-budget exhaustion ────────────────────────────────
        if not self._grader.is_done:
            exhaust_penalty = self._grader.check_exhaustion()
            reward += exhaust_penalty

        done = self._grader.is_done
        f1_info = self._grader.compute_f1()
        info: Dict[str, Any] = {
            "step": self._grader.steps_taken,
            "flagged_accounts": sorted(self._grader.flagged_accounts),
            **f1_info,
        }

        obs = self._build_observation(step_reward=reward)
        return obs, reward, done, info

    @property
    def state(self) -> GraphObservation:
        """
        Pure read of the current observation.

        No reward computation, no randomness, no side effects.
        Safe to call at any point without altering episode state.
        """
        if self._graph is None:
            # Pre-reset fallback – return an empty but valid observation
            return GraphObservation(
                nodes=[],
                edges=[],
                posts=[],
                temporal_windows=list(_TEMPORAL_WINDOWS),
                graph_stats=GraphStats(),
                step_budget=self.max_steps,
                reward=0.0,
                done=False,
                info={},
            )
        return self._build_observation(step_reward=0.0)

    # ------------------------------------------------------------------
    # Async shims (required for some OpenEnv runtimes)
    # ------------------------------------------------------------------

    async def reset_async(self) -> GraphObservation:
        return self.reset()

    async def step_async(
        self, action: InvestigationAction
    ) -> Tuple[GraphObservation, float, bool, Dict[str, Any]]:
        return self.step(action)

    async def state_async(self) -> GraphObservation:
        return self.state

    def close(self) -> None:
        """Release resources (no-op here)."""
        pass

    # ------------------------------------------------------------------
    # Private helpers – pure functions, no mutation except _post_cache
    # ------------------------------------------------------------------

    def _build_observation(self, step_reward: float = 0.0) -> GraphObservation:
        """Construct a GraphObservation from the current visible subgraph."""
        assert self._graph is not None

        subgraph = self._graph.subgraph(self._queried_nodes)

        nodes = []
        for n, data in subgraph.nodes(data=True):
            nodes.append(
                AccountNode(
                    id=n,
                    creation_date=data.get("creation_date", ""),
                    verified=data.get("verified", False),
                    follower_count=data.get("follower_count", 0),
                    following_count=data.get("following_count", 0),
                    feature_vector=data.get("feature_vector", []),
                )
            )

        edges = []
        for u, v, data in subgraph.edges(data=True):
            edges.append(
                EdgeRecord(
                    source_id=u,
                    target_id=v,
                    interaction_type=data.get("interaction_type", "follow"),
                    timestamp=data.get("timestamp", ""),
                    weight=data.get("weight", 1.0),
                )
            )

        # Global graph statistics (always visible, computed once from full graph)
        stats = GraphStats(
            total_nodes=self._graph.number_of_nodes(),
            total_edges=self._graph.number_of_edges(),
            density=round(nx.density(self._graph), 6),
            clustering_coefficient=0.0,  # intentionally skipped – too expensive
            community_count=0,
        )

        remaining_budget = (
            self.max_steps - self._grader.steps_taken if self._grader else self.max_steps
        )

        # Posts from the post cache (deterministic, built at reset)
        visible_posts = [
            self._post_cache[n]
            for n in self._queried_nodes
            if n in self._post_cache
        ]

        # Fill temporal windows with observed active accounts
        temporal_windows = self._build_temporal_windows(edges)

        return GraphObservation(
            nodes=nodes,
            edges=edges,
            posts=visible_posts,
            temporal_windows=temporal_windows,
            graph_stats=stats,
            step_budget=remaining_budget,
            reward=step_reward,
            done=self._grader.is_done if self._grader else False,
            info={"step": self._grader.steps_taken} if self._grader else {},
        )

    def _build_post_cache(self) -> None:
        """
        Pre-generates one deterministic PostRecord per node using the
        seeded GraphGenerator RNG.  Called once at reset().
        """
        assert self._graph is not None
        rng = self._generator._rng  # use the already-seeded generator RNG
        topics_pool = ["spam", "politics", "crypto", "health", "finance", "sports"]
        for n, data in self._graph.nodes(data=True):
            # Only ~60 % of accounts have visible posts
            if rng.random() > 0.40:
                continue
            post_id = f"POST_{n}_{rng.randint(1000, 9999)}"
            n_topics = rng.randint(1, 3)
            chosen_topics = rng.sample(topics_pool, k=n_topics)
            self._post_cache[n] = PostRecord(
                post_id=post_id,
                author_id=n,
                content_hash=f"HASH_{rng.randint(10000, 99999)}",
                topics=chosen_topics,
                timestamp=data.get("creation_date", "2024-01-01T00:00:00"),
                engagement_score=round(rng.random(), 4),
            )

    def _build_temporal_windows(self, edges: list[EdgeRecord]) -> list[TimeWindow]:
        """
        Maps visible edges into the four fixed 6-hour windows.
        Avoids any randomness – pure function over the edge list.
        """
        window_buckets: Dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
        interaction_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        for edge in edges:
            try:
                # Parse hour from ISO timestamp  (e.g. "2024-01-15T14:30:00")
                hour = int(edge.timestamp[11:13]) if len(edge.timestamp) >= 13 else 0
            except (ValueError, IndexError):
                hour = 0

            bucket = min(hour // 6, 3)
            window_buckets[bucket].append(edge.source_id)
            window_buckets[bucket].append(edge.target_id)
            interaction_counts[bucket] += 1

        result = []
        for i, tw in enumerate(_TEMPORAL_WINDOWS):
            unique_accounts = sorted(set(window_buckets[i]))[:20]  # cap for payload size
            result.append(
                TimeWindow(
                    start_time=tw.start_time,
                    end_time=tw.end_time,
                    active_account_ids=unique_accounts,
                    interaction_count=interaction_counts[i],
                )
            )
        return result
