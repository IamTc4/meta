"""
graph_generator.py – Deterministic synthetic social-graph generator.

All randomness is driven entirely by a fixed seed so that the same task_id
always produces bit-identical graphs. This ensures reproducible evaluation.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Tuple

import networkx as nx
from faker import Faker

# Interaction types available in the simulation
_INTERACTION_TYPES = ["follow", "retweet", "mention", "like"]

# Fixed epoch so temporal attributes are reproducible
_EPOCH = datetime(2024, 1, 1)


class GraphGenerator:
    """
    Produces synthetic social graphs with embedded CIB clusters.

    Design principles:
    - Pure determinism: seeded RNG reset before every generate_task call.
    - Faker used for realistic-looking (but fake) dates.
    - Bot / inauthentic accounts are labelled in the returned ground_truth dict.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)
        self._fake = Faker()
        Faker.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_task(self, task_id: str) -> Tuple[nx.DiGraph, Dict[str, bool]]:
        """
        Generate graph + ground truth for the given task.

        Returns
        -------
        graph : nx.DiGraph
            Nodes and edges fully populated with attributes.
        ground_truth : Dict[str, bool]
            Mapping account_id → True if inauthentic, False if organic.
        """
        # Hard-reset RNG each time so calling generate_task twice is idempotent.
        self._rng = random.Random(self.seed)
        self._fake = Faker()
        Faker.seed(self.seed)

        if task_id == "task_01":
            return self._task_01()
        elif task_id == "task_02":
            return self._task_02()
        elif task_id == "task_03":
            return self._task_03()
        else:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid values: task_01, task_02, task_03")

    def populate_attributes(self, G: nx.DiGraph) -> None:
        """
        Attach realistic account and edge attributes to every node/edge in G.
        Called after generate_task so caller can inspect before attribute inflation.
        """
        self._attach_node_attrs(G)
        self._attach_edge_attrs(G)

    # ------------------------------------------------------------------
    # Task generators
    # ------------------------------------------------------------------

    def _task_01(self) -> Tuple[nx.DiGraph, Dict[str, bool]]:
        """
        Task 01 – Bot Farm Detection (Easy).

        450 organic accounts in a sparse random graph.
        50 bots forming a dense clique that also follows organics.
        Clear structural signal: high mutual in-degree within bot cluster.
        """
        rng = self._rng
        G = nx.DiGraph()

        # ---- Organic accounts ----
        organic_ids = [f"ACC_ORG_{i:03d}" for i in range(450)]
        G.add_nodes_from(organic_ids)

        # Sparse Erdős–Rényi edges among organics (deterministic)
        for i, u in enumerate(organic_ids):
            for j, v in enumerate(organic_ids):
                if i != j and rng.random() < 0.015:
                    G.add_edge(u, v, interaction_type="follow", weight=1.0)

        # ---- Bot farm ----
        bot_ids = [f"ACC_BOT_{i:03d}" for i in range(50)]
        G.add_nodes_from(bot_ids)

        # Bots form a dense clique (~80 % connectivity)
        for b1 in bot_ids:
            for b2 in bot_ids:
                if b1 != b2 and rng.random() < 0.80:
                    G.add_edge(b1, b2, interaction_type="follow", weight=2.0)

        # Each bot follows a random subset of organic accounts
        for b in bot_ids:
            targets = rng.sample(organic_ids, k=10)
            for t in targets:
                G.add_edge(b, t, interaction_type="follow", weight=1.0)

        ground_truth = {n: (n in set(bot_ids)) for n in G.nodes()}
        return G, ground_truth

    def _task_02(self) -> Tuple[nx.DiGraph, Dict[str, bool]]:
        """
        Task 02 – Astroturfing Ring Identification (Medium).

        1 910 organic accounts.
        3 astroturfing rings of 30 accounts each; rings interact via retweets
        in bursts but look organic otherwise.
        """
        rng = self._rng
        G = nx.DiGraph()

        # ---- Organic accounts ----
        organic_ids = [f"ACC_ORG_{i:04d}" for i in range(1910)]
        G.add_nodes_from(organic_ids)

        for i, u in enumerate(organic_ids):
            for j, v in enumerate(organic_ids):
                if i != j and rng.random() < 0.005:
                    G.add_edge(u, v, interaction_type=rng.choice(_INTERACTION_TYPES), weight=1.0)

        # ---- Three astroturfing rings ----
        astro_ids: list[str] = []
        for ring_id in range(3):
            ring = [f"ACC_ASTRO_R{ring_id}_{i:02d}" for i in range(30)]
            G.add_nodes_from(ring)
            astro_ids.extend(ring)

            # Intra-ring retweet bursts (coordinated amplification)
            for actor in ring:
                targets = rng.sample([m for m in ring if m != actor], k=min(8, len(ring) - 1))
                for t in targets:
                    G.add_edge(actor, t, interaction_type="retweet", weight=2.5)

            # Each ring member also follows some organic accounts (camouflage)
            for actor in ring:
                follows = rng.sample(organic_ids, k=15)
                for t in follows:
                    G.add_edge(actor, t, interaction_type="follow", weight=1.0)

        ground_truth = {n: (n in set(astro_ids)) for n in G.nodes()}
        return G, ground_truth

    def _task_03(self) -> Tuple[nx.DiGraph, Dict[str, bool]]:
        """
        Task 03 – Adversarial CIB with Infiltration (Hard).

        900 organic accounts.
        100 sophisticated infiltrators that mimic organic follow behaviour
        but exhibit coordinated burst activity in narrow temporal windows.
        Requires temporal analysis to distinguish from legitimates.
        """
        rng = self._rng
        G = nx.DiGraph()

        # ---- Organic accounts ----
        organic_ids = [f"ACC_ORG_{i:04d}" for i in range(900)]
        G.add_nodes_from(organic_ids)

        for i, u in enumerate(organic_ids):
            for j, v in enumerate(organic_ids):
                if i != j and rng.random() < 0.02:
                    G.add_edge(u, v, interaction_type=rng.choice(_INTERACTION_TYPES), weight=1.0)

        # ---- Infiltrators ----
        infiltrator_ids = [f"ACC_SOP_{i:03d}" for i in range(100)]
        G.add_nodes_from(infiltrator_ids)

        for inf in infiltrator_ids:
            # Looks organic: follows many real accounts
            follows = rng.sample(organic_ids, k=20)
            for t in follows:
                G.add_edge(inf, t, interaction_type="follow", weight=1.0)

            # Coordinated burst: each infiltrator mentions every other in a tightly scoped ring
            burst_targets = rng.sample(infiltrator_ids, k=min(5, len(infiltrator_ids) - 1))
            for t in burst_targets:
                if t != inf:
                    G.add_edge(inf, t, interaction_type="mention", weight=3.0)

        ground_truth = {n: (n in set(infiltrator_ids)) for n in G.nodes()}
        return G, ground_truth

    # ------------------------------------------------------------------
    # Attribute inflation helpers
    # ------------------------------------------------------------------

    def _attach_node_attrs(self, G: nx.DiGraph) -> None:
        rng = self._rng
        for n in G.nodes():
            in_deg = G.in_degree(n)
            out_deg = G.out_degree(n)
            created = _EPOCH - timedelta(days=rng.randint(30, 5 * 365))
            G.nodes[n].setdefault("creation_date", created.isoformat())
            G.nodes[n].setdefault("verified", rng.random() < 0.05)
            G.nodes[n].setdefault("follower_count", max(0, in_deg * rng.randint(1, 100)))
            G.nodes[n].setdefault("following_count", max(0, out_deg * rng.randint(1, 10)))
            G.nodes[n].setdefault("feature_vector", [rng.random() for _ in range(8)])

    def _attach_edge_attrs(self, G: nx.DiGraph) -> None:
        rng = self._rng
        for u, v in G.edges():
            # Spread edge timestamps reproducibly across the year
            day_offset = rng.randint(0, 364)
            hour_offset = rng.randint(0, 23)
            ts = (_EPOCH + timedelta(days=day_offset, hours=hour_offset)).isoformat()
            G.edges[u, v].setdefault("interaction_type", rng.choice(_INTERACTION_TYPES))
            G.edges[u, v].setdefault("weight", 1.0)
            G.edges[u, v].setdefault("timestamp", ts)
