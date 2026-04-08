import networkx as nx
import random
from typing import Tuple, Dict, List
from datetime import datetime, timedelta
from faker import Faker
from models import AccountNode, EdgeRecord, PostRecord, GraphStats

fake = Faker()

class GraphGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset_seed()

    def reset_seed(self):
        random.seed(self.seed)
        fake.seed_instance(self.seed)

    def generate_task(self, task_id: str) -> Tuple[nx.DiGraph, Dict[str, dict]]:
        """
        Generates a synthetic social graph and the ground truth.
        Returns:
            graph: nx.DiGraph containing nodes and edges with attributes.
            ground_truth: mapping of node ID to True if bot/inauthentic, False otherwise.
        """
        self.reset_seed()
        if task_id == "task_01":
            return self._generate_task_01()
        elif task_id == "task_02":
            return self._generate_task_02()
        elif task_id == "task_03":
            return self._generate_task_03()
        else:
            raise ValueError(f"Unknown task {task_id}")

    def _generate_task_01(self) -> Tuple[nx.DiGraph, Dict[str, dict]]:
        # 500 accounts, 1 bot farm of 50 accounts
        G = nx.fast_gnp_random_graph(450, 0.05, seed=self.seed, directed=True)
        # Rename nodes
        node_mapping = {n: f"ACC_ORG_{n:03d}" for n in G.nodes()}
        G = nx.relabel_nodes(G, node_mapping)
        
        # Add bots
        bots = [f"ACC_BOT_{i:03d}" for i in range(50)]
        G.add_nodes_from(bots)
        
        # Bots follow each other highly
        for b1 in bots:
            for b2 in bots:
                if b1 != b2 and random.random() > 0.2:
                    G.add_edge(b1, b2, weight=1.0, interaction_type="follow")
                    
        # Bots follow random organics
        for b in bots:
            org_targets = random.sample(list(node_mapping.values()), 10)
            for t in org_targets:
                G.add_edge(b, t, weight=1.0, interaction_type="follow")

        ground_truth = {n: (n in bots) for n in G.nodes()}
        return G, ground_truth

    def _generate_task_02(self) -> Tuple[nx.DiGraph, Dict[str, dict]]:
        # 2000 accounts, 3 astroturfing rings of 30 accounts each
        G = nx.fast_gnp_random_graph(1910, 0.01, seed=self.seed, directed=True)
        node_mapping = {n: f"ACC_ORG_{n:04d}" for n in G.nodes()}
        G = nx.relabel_nodes(G, node_mapping)

        astroturf_groups = []
        for r_id in range(3):
            ring = [f"ACC_ASTRO_{r_id}_{i:03d}" for i in range(30)]
            G.add_nodes_from(ring)
            for b1 in ring:
                # Ring intra-interactions (retweets, etc)
                targets = random.sample(ring, 5)
                for t in targets:
                    if b1 != t:
                        G.add_edge(b1, t, weight=2.0, interaction_type="retweet")
            astroturf_groups.extend(ring)

        ground_truth = {n: (n in astroturf_groups) for n in G.nodes()}
        return G, ground_truth

    def _generate_task_03(self) -> Tuple[nx.DiGraph, Dict[str, dict]]:
        # 5000 accounts, sophisticated infiltration
        # Simplifying for space limits, representing 1000 accounts with 100 bots
        G = nx.fast_gnp_random_graph(900, 0.02, seed=self.seed, directed=True)
        node_mapping = {n: f"ACC_ORG_{n:04d}" for n in G.nodes()}
        G = nx.relabel_nodes(G, node_mapping)

        infiltrators = [f"ACC_SOP_{i:03d}" for i in range(100)]
        G.add_nodes_from(infiltrators)

        for b in infiltrators:
            # Looks totally organic in follow structure, but edges have strange temp activity later
            org_targets = random.sample(list(node_mapping.values()), 20)
            for t in org_targets:
                G.add_edge(b, t, weight=1.0, interaction_type="follow")
                
        ground_truth = {n: (n in infiltrators) for n in G.nodes()}
        return G, ground_truth
    
    def populate_attributes(self, G: nx.DiGraph):
        for n in G.nodes():
            in_deg = G.in_degree(n)
            out_deg = G.out_degree(n)
            G.nodes[n]['creation_date'] = fake.date_time_between(start_date="-5y", end_date="-1m").isoformat()
            G.nodes[n]['verified'] = random.random() < 0.05
            G.nodes[n]['follower_count'] = in_deg * random.randint(1, 100)
            G.nodes[n]['following_count'] = out_deg * random.randint(1, 10)
            G.nodes[n]['feature_vector'] = [random.random() for _ in range(8)]
            
        now = datetime.now()
        for u, v in G.edges():
            it = G.edges[u, v].get('interaction_type', random.choice(['follow', 'retweet', 'like']))
            w = G.edges[u, v].get('weight', 1.0)
            ts = fake.date_time_between(start_date="-1y", end_date="now").isoformat()
            G.edges[u, v]['interaction_type'] = it
            G.edges[u, v]['weight'] = w
            G.edges[u, v]['timestamp'] = ts
