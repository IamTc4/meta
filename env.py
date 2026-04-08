from typing import Tuple, Optional, Dict, Any
from models import GraphObservation, InvestigationAction, ActionType, AccountNode, EdgeRecord, GraphStats
from graph_generator import GraphGenerator
from grader import Grader
import networkx as nx

class SocialGraphEnv:
    def __init__(self, task_id: str = "task_01", seed: int = 42):
        self.task_id = task_id
        self.generator = GraphGenerator(seed=seed)
        self.graph: Optional[nx.DiGraph] = None
        self.ground_truth: Dict[str, bool] = {}
        self.grader: Optional[Grader] = None
        self.max_steps = 20 if task_id == "task_01" else (40 if task_id == "task_02" else 80)
        self.queried_nodes = set()
        
    def reset(self) -> GraphObservation:
        """
        Resets the environment for the current task.
        Returns the initial observation space.
        """
        self.graph, self.ground_truth = self.generator.generate_task(self.task_id)
        self.generator.populate_attributes(self.graph)
        self.grader = Grader(self.ground_truth, self.max_steps)
        self.queried_nodes = set()
        
        # Start by revealing a random set of 10 generic accounts
        start_nodes = list(self.graph.nodes())[:10]
        self.queried_nodes.update(start_nodes)
        
        return self.state()

    def step(self, action: InvestigationAction) -> Tuple[GraphObservation, float, bool, Dict[str, Any]]:
        """
        Takes a step using the provided InvestigationAction.
        Returns (Observation, Reward, Done, Info).
        """
        if self.grader.is_done:
            return self.state(), 0.0, True, {"msg": "Episode already finished"}

        self.grader.step()
        reward = 0.0
        
        # Process the action
        if action.action_type == ActionType.FLAG_ACCOUNT:
            if action.target_ids:
                reward += self.grader.score_flag_action(action.target_ids)
                
        elif action.action_type == ActionType.QUERY_NEIGHBORHOOD:
            for tid in action.target_ids:
                if tid in self.graph:
                    reward += self.grader.score_query_action(tid)
                    # Reveal neighborhood
                    neighbors = list(self.graph.successors(tid)) + list(self.graph.predecessors(tid))
                    self.queried_nodes.update(neighbors)
                    self.queried_nodes.add(tid)
                    
        elif action.action_type == ActionType.REQUEST_TIMESERIES:
            # We don't have deep time series implemented in the mock, just giving a generic reward
            reward += 0.01 
            
        elif action.action_type == ActionType.SUBMIT_REPORT:
            reward += self.grader.score_submit_action()
            
        # Check budget limit if not already done
        if not self.grader.is_done:
            exhaust_penalty = self.grader.check_exhaustion()
            reward += exhaust_penalty
            
        info = {
            "flagged_accounts": list(self.grader.flagged_accounts),
            "step": self.grader.steps_taken
        }
        
        return self.state(), reward, self.grader.is_done, info

    def state(self) -> GraphObservation:
        """
        Returns the current state representation built from visible nodes and subgraphs.
        """
        subgraph = self.graph.subgraph(self.queried_nodes)
        
        nodes, edges = [], []
        for n, data in subgraph.nodes(data=True):
            nodes.append(AccountNode(
                id=n,
                creation_date=data.get("creation_date", ""),
                verified=data.get("verified", False),
                follower_count=data.get("follower_count", 0),
                following_count=data.get("following_count", 0),
                feature_vector=data.get("feature_vector", [])
            ))
            
        for u, v, data in subgraph.edges(data=True):
            edges.append(EdgeRecord(
                source_id=u,
                target_id=v,
                interaction_type=data.get("interaction_type", "follow"),
                timestamp=data.get("timestamp", ""),
                weight=data.get("weight", 1.0)
            ))
            
        stats = GraphStats(
            total_nodes=len(self.graph.nodes()),
            total_edges=len(self.graph.edges()),
            density=nx.density(self.graph),
            clustering_coefficient=0.0, # skip heavy compute in state call
            community_count=0
        )
        
        remaining_budget = self.max_steps - self.grader.steps_taken if self.grader else self.max_steps
        
        return GraphObservation(
            nodes=nodes,
            edges=edges,
            posts=[],
            temporal_windows=[],
            graph_stats=stats,
            step_budget=remaining_budget
        )
