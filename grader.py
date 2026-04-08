from typing import Dict, List, Set

class Grader:
    def __init__(self, ground_truth: Dict[str, bool], max_steps: int):
        self.ground_truth = ground_truth
        self.max_steps = max_steps
        self.bot_count = sum(1 for v in ground_truth.values() if v)
        
        self.flagged_accounts: Set[str] = set()
        self.queried_subgraphs: List[str] = []
        self.query_counts: Dict[str, int] = {}
        
        self.steps_taken = 0
        self.is_done = False
        
    def score_flag_action(self, target_ids: List[str]) -> float:
        reward = 0.0
        for tid in target_ids:
            if tid in self.flagged_accounts:
                continue # Ignore already flagged
            self.flagged_accounts.add(tid)
            if self.ground_truth.get(tid, False):
                reward += 0.15 # Correct FLAG
            else:
                reward -= 0.20 # False Positive
        return reward
        
    def score_query_action(self, target_id: str) -> float:
        self.query_counts[target_id] = self.query_counts.get(target_id, 0) + 1
        
        if self.query_counts[target_id] >= 3:
            return -0.30 # Infinite loop detected (same query x3)
            
        if target_id in self.queried_subgraphs:
            return -0.02 # Redundant QUERY
            
        self.queried_subgraphs.append(target_id)
        return 0.05 # QUERY new signal
        
    def score_submit_action(self) -> float:
        self.is_done = True
        reward = 0.0
        
        # Calculate F1
        true_pos = sum(1 for a in self.flagged_accounts if self.ground_truth.get(a, False))
        false_pos = len(self.flagged_accounts) - true_pos
        false_neg = self.bot_count - true_pos
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 >= 0.8:
            reward += 0.30
            
        if self.steps_taken < self.max_steps:
            reward += 0.10
        
        # Ensure base cumulative reward + this final reward is within (0, 1)
        # We handle the clamping in the environment/inference layer, but we can set 
        # a baseline here to help.
        return reward
        
    def check_exhaustion(self) -> float:
        if self.steps_taken >= self.max_steps and not self.is_done:
            self.is_done = True
            return -0.15
        return 0.0
    
    def step(self):
        self.steps_taken += 1
