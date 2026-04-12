from typing import Dict, List, Set


class Grader:
    """
    Stateful per-episode scorer.

    Reward structure:
    ─────────────────────────────────────────────────────────────────────
    Action              Reward
    ─────────────────────────────────────────────────────────────────────
    FLAG correct bot    +0.10 per account (small incremental signal)
    FLAG false-pos      -0.15 per account (penalise noise hard)
    QUERY new node       +0.05 first visit; -0.02 revisit; -0.30 ≥3rd visit
    REQUEST_TIMESERIES   +0.02 flat (information gathering)
    SUBMIT_REPORT        F1-based bonus (see below)
    Budget exhausted     -0.10 (episodic penalty, terminates episode)
    ─────────────────────────────────────────────────────────────────────

    SUBMIT_REPORT bonus (added to final step reward):
        F1 ≥ 0.90  →  +0.50
        F1 ≥ 0.75  →  +0.30
        F1 ≥ 0.50  →  +0.15
        F1  < 0.50 →   0.00
        Completed with steps to spare (< max_steps used) → +0.10 efficiency bonus
    """

    def __init__(self, ground_truth: Dict[str, bool], max_steps: int):
        self.ground_truth = ground_truth
        self.max_steps = max_steps
        self.bot_ids: Set[str] = {k for k, v in ground_truth.items() if v}

        self.flagged_accounts: Set[str] = set()
        self.queried_nodes: Set[str] = set()
        self.query_counts: Dict[str, int] = {}

        self.steps_taken = 0
        self.is_done = False

    # ------------------------------------------------------------------
    # Per-action scorers
    # ------------------------------------------------------------------

    def score_flag_action(self, target_ids: List[str]) -> float:
        """Score FLAG_ACCOUNT action; penalises false positives."""
        reward = 0.0
        for tid in target_ids:
            if tid in self.flagged_accounts:
                continue  # idempotent – already counted
            self.flagged_accounts.add(tid)
            if self.ground_truth.get(tid, False):
                reward += 0.10  # correct identification
            else:
                reward -= 0.15  # false positive penalty
        return reward

    def score_query_action(self, target_id: str) -> float:
        """Score QUERY_NEIGHBORHOOD for a single node."""
        count = self.query_counts.get(target_id, 0) + 1
        self.query_counts[target_id] = count

        if count >= 3:
            return -0.30  # loop-detection penalty

        if target_id in self.queried_nodes:
            return -0.02  # redundant query

        self.queried_nodes.add(target_id)
        return 0.05  # new signal discovered

    def score_timeseries_action(self) -> float:
        """Score REQUEST_TIMESERIES (flat information-gathering reward)."""
        return 0.02

    def score_submit_action(self) -> float:
        """
        Terminal scorer. Computes F1 over all flagged vs. ground-truth bots
        and returns a scaled bonus. Sets `is_done = True`.
        """
        self.is_done = True

        true_pos = sum(1 for a in self.flagged_accounts if self.ground_truth.get(a, False))
        false_pos = len(self.flagged_accounts) - true_pos
        false_neg = len(self.bot_ids) - true_pos

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        reward = 0.0
        if f1 >= 0.90:
            reward += 0.50
        elif f1 >= 0.75:
            reward += 0.30
        elif f1 >= 0.50:
            reward += 0.15

        # Efficiency bonus for finishing without exhausting budget
        if self.steps_taken < self.max_steps:
            reward += 0.10

        return reward

    def check_exhaustion(self) -> float:
        """
        Returns a penalty and marks the episode done if step budget is exhausted
        without an explicit SUBMIT_REPORT.
        """
        if self.steps_taken >= self.max_steps and not self.is_done:
            self.is_done = True
            return -0.10
        return 0.0

    def compute_f1(self) -> Dict[str, float]:
        """Utility: compute current precision/recall/F1 without terminating."""
        true_pos = sum(1 for a in self.flagged_accounts if self.ground_truth.get(a, False))
        false_pos = len(self.flagged_accounts) - true_pos
        false_neg = len(self.bot_ids) - true_pos
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1,
                "true_pos": true_pos, "false_pos": false_pos, "false_neg": false_neg}

    def step(self):
        """Advance the internal step counter."""
        self.steps_taken += 1
