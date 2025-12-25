"""
Tunix Kaggle Hackathon - Reward Functions
==========================================
Composite reward for GRPO training: correctness + trace structure + confidence.
JAX-friendly, deterministic, and fast for batched rollouts.
"""

import re
from typing import List, Dict, Optional, Tuple

# JAX is optional (only needed for TPU training, not local eval)
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# =============================================================================
# Constants
# =============================================================================

TRANSITION_WORDS = [
    'therefore', 'thus', 'hence', 'so', 'because', 'first', 'second', 'third',
    'step', 'next', 'then', 'finally', 'since', 'given', 'considering',
    'we get', 'this means', 'it follows', 'consequently'
]

REASONING_TAG_PATTERN = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL | re.IGNORECASE)
ANSWER_TAG_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)

# Default reward weights (tunable)
DEFAULT_WEIGHTS = {
    'correctness': 0.60,
    'trace_structure': 0.25,
    'confidence': 0.15
}


# =============================================================================
# Core Reward Components
# =============================================================================

def extract_answer(text: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    match = ANSWER_TAG_PATTERN.search(text)
    return match.group(1).strip() if match else None


def extract_reasoning(text: str) -> Optional[str]:
    """Extract content from <reasoning>...</reasoning> tags."""
    match = REASONING_TAG_PATTERN.search(text)
    return match.group(1).strip() if match else None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip, remove units)."""
    ans = answer.lower().strip()
    # Remove common units and symbols
    ans = re.sub(r'\$|€|£|%|km/h|m/s|hours?|minutes?|seconds?|days?|years?', '', ans)
    ans = re.sub(r'\s+', ' ', ans).strip()
    return ans


def correctness_score(pred_text: str, ref_answer: str, tolerance: float = 1e-3) -> float:
    """
    Score correctness of the predicted answer against reference.
    
    Args:
        pred_text: Full model output containing <answer> tags
        ref_answer: Ground truth answer
        tolerance: Numeric tolerance for float comparison
        
    Returns:
        1.0 if correct, 0.0 otherwise
    """
    pred_ans = extract_answer(pred_text)
    if pred_ans is None:
        return 0.0
    
    pred_norm = normalize_answer(pred_ans)
    ref_norm = normalize_answer(ref_answer)
    
    # Try numeric comparison first
    try:
        pred_num = float(re.sub(r'[^\d.\-]', '', pred_norm))
        ref_num = float(re.sub(r'[^\d.\-]', '', ref_norm))
        return 1.0 if abs(pred_num - ref_num) <= tolerance else 0.0
    except (ValueError, TypeError):
        pass
    
    # Fall back to exact string match
    return 1.0 if pred_norm == ref_norm else 0.0


def trace_structure_score(text: str) -> float:
    """
    Score the quality of reasoning trace structure (judge-aligned).
    
    Scoring breakdown:
    - Presence of <reasoning> tags: +0.15
    - Presence of <answer> tags: +0.15
    - Number of reasoning steps: +0.40 (3+ steps = max)
    - Transition words (therefore, thus, etc): +0.20
    - Explicit "Step X" markers bonus: +0.10
    
    Rationale: Judges prefer clarity over verbosity. We reward multi-step
    reasoning without penalizing concise traces.
    
    Returns:
        Score between 0.0 and 1.0
    """
    score = 0.0
    
    # Check for required tags
    has_reasoning = REASONING_TAG_PATTERN.search(text) is not None
    has_answer = ANSWER_TAG_PATTERN.search(text) is not None
    
    if has_reasoning:
        score += 0.15
    if has_answer:
        score += 0.15
    
    # Extract reasoning content
    reasoning = extract_reasoning(text)
    if reasoning is None:
        return score  # No reasoning trace found
    
    # Count reasoning steps (sentences or explicit steps)
    # Split by period, newline, or "Step X:" patterns
    steps = re.split(r'[.\n]|step\s*\d+:', reasoning.lower())
    steps = [s.strip() for s in steps if len(s.strip()) > 10]
    num_steps = len(steps)
    
    # Reward 3+ steps, max contribution at 3+ steps (judge-aligned)
    step_score = min(1.0, num_steps / 3.0)
    score += 0.40 * step_score
    
    # Count transition words (therefore, thus, hence, etc.)
    reasoning_lower = reasoning.lower()
    transition_count = sum(1 for word in TRANSITION_WORDS if word in reasoning_lower)
    trans_score = min(1.0, transition_count / 2.0)  # Easier threshold: 2 words = max
    score += 0.20 * trans_score
    
    # Bonus for explicit step markers ("Step 1:", "Step 2:", etc.)
    explicit_steps = len(re.findall(r'step\s*\d+', reasoning_lower))
    if explicit_steps >= 2:
        score += 0.10  # Bonus for structured reasoning
    
    return max(0.0, min(1.0, score))


def confidence_score(log_probs: List[float]) -> float:
    """
    Compute confidence score from token log probabilities.
    
    Uses average log-prob of answer tokens to estimate model confidence.
    Maps typical log-prob range [-10, -0.1] to [0, 1].
    
    Args:
        log_probs: List of log probabilities for answer tokens
        
    Returns:
        Normalized confidence score between 0.0 and 1.0
    """
    if not log_probs:
        return 0.5  # Neutral confidence if no log probs
    
    avg_log_prob = sum(log_probs) / len(log_probs)
    
    # Map log-prob to 0-1 range
    # Typical range: -10 (low confidence) to -0.1 (high confidence)
    normalized = (avg_log_prob + 10.0) / 9.9
    return max(0.0, min(1.0, normalized))


def format_compliance_score(text: str) -> float:
    """
    Check if output follows the required format.
    
    Required format:
    <reasoning>...</reasoning>
    <answer>...</answer>
    
    Returns:
        1.0 if compliant, 0.0-0.5 for partial compliance
    """
    has_reasoning = REASONING_TAG_PATTERN.search(text) is not None
    has_answer = ANSWER_TAG_PATTERN.search(text) is not None
    
    if has_reasoning and has_answer:
        # Check ordering (reasoning should come before answer)
        reasoning_pos = text.lower().find('<reasoning>')
        answer_pos = text.lower().find('<answer>')
        if reasoning_pos < answer_pos:
            return 1.0
        return 0.8  # Wrong order but both present
    elif has_answer:
        return 0.3  # Answer only
    elif has_reasoning:
        return 0.2  # Reasoning only
    return 0.0


# =============================================================================
# Composite Reward Function
# =============================================================================

def composite_reward(
    pred_text: str,
    ref_answer: Optional[str] = None,
    log_probs: Optional[List[float]] = None,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite reward for GRPO training.
    
    R = w1 * Correctness + w2 * TraceStructure + w3 * (1 - ConfidencePenalty)
    
    If ref_answer is None (non-verifiable task), Correctness is set to 0.5
    and TraceStructure is prioritized.
    """
    w = weights or DEFAULT_WEIGHTS.copy()
    
    # Compute individual components
    if ref_answer:
        correct = correctness_score(pred_text, ref_answer)
    else:
        # For non-verifiable tasks (Creative Writing, Summarization), 
        # we give a neutral 0.5 correctness and rely heavier on trace structure.
        correct = 0.5 
    
    trace = trace_structure_score(pred_text)
    
    if log_probs:
        conf = confidence_score(log_probs)
        # Penalize overconfident wrong answers, reward confident correct answers
        if correct > 0.6:
            conf_component = conf  # High confidence + correct = good
        elif correct < 0.4:
            conf_component = 1.0 - conf  # High confidence + wrong = bad
        else:
            # For neutral correctness (0.5), we don't penalize confidence heavily
            conf_component = 0.5
    else:
        conf_component = 0.5
    
    # Weighted sum
    total = (
        w['correctness'] * correct +
        w['trace_structure'] * trace +
        w['confidence'] * conf_component
    )
    
    components = {
        'correctness': correct,
        'trace_structure': trace,
        'confidence': conf_component,
        'format_compliance': format_compliance_score(pred_text),
        'total': total
    }
    
    return total, components


# =============================================================================
# Batch Processing for GRPO
# =============================================================================

def batch_rewards(
    predictions: List[str],
    references: List[str],
    log_probs_batch: Optional[List[List[float]]] = None,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Compute rewards for a batch of predictions (GRPO rollouts).
    
    Args:
        predictions: List of model outputs
        references: List of ground truth answers
        log_probs_batch: Optional list of log-prob lists
        weights: Reward weights
        
    Returns:
        Tuple of (reward_list, component_dicts_list)
    """
    rewards = []
    all_components = []
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        lp = log_probs_batch[i] if log_probs_batch else None
        r, comps = composite_reward(pred, ref, lp, weights)
        rewards.append(r)
        all_components.append(comps)
    
    return rewards, all_components


def grpo_advantages(rewards: List[float]) -> List[float]:
    """
    Compute GRPO-style advantages (group relative).
    
    For each response in a group, advantage = reward - mean(group_rewards)
    
    Args:
        rewards: List of rewards for G responses to the same prompt
        
    Returns:
        List of advantage values
    """
    if not rewards:
        return []
    
    mean_reward = sum(rewards) / len(rewards)
    return [r - mean_reward for r in rewards]


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    # Test examples
    test_output_good = """
<reasoning>
Step 1: Convert 1 hour 30 minutes to hours -> 1.5 hours.
Step 2: Average speed = distance / time = 60 / 1.5 = 40 km/h.
Therefore, the train's average speed is 40 km/h.
</reasoning>
<answer>40 km/h</answer>
"""
    
    test_output_bad = "The answer is 40."
    
    test_ref = "40 km/h"
    
    print("=== Good Output ===")
    r, c = composite_reward(test_output_good, test_ref)
    print(f"Total Reward: {r:.3f}")
    for k, v in c.items():
        print(f"  {k}: {v:.3f}")
    
    print("\n=== Bad Output ===")
    r, c = composite_reward(test_output_bad, test_ref)
    print(f"Total Reward: {r:.3f}")
    for k, v in c.items():
        print(f"  {k}: {v:.3f}")
    
    # Test GRPO advantages
    print("\n=== GRPO Advantages ===")
    sample_rewards = [0.8, 0.5, 0.9, 0.3]
    advs = grpo_advantages(sample_rewards)
    print(f"Rewards: {sample_rewards}")
    print(f"Advantages: {[f'{a:.3f}' for a in advs]}")
