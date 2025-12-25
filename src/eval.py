"""
Tunix Kaggle Hackathon - Evaluation Harness
============================================
Evaluation utilities for reasoning models: accuracy, format compliance, LLM-as-judge.
"""

import json
import re
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from .rewards import (
    correctness_score, 
    trace_structure_score, 
    format_compliance_score,
    extract_answer,
    extract_reasoning
)


@dataclass
class EvalResult:
    """Single evaluation result."""
    prompt: str
    prediction: str
    reference: str
    is_correct: bool
    format_compliant: bool
    trace_score: float
    extracted_answer: Optional[str]
    extracted_reasoning: Optional[str]
    judge_score: Optional[float] = None
    judge_feedback: Optional[str] = None


@dataclass
class EvalSummary:
    """Aggregate evaluation summary."""
    total_examples: int
    accuracy: float
    format_compliance_rate: float
    avg_trace_score: float
    avg_judge_score: Optional[float]
    by_domain: Dict[str, Dict]
    timestamp: str


# =============================================================================
# Core Evaluation Functions
# =============================================================================

def evaluate_single(
    prediction: str,
    reference: str,
    prompt: str = ""
) -> EvalResult:
    """
    Evaluate a single prediction.
    
    Args:
        prediction: Model output
        reference: Ground truth answer
        prompt: Original prompt (for logging)
        
    Returns:
        EvalResult with all metrics
    """
    correct = correctness_score(prediction, reference) > 0.5
    format_ok = format_compliance_score(prediction) > 0.8
    trace = trace_structure_score(prediction)
    
    return EvalResult(
        prompt=prompt,
        prediction=prediction,
        reference=reference,
        is_correct=correct,
        format_compliant=format_ok,
        trace_score=trace,
        extracted_answer=extract_answer(prediction),
        extracted_reasoning=extract_reasoning(prediction)
    )


def evaluate_batch(
    predictions: List[str],
    references: List[str],
    prompts: Optional[List[str]] = None,
    domains: Optional[List[str]] = None
) -> Tuple[List[EvalResult], EvalSummary]:
    """
    Evaluate a batch of predictions.
    
    Args:
        predictions: List of model outputs
        references: List of ground truth answers
        prompts: Optional list of prompts
        domains: Optional domain labels for breakdown
        
    Returns:
        Tuple of (results list, summary)
    """
    if prompts is None:
        prompts = [""] * len(predictions)
    if domains is None:
        domains = ["general"] * len(predictions)
    
    results = []
    domain_stats = {}
    
    for pred, ref, prompt, domain in zip(predictions, references, prompts, domains):
        result = evaluate_single(pred, ref, prompt)
        results.append(result)
        
        # Accumulate domain stats
        if domain not in domain_stats:
            domain_stats[domain] = {'correct': 0, 'total': 0, 'trace_sum': 0}
        domain_stats[domain]['total'] += 1
        domain_stats[domain]['correct'] += int(result.is_correct)
        domain_stats[domain]['trace_sum'] += result.trace_score
    
    # Compute summary
    n = len(results)
    accuracy = sum(r.is_correct for r in results) / n if n > 0 else 0
    format_rate = sum(r.format_compliant for r in results) / n if n > 0 else 0
    avg_trace = sum(r.trace_score for r in results) / n if n > 0 else 0
    
    # Judge scores if available
    judge_scores = [r.judge_score for r in results if r.judge_score is not None]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None
    
    # Domain breakdown
    by_domain = {}
    for domain, stats in domain_stats.items():
        by_domain[domain] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            'count': stats['total'],
            'avg_trace': stats['trace_sum'] / stats['total'] if stats['total'] > 0 else 0
        }
    
    summary = EvalSummary(
        total_examples=n,
        accuracy=accuracy,
        format_compliance_rate=format_rate,
        avg_trace_score=avg_trace,
        avg_judge_score=avg_judge,
        by_domain=by_domain,
        timestamp=datetime.now().isoformat()
    )
    
    return results, summary


# =============================================================================
# LLM-as-Judge
# =============================================================================

JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of a reasoning trace and answer.

Question: {question}

Model Response:
{response}

Reference Answer: {reference}

Rate the response on these criteria (each 1-5):
1. Correctness: Is the final answer correct?
2. Reasoning Quality: Are the steps logical and well-explained?
3. Completeness: Does the response fully address the question?
4. Clarity: Is the explanation easy to follow?

Provide your ratings as JSON:
{{"correctness": X, "reasoning_quality": X, "completeness": X, "clarity": X, "overall": X, "feedback": "..."}}
"""


def create_judge_prompt(question: str, response: str, reference: str) -> str:
    """Create a prompt for LLM-as-judge evaluation."""
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
        reference=reference
    )


def parse_judge_response(judge_output: str) -> Tuple[float, str]:
    """
    Parse LLM judge response.
    
    Returns:
        Tuple of (overall_score, feedback)
    """
    # Try to extract JSON
    try:
        # Find JSON in response
        json_match = re.search(r'\{[^}]+\}', judge_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            overall = data.get('overall', 3) / 5.0  # Normalize to 0-1
            feedback = data.get('feedback', '')
            return overall, feedback
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to extract numbers
    numbers = re.findall(r'(\d+)', judge_output)
    if numbers:
        avg = sum(int(n) for n in numbers[:5]) / len(numbers[:5])
        return avg / 5.0, judge_output[:200]
    
    return 0.5, "Could not parse judge response"


async def evaluate_with_judge(
    predictions: List[str],
    references: List[str],
    prompts: List[str],
    judge_fn,  # async function: str -> str (prompt -> response)
    max_samples: int = 50
) -> List[EvalResult]:
    """
    Evaluate with LLM-as-judge (async).
    
    Args:
        predictions: Model outputs
        references: Ground truths
        prompts: Original questions
        judge_fn: Async function to call judge LLM
        max_samples: Maximum samples to judge (for cost/time)
        
    Returns:
        List of EvalResults with judge scores
    """
    import asyncio
    
    results = []
    tasks = []
    
    for i, (pred, ref, prompt) in enumerate(zip(predictions[:max_samples], 
                                                  references[:max_samples], 
                                                  prompts[:max_samples])):
        # Basic eval first
        result = evaluate_single(pred, ref, prompt)
        
        # Create judge task
        judge_prompt = create_judge_prompt(prompt, pred, ref)
        task = judge_fn(judge_prompt)
        tasks.append((result, task))
    
    # Run judge calls
    for result, task in tasks:
        try:
            judge_output = await task
            score, feedback = parse_judge_response(judge_output)
            result.judge_score = score
            result.judge_feedback = feedback
        except Exception as e:
            result.judge_feedback = f"Judge error: {str(e)}"
        results.append(result)
    
    return results


# =============================================================================
# Report Generation
# =============================================================================

def save_eval_report(
    results: List[EvalResult],
    summary: EvalSummary,
    output_dir: str,
    name: str = "eval"
) -> str:
    """
    Save evaluation report to disk.
    
    Args:
        results: List of evaluation results
        summary: Evaluation summary
        output_dir: Output directory
        name: Report name prefix
        
    Returns:
        Path to saved report
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_path = os.path.join(output_dir, f"{name}_results_{timestamp}.jsonl")
    with open(results_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')
    
    # Save summary
    summary_path = os.path.join(output_dir, f"{name}_summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(summary), f, indent=2)
    
    # Generate markdown report
    report_path = os.path.join(output_dir, f"{name}_report_{timestamp}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Evaluation Report\n\n")
        f.write(f"**Timestamp:** {summary.timestamp}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total Examples | {summary.total_examples} |\n")
        f.write(f"| Accuracy | {summary.accuracy:.2%} |\n")
        f.write(f"| Format Compliance | {summary.format_compliance_rate:.2%} |\n")
        f.write(f"| Avg Trace Score | {summary.avg_trace_score:.3f} |\n")
        if summary.avg_judge_score:
            f.write(f"| Avg Judge Score | {summary.avg_judge_score:.3f} |\n")
        
        f.write(f"\n## By Domain\n\n")
        f.write(f"| Domain | Accuracy | Count | Avg Trace |\n|--------|----------|-------|----------|\n")
        for domain, stats in summary.by_domain.items():
            f.write(f"| {domain} | {stats['accuracy']:.2%} | {stats['count']} | {stats['avg_trace']:.3f} |\n")
        
        f.write(f"\n## Sample Predictions\n\n")
        for r in results[:5]:
            f.write(f"### Example\n")
            f.write(f"**Prompt:** {r.prompt[:200]}...\n\n")
            f.write(f"**Extracted Answer:** {r.extracted_answer}\n\n")
            f.write(f"**Reference:** {r.reference}\n\n")
            f.write(f"**Correct:** {'✓' if r.is_correct else '✗'} | **Trace Score:** {r.trace_score:.3f}\n\n")
            f.write("---\n\n")
    
    return report_path


# =============================================================================
# Simple Dataset Evaluation (for preds.jsonl format)
# =============================================================================

def evaluate_dataset(predictions: List[Dict], weights: dict = None) -> Dict:
    """
    Evaluate a dataset of predictions from preds.jsonl format.
    
    Args:
        predictions: List of dicts with keys: pred, ref (optional), prompt (optional)
        weights: Optional reward weights
        
    Returns:
        Dict with metrics: n, accuracy, format_rate, avg_trace_score, avg_composite
    """
    from .rewards import composite_reward
    
    n = len(predictions)
    if n == 0:
        return {'n': 0, 'accuracy': 0, 'format_rate': 0, 'avg_trace_score': 0, 'avg_composite': 0}
    
    correct_count = 0
    format_ok_count = 0
    trace_sum = 0.0
    composite_sum = 0.0
    verifiable_count = 0
    
    for p in predictions:
        pred_text = p.get('pred', '')
        ref = p.get('ref')
        
        # Format check
        format_ok = format_compliance_score(pred_text) >= 0.8
        if format_ok:
            format_ok_count += 1
        
        # Trace score
        trace = trace_structure_score(pred_text)
        trace_sum += trace
        
        # Correctness (only for verifiable examples)
        if ref is not None:
            verifiable_count += 1
            correct = correctness_score(pred_text, str(ref)) > 0.5
            if correct:
                correct_count += 1
        
        # Composite reward
        comp, _ = composite_reward(pred_text, str(ref) if ref else '', weights=weights)
        composite_sum += comp
    
    return {
        'n': n,
        'accuracy': correct_count / verifiable_count if verifiable_count > 0 else 0.0,
        'format_rate': format_ok_count / n,
        'avg_trace_score': trace_sum / n,
        'avg_composite': composite_sum / n
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test evaluation
    preds = [
        "<reasoning>Step 1: 2+2=4. Therefore the answer is 4.</reasoning><answer>4</answer>",
        "The answer is 5",
        "<reasoning>Wrong calculation.</reasoning><answer>10</answer>"
    ]
    refs = ["4", "5", "4"]
    prompts = ["What is 2+2?", "What is 2+3?", "What is 2+2?"]
    
    results, summary = evaluate_batch(preds, refs, prompts)
    
    print("=== Evaluation Summary ===")
    print(f"Accuracy: {summary.accuracy:.2%}")
    print(f"Format Compliance: {summary.format_compliance_rate:.2%}")
    print(f"Avg Trace Score: {summary.avg_trace_score:.3f}")
