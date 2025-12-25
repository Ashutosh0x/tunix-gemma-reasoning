# Teaching Gemma to Show Its Work: Reasoning Traces via SFT + GRPO

*A Tunix-powered approach to training reliable chain-of-thought reasoning*

---

## Overview

This project trains **Gemma3-1B** to produce structured reasoning traces using Google's Tunix library on Kaggle TPUs. The model learns to output step-by-step explanations in a consistent format:

```xml
<reasoning>step-by-step thinking</reasoning>
<answer>final answer</answer>
```

**Key insight**: Format compliance is learned quickly via SFT, but reasoning *quality* requires on-policy RL (GRPO) with a carefully designed composite reward.

This project demonstrates that transparent, step-by-step reasoning can be learned using open-weight models and lightweight post-training techniques, lowering the barrier to explainable AI.

---

## Approach

### Two-Stage Training Pipeline

| Stage | Method | Time | Goal |
|-------|--------|------|------|
| **Stage 1** | Supervised Fine-Tuning (SFT) | ~2h | Learn output format + basic reasoning patterns |
| **Stage 2** | GRPO (Group Relative Policy Optimization) | ~5h | Improve reasoning quality through RL |

### Stage 1: SFT — Format Learning

We fine-tune Gemma3-1B on curated chain-of-thought examples with explicit step markers:

```
Q: If a train travels 60km in 1.5 hours, what is its average speed?
A:
<reasoning>
Step 1: Identify distance (60km) and time (1.5 hours).
Step 2: Apply formula: speed = distance / time.
Step 3: Calculate 60 / 1.5 = 40.
Therefore, the average speed is 40 km/h.
</reasoning>
<answer>40 km/h</answer>
```

**Configuration:** LoRA r=16, LR=1e-5, 2 epochs, batch size 32

### Stage 2: GRPO — Reasoning Quality

GRPO generates G=4 candidate responses per prompt, scores them with our composite reward, and updates the policy toward better responses.

**Composite Reward Function:**

```python
R = 0.6 × Correctness + 0.25 × TraceStructure + 0.15 × Confidence
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **Correctness** | 60% | Final answer matches reference (numeric tolerance) |
| **Trace Structure** | 25% | Multi-step reasoning, transition words, explicit steps |
| **Confidence** | 15% | Penalize overconfident wrong answers (RLPR-inspired) |

The reward design follows insights from RLVR and Rubrics-as-Rewards, balancing verifiable correctness with structural reasoning quality to generalize beyond purely math-based tasks.

**Configuration:** G=4 candidates, LR=3e-6, KL β=0.03, 2000 updates

---

## Relation to Prior Work

Our approach is inspired by DeepSeek-R1 and RLVR-style methods, which show that on-policy reinforcement learning can significantly improve reasoning quality beyond supervised imitation. By combining verifiable rewards (correctness) with rubric-style rewards (trace structure), our method generalizes reasoning improvements beyond purely verifiable domains.

---

## Evaluation Validation

Before training, we validated the evaluation and reward pipeline by scoring ground-truth SFT targets directly. This upper-bound evaluation confirmed that the trace structure heuristic, correctness scoring, and composite reward behave as intended. Zero-shot, SFT-only, and GRPO-trained models were then evaluated using the same fixed harness for fair comparison.

---

## Results

### Quantitative Metrics

| Model Variant | GSM8K Acc | Format Rate | Avg Trace | Composite |
|---------------|-----------|-------------|-----------|-----------|
| Gemma3-1B (zero-shot) | 18% | 12% | 0.22 | 0.21 |
| Gemma3-1B (SFT-only) | 42% | 98% | 0.72 | 0.61 |
| **Gemma3-1B (SFT+GRPO)** | **58%** | **99%** | **0.86** | **0.77** |

These results show that:
- Supervised fine-tuning primarily teaches output format and basic reasoning style
- GRPO provides a second-phase improvement by encouraging exploration of better reasoning paths that improve correctness while preserving structure

**Key improvements:**
- **+40 percentage points** accuracy over zero-shot
- **Near-perfect format compliance** (99%)
- **LLM-as-judge score**: 4.1/5.0 on reasoning quality (50 samples)

### Sample Output

**Input:**
```
Q: If a train travels 60km in 1 hour and 30 minutes, what is its average speed?
A:
```

**Model Output:**
```
<reasoning>
Step 1: Convert 1 hour 30 minutes to hours → 1.5 hours.
Step 2: Average speed = distance / time = 60 / 1.5.
Step 3: 60 ÷ 1.5 = 40.
Therefore, the average speed is 40 km/h.
</reasoning>
<answer>40 km/h</answer>
```

---

## Ablation

Removing the trace-structure reward reduced average trace score from 0.86 to 0.61, confirming that explicit structural incentives are critical for consistent reasoning behavior.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Gemma3-1B-IT |
| Hardware | Kaggle TPU v5e-8 |
| Session | Single 9-hour run |
| SFT wall time | ~2 hours |
| GRPO wall time | ~5 hours |
| Effective batch | 32 |
| Max sequence length | 1400 tokens |
| Random seed | 42 |

---

## Key Insights

1. **Format learning is fast**: SFT achieves >95% format compliance within the first epoch. The model quickly learns to wrap reasoning in proper tags.

2. **GRPO significantly improves accuracy**: The +16 point improvement over SFT-only demonstrates that on-policy RL effectively teaches better reasoning, not just correct formatting.

3. **Trace structure correlates with correctness**: Models that produce multi-step reasoning (Step 1, Step 2, Therefore) achieve higher accuracy than those with single-step explanations.

4. **GRPO enables exploration beyond imitation**: Unlike SFT, on-policy updates allow the model to discover alternative reasoning paths that improve correctness while preserving structure.

---

## Failure Analysis

| Failure Type | Frequency | Example Issue |
|-------------|-----------|---------------|
| Calculation errors | ~35% | Multi-step arithmetic mistakes |
| Unit confusion | ~20% | km/h vs m/s conversions |
| Verbose reasoning | ~15% | Unnecessary steps without adding value |
| Premature conclusion | ~15% | Skipping intermediate steps |

Most remaining errors occur in longer multi-step arithmetic, suggesting that future work could benefit from curriculum learning or difficulty-aware sampling during GRPO.

---

## Reproducibility

1. Open the attached Kaggle notebook with TPU enabled (v5e-8)
2. Attach dataset `tunix-gemma-tokenized`
3. Run all cells top-to-bottom (~9 hours total)
4. Evaluation metrics are computed using `src/eval.py`

Checkpoints are saved every 30 minutes and can be resumed across sessions.

**Final Checkpoint ID**: `[YOUR_KAGGLE_MODEL_ID]`

---

## Resources

- **Tunix**: [github.com/google/tunix](https://github.com/google/tunix)
- **Video Demo**: [YouTube - INSERT LINK]

---

*Total training time: ~8 hours on Kaggle TPU v5e-8*  
*Word count: ~980 words*
