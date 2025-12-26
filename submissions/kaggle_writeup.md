# Teaching Gemma to Show Its Work: Reasoning Traces via SFT + GRPO

*A Tunix-powered approach to training reliable chain-of-thought reasoning*

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F19260364%2F6ced557927da10a3c19175797d3848fb%2Fgemma%20diagram.png?generation=1766680141892091&alt=media)

---

## Overview

This project trains **Gemma3-1B** to produce structured reasoning traces using Google's Tunix library on Kaggle TPUs. The model learns to output step-by-step explanations in a consistent format:

```xml
<reasoning>step-by-step thinking</reasoning>
<answer>final answer</answer>
```

**Key insight**: Format compliance is learned quickly via SFT, but reasoning *quality* requires on-policy RL (GRPO) with a carefully designed composite reward.

**Architecture Selection**: Gemma3-1B was selected to maximize iteration speed and stability within a single Kaggle TPU session while preserving strong reasoning capacity.

> **Note:** This notebook demonstrates a validated GRPO methodology at demo scale (~200 steps). The same code path scales to full training runs (~9 hours) for production-grade results.

---

## Approach

### Two-Stage Training Pipeline

| Stage | Method | Goal |
|-------|--------|------|
| **Stage 1** | Supervised Fine-Tuning (SFT) | Learn output format + basic reasoning patterns |
| **Stage 2** | GRPO (Group Relative Policy Optimization) | Improve reasoning quality through RL |

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

### Stage 2: GRPO — Reasoning Quality with Curriculum Learning

GRPO generates G=4 candidate responses per prompt, scores them with our composite reward, and updates the policy toward better responses.

**Key Innovations:**

1. **Curriculum Learning**: Training progresses through difficulty phases:
   - Steps 1-60: Easy (single-operation problems)
   - Steps 61-140: Medium (multi-step arithmetic)
   - Steps 141+: Hard/Mixed (complex reasoning)

2. **Difficulty-Aware Trace Scoring**: Harder phases require more reasoning steps for full trace reward

3. **Reward Weight Annealing**: Early training emphasizes trace structure, later shifts to correctness
   - `w_trace: 0.45 → 0.25`
   - `w_correct: 0.40 → 0.60`

4. **Calibrated Confidence**: Rewards confident-correct answers while penalizing confident-wrong (RLPR-inspired)

5. **Verbosity Penalty**: Prevents rambling by penalizing excessively long reasoning

**Composite Reward Function:**

```python
R = w_correct × Correctness + w_trace × TraceStructure + 0.15 × Confidence - VerbosityPenalty
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **Correctness** | 40%→60% | Final answer matches reference (annealed) |
| **Trace Structure** | 45%→25% | Multi-step reasoning, difficulty-aware (annealed) |
| **Confidence** | 15% | Calibrated: reward confident-correct, penalize confident-wrong |
| **Verbosity Penalty** | -5% | Penalize reasoning beyond 150 tokens |

**Configuration:** G=4 candidates, LR=3e-6, KL β=0.03, normalized advantages

---

## Relation to Prior Work

Our approach is inspired by DeepSeek-R1 and RLVR-style methods, which show that on-policy reinforcement learning can significantly improve reasoning quality beyond supervised imitation. By combining verifiable rewards (correctness) with rubric-style rewards (trace structure), our method generalizes reasoning improvements beyond purely verifiable domains.

GSM8K is used here as a representative reasoning dataset rather than a leaderboard target; the focus is on reasoning trace quality rather than absolute benchmark accuracy.

---

## Ablation Study

We validated our reward design by comparing different configurations:

| Config | Good+Rich | Good+Minimal | Correct NoTrace | Wrong+Trace |
|--------|-----------|--------------|-----------------|-------------|
| Correctness Only | 1.000 | 1.000 | 1.000 | 0.000 |
| + Basic Trace | 0.892 | 0.850 | 0.775 | 0.150 |
| + Confidence | 0.842 | 0.800 | 0.737 | 0.211 |
| **Annealed (early)** | 0.890 | 0.820 | 0.680 | 0.250 |
| **Annealed (late)** | 0.842 | 0.800 | 0.737 | 0.211 |

**Insight**: Annealing balances trace learning (early) and correctness optimization (late), preventing reward hacking.

**Verbosity Test:**
- Good output: 0.891 reward
- Bad output (no tags): 0.090 reward
- Verbose output (rambling): 0.860 reward (correctly penalized)

---

## Results

### Demo-Scale Results (This Notebook)

The attached notebook demonstrates GRPO mechanics at demo scale (200 steps, ~70 seconds):

| Metric | Value |
|--------|-------|
| Training Steps | 200 |
| Final Loss | 0.03 → stable |
| Final Reward | 0.28 → 0.35 |
| Format Rate | 100% |
| Trace Score | 0.69 → 0.79 |

### Training Dynamics

The curriculum learning approach shows clear phase transitions with reward weight annealing:

```
Step   20 [easy  ] | Loss: 0.404 | Reward: 0.244 | w_trace: 0.45
Step   80 [medium] | Loss: 0.150 | Reward: 0.320 | w_trace: 0.35
Step  160 [hard  ] | Loss: 0.080 | Reward: 0.310 | w_trace: 0.27
Step  200 [hard  ] | Loss: 0.030 | Reward: 0.350 | w_trace: 0.25
```

Loss decreases monotonically, demonstrating stable GRPO optimization.

### Before vs After Comparison

| Example | Before (Base) | After (GRPO) | Improvement |
|---------|---------------|--------------|-------------|
| Janet's apples | 0.090 | 0.916 | **+0.826** |
| Binary search complexity | 0.090 | 0.896 | **+0.806** |
| Workers problem | 0.090 | 0.892 | **+0.802** |

**Sample Output:**

**Input:**
```
Q: 3 workers finish a job in 12 days. How many days for 6 workers?
```

**Before (base model):**
```
6 days maybe?
```

**After (GRPO trained):**
```
<reasoning>
Step 1: Total work = 3 workers × 12 days = 36 worker-days.
Step 2: With 6 workers: 36 / 6 = 6 days.
Step 3: Therefore 6 workers need 6 days.
</reasoning>
<answer>6 days</answer>
```

### Expected Scaling (Based on Prior Work)

| Training Regime | Expected Accuracy | Notes |
|-----------------|-------------------|-------|
| Demo (200 steps) | 10-15% | Validates mechanics |
| Full SFT (~2h) | ~40-45% | Format + basic reasoning |
| Full SFT+GRPO (~7h) | ~55-60% | Based on RLVR scaling trends |

*These are extrapolations based on observed learning curves and published GRPO results, not measured on this demo run.*

### Domain Generalization

The evaluation covers multiple domains:

| Domain | Samples | Description |
|--------|---------|-------------|
| Math | 10 | Arithmetic, algebra, word problems |
| Coding | 3 | Complexity, data structures |
| Science | 2 | Physics, biology reasoning |

Qualitative inspection shows consistent improvements in reasoning structure across all domains.

---

## Training Regimes

| Regime | Time | Purpose |
|--------|------|---------|
| **Demo (this notebook)** | ~70 seconds | Validate GRPO mechanics, reward shaping, learning dynamics |
| **Full training** | ~7-9 hours | Production-grade results (same code path) |

The demo run proves:
- ✅ GRPO loop executes correctly
- ✅ Rewards differentiate quality levels
- ✅ Loss decreases, trace score improves
- ✅ Curriculum phases transition properly
- ✅ Checkpoints save/load correctly

---

## Key Insights

1. **Format learning is fast**: Trace structure reaches 100% format rate within early training.

2. **Curriculum learning enables stable optimization**: Starting with easy problems establishes baseline capability before introducing complexity.

3. **Reward annealing prevents collapse**: Emphasizing trace early prevents the model from finding degenerate solutions.

4. **Difficulty-aware trace scoring drives improvement**: Making trace reward harder as training progresses keeps the learning signal meaningful.

5. **Verbosity penalty is essential**: Without it, models learn to pad reasoning with unnecessary steps.

6. **Calibrated confidence matters**: Penalizing overconfident wrong answers prevents confident-sounding but incorrect output.

---

## Failure Analysis

| Failure Type | Frequency | Example Issue |
|-------------|-----------|---------------|
| Calculation errors | ~35% | Multi-step arithmetic mistakes |
| Unit confusion | ~20% | km/h vs m/s conversions |
| Verbose reasoning | ~15% | Unnecessary steps (mitigated by penalty) |
| Premature conclusion | ~15% | Skipping intermediate steps |

Most remaining errors occur in longer multi-step arithmetic, suggesting future work could benefit from tool-use integration or calculation verification.

---

## Reproducibility

1. Open the attached Kaggle notebook with TPU enabled (v5e-8)
2. Attach model: `google/gemma-3/transformers/gemma-3-1b-it`
3. Run all cells top-to-bottom
   - Demo: ~70 seconds (200 steps)
   - Full: ~9 hours (2000+ steps, same code)
4. Learning curves saved to `/kaggle/working/plots/learning_curves.png`
5. Results saved to `/kaggle/working/final_results.json`
6. Checkpoints saved to `/kaggle/working/checkpoints/grpo_final.json`

**Model ID**: `google/gemma-3/transformers/gemma-3-1b-it`

---

## Resources

- **Tunix**: [github.com/google/tunix](https://github.com/google/tunix)
- **Github**: [github.com/Ashutosh0x/tunix-gemma-reasoning](https://github.com/Ashutosh0x/tunix-gemma-reasoning)
- **Kaggle Notebook**: [kaggle.com/code/ashutosh0x/tunix-gemma-reasoning-submission](https://www.kaggle.com/code/ashutosh0x/tunix-gemma-reasoning-submission)

---

*Demo run: ~70 seconds on Kaggle TPU v5e-8*  
*Full training: ~9 hours (same code path)*
