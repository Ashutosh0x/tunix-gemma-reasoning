# Tunix Hackathon Video Script
# =============================
# Duration: ~3 minutes
# Target: Instructional value for developers learning reasoning model training

## INTRO (0:00 - 0:20)
# -------------------
# [On screen: Title card with Tunix logo + Gemma logo]

SCRIPT:
"Hi! In this video, I'll show you how to train a Gemma model to show its 
reasoning step by step using Tunix, Google's new JAX-native library for 
LLM post-training.

By the end, your model will output structured thinking traces before 
giving an answer."


## PROBLEM STATEMENT (0:20 - 0:40)  
# --------------------------------
# [On screen: Before/after example comparison]

SCRIPT:
"Most LLMs just give you an answer. But we want models that SHOW their work.

Here's what we're building:
- Input: 'What is 60km divided by 1.5 hours?'
- Before: '40 km/h' (just the answer)
- After: Step 1, convert time. Step 2, divide. Therefore 40 km/h."

DEMO COMMAND:
# Show the output format
<reasoning>
Step 1: Convert 1 hour 30 min to 1.5 hours
Step 2: Speed = 60 / 1.5 = 40 km/h
</reasoning>
<answer>40 km/h</answer>


## APPROACH OVERVIEW (0:40 - 1:10)
# --------------------------------
# [On screen: Two-stage diagram - SFT → GRPO]

SCRIPT:
"Our approach uses two stages:

Stage 1: Supervised Fine-Tuning on chain-of-thought examples.
This teaches the model the output FORMAT.

Stage 2: GRPO - Group Relative Policy Optimization.
This is on-policy RL that improves CORRECTNESS and trace quality.

The key insight: we use a composite reward that combines:
- Answer correctness (60%)
- Reasoning structure (25%)  
- Confidence calibration (15%)"


## CODE WALKTHROUGH (1:10 - 2:10)
# -------------------------------
# [On screen: Notebook cells running on Kaggle]

SCRIPT:
"Let me show you the key code.

First, we install Tunix and load the Gemma model:"

DEMO COMMANDS:
```python
!pip install git+https://github.com/google/tunix.git

from tunix import modeling, trainers
model = modeling.Gemma.from_pretrained("google/gemma-3-1b-it")
```

SCRIPT:
"Next, we format our training data with reasoning tags:"

```python
formatted = f"""Q: {question}
A:
<reasoning>{reasoning}</reasoning>
<answer>{answer}</answer>"""
```

SCRIPT:
"For SFT, we use LoRA for efficiency on Kaggle TPUs:"

```python
trainer = trainers.PeftTrainer(
    model=model,
    config=sft_config,
    train_dataset=train_data
)
trainer.train()
```

SCRIPT:
"Then we switch to GRPO. The reward function is the secret sauce:"

```python
def composite_reward(pred, ref):
    r = 0.6 * correctness_score(pred, ref)
    r += 0.25 * trace_structure_score(pred)
    r += 0.15 * confidence_score(log_probs)
    return r
```

SCRIPT:
"GRPO generates multiple answers per prompt, scores them, and trains 
the model to prefer higher-scoring outputs."


## RESULTS (2:10 - 2:40)
# ----------------------
# [On screen: Results table + sample outputs]

SCRIPT:
"After one 9-hour Kaggle session, here's what we achieved:

- Format compliance: over 95%
- GSM8K accuracy: improved from 52% to about 64%
- Trace quality score: 0.85

Here's a real output from the trained model..."

# [Show actual model output example]


## CONCLUSION (2:40 - 3:00)
# -------------------------
# [On screen: Resources and links]

SCRIPT:
"That's how you train a reasoning model with Tunix!

Key takeaways:
1. SFT first to teach format
2. GRPO to improve correctness
3. Composite rewards for balanced learning

Check out the notebook linked below and the Tunix documentation.
Thanks for watching!"

# [End card with links:
#  - Kaggle Notebook: [link]
#  - Tunix GitHub: github.com/google/tunix
#  - Documentation: tunix.readthedocs.io
# ]


## RECORDING TIPS
# ---------------
# 1. Use OBS or Loom for screen recording
# 2. Record Kaggle notebook in action (not just slides)
# 3. Show actual model outputs, not mocks
# 4. Keep energy high but pace steady
# 5. Add captions for accessibility
# 6. Target 1080p resolution
# 7. Background music optional (low volume)


## B-ROLL SUGGESTIONS
# -------------------
# - Tunix GitHub repo scrolling
# - TPU metrics in Kaggle
# - TensorBoard training curves
# - Model generating output in real-time
# - Before/after output comparison
