# Example Model Outputs — Judge Ready
# =====================================
# These are examples of the expected output format

## ✅ Math (verifiable)

**Input:**
```
Q: If a train travels 60km in 1 hour and 30 minutes, what is its average speed?
A:
```

**Model Output:**
```
<reasoning>
Step 1: Convert 1 hour 30 minutes to hours, which is 1.5 hours.
Step 2: Average speed is distance divided by time.
Step 3: 60 divided by 1.5 equals 40.
</reasoning>
<answer>40 km/h</answer>
```

---

## ✅ Coding Reasoning

**Input:**
```
Q: What is the time complexity of binary search and why?
A:
```

**Model Output:**
```
<reasoning>
Binary search repeatedly halves the search space.
At each step, one comparison is made and half of the remaining elements are discarded.
The number of steps required grows logarithmically with the input size.
</reasoning>
<answer>O(log n)</answer>
```

---

## ✅ Creative / Non-verifiable

**Input:**
```
Q: Why is having a clear reasoning process important in decision making?
A:
```

**Model Output:**
```
<reasoning>
Clear reasoning helps break complex problems into smaller steps.
It reduces cognitive bias by making assumptions explicit.
It also allows others to evaluate and trust the decision process.
</reasoning>
<answer>
Because it improves clarity, reduces errors, and builds trust in decisions.
</answer>
```

---

## ✅ Science

**Input:**
```
Q: Why do we see lightning before we hear thunder?
A:
```

**Model Output:**
```
<reasoning>
Step 1: Light travels at approximately 300,000 km/s.
Step 2: Sound travels at approximately 343 m/s.
Step 3: Since light is much faster than sound, we see lightning before hearing thunder.
</reasoning>
<answer>Light travels much faster than sound.</answer>
```

---

## ✅ Multi-step Math

**Input:**
```
Q: A jacket originally costs $80. It is on sale for 25% off. What is the sale price?
A:
```

**Model Output:**
```
<reasoning>
Step 1: Original price = $80.
Step 2: Discount = 25% of $80 = 0.25 × 80 = $20.
Step 3: Sale price = $80 - $20 = $60.
</reasoning>
<answer>$60</answer>
```

---

## Key Qualities Judges Look For

✅ **Concise** — 3-5 reasoning steps  
✅ **Logical flow** — Each step builds on previous  
✅ **No hallucinations** — Math is correct  
✅ **Tags intact** — `<reasoning>` and `<answer>` properly closed  
✅ **Generalizable** — Works across domains
