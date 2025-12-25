# Tunix Kaggle Hackathon — Teach Gemma to Show Its Work

Train a Gemma model (Gemma2 2B or Gemma3 1B) using [Tunix](https://github.com/google/tunix) to output structured reasoning traces followed by answers.

## 🎯 Goal

Make the model output:
```xml
<reasoning>step-by-step thinking</reasoning>
<answer>final answer</answer>
```

## 📁 Project Structure

```
├── notebooks/           # Kaggle-ready notebooks
│   ├── 00_setup_environment.ipynb
│   ├── 01_data_prep_and_tokenize.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_rl_grpo_training.ipynb
│   └── 04_evaluation_and_export.ipynb
├── configs/             # Training configurations
│   ├── sft_config.yaml
│   └── rl_config.yaml
├── src/                 # Core modules
│   ├── rewards.py       # Composite reward functions
│   ├── data_utils.py    # Dataset loaders
│   ├── eval.py          # Evaluation harness
│   └── export.py        # Checkpoint export
├── data/                # Datasets
│   ├── raw/             # Original datasets
│   ├── prepared/        # Formatted for training
│   └── tokenized/       # Pre-tokenized shards
├── checkpoints/         # Model checkpoints
├── logs/                # Training logs
└── submissions/         # Kaggle submission files
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install git+https://github.com/google/tunix.git
pip install jax jaxlib flax optax transformers datasets
```

### 2. Prepare Data
```python
from src.data_utils import load_gsm8k, prepare_training_data

examples = load_gsm8k("data/raw/gsm8k.jsonl")
prepare_training_data(examples, "data/prepared/train.jsonl")
```

### 3. Run SFT Training
```python
# Use configs/sft_config.yaml
# See notebooks/02_sft_training.ipynb
```

### 4. Run GRPO (RL)
```python
# Use configs/rl_config.yaml  
# See notebooks/03_rl_grpo_training.ipynb
```

### 5. Export & Submit
```python
from src.export import export_for_kaggle

export_for_kaggle("checkpoints/rl/best", "submissions/model")
```

## 📊 Reward Function

Composite reward for GRPO training:

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 0.60 | Answer matches reference |
| Trace Structure | 0.25 | Logical steps, transition words |
| Confidence | 0.15 | Calibrated confidence (RLPR) |

## ⏱️ Kaggle 9-Hour Session Plan

| Time | Phase | Description |
|------|-------|-------------|
| 0:00-0:30 | Setup | Install, load data |
| 0:30-3:00 | SFT | Fine-tune on CoT examples |
| 3:00-7:30 | GRPO | On-policy RL with rewards |
| 7:30-8:30 | Eval | Validate and select best |
| 8:30-9:00 | Export | Save Kaggle-compatible model |

## 📈 Expected Results

- **Format Compliance**: ≥95%
- **GSM8K Accuracy**: ~40-65% (from ~52% baseline)
- **Trace Score**: 0.75-0.95

## 📚 Resources

- [Tunix Documentation](https://tunix.readthedocs.io/)
- [GRPO Demo (Gemma3)](https://www.kaggle.com/code/windmaple/grpo-demo-gemma3-1b)
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954)

## 📄 License

Apache 2.0