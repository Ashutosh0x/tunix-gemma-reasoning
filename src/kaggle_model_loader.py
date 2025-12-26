"""
Kaggle Model Loader - Load Gemma from Kaggle's native model path
No Hugging Face authentication required!
"""
import os
from pathlib import Path

# Kaggle model path (when using model_sources in kernel-metadata.json)
KAGGLE_MODEL_PATH = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"

def get_model_path():
    """Get the model path - works on both Kaggle and local."""
    if os.path.exists(KAGGLE_MODEL_PATH):
        print(f"✅ Using Kaggle model: {KAGGLE_MODEL_PATH}")
        return KAGGLE_MODEL_PATH
    else:
        # Fallback to HuggingFace (requires auth)
        print("⚠️ Kaggle model not found, falling back to HuggingFace...")
        return "google/gemma-3-1b-it"

def load_tokenizer(model_path=None):
    """Load tokenizer from Kaggle path or HuggingFace."""
    from transformers import AutoTokenizer
    
    path = model_path or get_model_path()
    tokenizer = AutoTokenizer.from_pretrained(path)
    print(f"✅ Loaded tokenizer from: {path}")
    return tokenizer

def load_model(model_path=None, **kwargs):
    """Load model from Kaggle path or HuggingFace."""
    from transformers import AutoModelForCausalLM
    
    path = model_path or get_model_path()
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    print(f"✅ Loaded model from: {path}")
    return model


# === CODE TO ADD TO YOUR NOTEBOOK ===
# Copy this block to the top of 03_rl_grpo_training.ipynb:
"""
# Cell 1: Model Path Setup (RUN THIS FIRST)
import os

# Use Kaggle's pre-loaded model (no HF auth needed!)
KAGGLE_MODEL_PATH = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"

if os.path.exists(KAGGLE_MODEL_PATH):
    MODEL_PATH = KAGGLE_MODEL_PATH
    print(f"✅ Using Kaggle model: {MODEL_PATH}")
else:
    # Fallback for local development
    MODEL_PATH = "google/gemma-3-1b-it"
    print(f"⚠️ Using HuggingFace: {MODEL_PATH}")

# Then replace:
#   CFG['model_name'] = 'google/gemma-3-1b-it'
# With:
#   CFG['model_name'] = MODEL_PATH
"""
