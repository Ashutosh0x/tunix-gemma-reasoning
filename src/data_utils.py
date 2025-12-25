"""
Tunix Kaggle Hackathon - Data Utilities
=======================================
Dataset loading, formatting, and preprocessing for CoT reasoning training.
"""

import json
import os
import re
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass


@dataclass
class CoTExample:
    """A single chain-of-thought training example."""
    prompt: str
    reasoning: str
    answer: str
    domain: str = "general"
    difficulty: str = "medium"


# =============================================================================
# Formatting Templates
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step. 
Always show your reasoning process before giving the final answer.
Format your response as:
<reasoning>your step-by-step thinking</reasoning>
<answer>your final answer</answer>"""


def format_training_example(example: CoTExample) -> str:
    """
    Format a CoT example into the required training format.
    
    Output format:
    Q: {prompt}
    A:
    <reasoning>{reasoning}</reasoning>
    <answer>{answer}</answer>
    """
    return f"""Q: {example.prompt}
A:
<reasoning>{example.reasoning}</reasoning>
<answer>{example.answer}</answer>"""


def format_inference_prompt(question: str) -> str:
    """Format a question for inference (no answer)."""
    return f"""Q: {question}
A:
"""


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_gsm8k(filepath: str, split: str = "train") -> List[CoTExample]:
    """
    Load GSM8K dataset and convert to CoT examples.
    
    GSM8K format:
    {"question": "...", "answer": "... #### final_answer"}
    
    The answer contains step-by-step solution followed by #### and the final answer.
    """
    examples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data['question']
            full_answer = data['answer']
            
            # Split reasoning from final answer (GSM8K uses ####)
            if '####' in full_answer:
                reasoning_part, final_answer = full_answer.rsplit('####', 1)
                reasoning = reasoning_part.strip()
                answer = final_answer.strip()
            else:
                reasoning = full_answer
                answer = full_answer.split('\n')[-1].strip()
            
            examples.append(CoTExample(
                prompt=question,
                reasoning=reasoning,
                answer=answer,
                domain="math",
                difficulty="medium"
            ))
    
    return examples


def load_strategyqa(filepath: str) -> List[CoTExample]:
    """
    Load StrategyQA dataset.
    
    StrategyQA format:
    {"question": "...", "answer": true/false, "facts": [...], "decomposition": [...]}
    """
    examples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data['question']
            answer = "Yes" if data['answer'] else "No"
            
            # Build reasoning from decomposition or facts
            decomposition = data.get('decomposition', [])
            facts = data.get('facts', [])
            
            reasoning_parts = []
            if decomposition:
                for i, step in enumerate(decomposition, 1):
                    reasoning_parts.append(f"Step {i}: {step}")
            if facts:
                reasoning_parts.append("Supporting facts: " + "; ".join(facts[:3]))
            
            reasoning = "\n".join(reasoning_parts) if reasoning_parts else "Direct reasoning leads to the answer."
            
            examples.append(CoTExample(
                prompt=question,
                reasoning=reasoning,
                answer=answer,
                domain="reasoning",
                difficulty="medium"
            ))
    
    return examples


def load_generic_cot(filepath: str) -> List[CoTExample]:
    """
    Load generic CoT JSONL format.
    
    Expected format:
    {"prompt": "...", "reasoning": "...", "answer": "...", "domain": "..."}
    """
    examples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(CoTExample(
                prompt=data['prompt'],
                reasoning=data.get('reasoning', ''),
                answer=data['answer'],
                domain=data.get('domain', 'general'),
                difficulty=data.get('difficulty', 'medium')
            ))
    
    return examples


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_training_data(
    examples: List[CoTExample],
    output_path: str,
    include_system_prompt: bool = False
) -> int:
    """
    Prepare and save formatted training data.
    
    Args:
        examples: List of CoT examples
        output_path: Path to save JSONL file
        include_system_prompt: Whether to prepend system prompt
        
    Returns:
        Number of examples written
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            formatted = format_training_example(ex)
            if include_system_prompt:
                formatted = f"{SYSTEM_PROMPT}\n\n{formatted}"
            
            record = {
                'text': formatted,
                'domain': ex.domain,
                'difficulty': ex.difficulty
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    
    return count


def create_train_val_split(
    examples: List[CoTExample],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[CoTExample], List[CoTExample]]:
    """Split examples into training and validation sets."""
    import random
    random.seed(seed)
    
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_ratio)
    return shuffled[val_size:], shuffled[:val_size]


def mix_datasets(
    dataset_configs: List[Dict],
    output_train: str,
    output_val: str,
    val_ratio: float = 0.1
) -> Dict[str, int]:
    """
    Mix multiple datasets with optional sampling weights.
    
    Args:
        dataset_configs: List of dicts with 'path', 'loader', 'weight' keys
        output_train: Path for training output
        output_val: Path for validation output
        val_ratio: Validation split ratio
        
    Returns:
        Dict with counts per dataset
    """
    all_examples = []
    counts = {}
    
    loaders = {
        'gsm8k': load_gsm8k,
        'strategyqa': load_strategyqa,
        'generic': load_generic_cot
    }
    
    for config in dataset_configs:
        loader_fn = loaders.get(config['loader'], load_generic_cot)
        examples = loader_fn(config['path'])
        
        # Apply sampling weight
        weight = config.get('weight', 1.0)
        if weight < 1.0:
            import random
            examples = random.sample(examples, int(len(examples) * weight))
        
        all_examples.extend(examples)
        counts[config['path']] = len(examples)
    
    train, val = create_train_val_split(all_examples, val_ratio)
    
    prepare_training_data(train, output_train)
    prepare_training_data(val, output_val)
    
    counts['total_train'] = len(train)
    counts['total_val'] = len(val)
    
    return counts


# =============================================================================
# Streaming / Batching
# =============================================================================

def iterate_jsonl(filepath: str) -> Iterator[Dict]:
    """Iterate over JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.strip())


def batch_iterator(filepath: str, batch_size: int = 8) -> Iterator[List[Dict]]:
    """Yield batches from JSONL file."""
    batch = []
    for record in iterate_jsonl(filepath):
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# =============================================================================
# Synthetic Data Augmentation
# =============================================================================

def augment_with_paraphrase(
    examples: List[CoTExample],
    paraphrase_fn,  # Function: str -> List[str]
    num_variants: int = 2
) -> List[CoTExample]:
    """
    Augment examples by paraphrasing reasoning steps.
    
    Args:
        examples: Original examples
        paraphrase_fn: Function that generates paraphrases
        num_variants: Number of paraphrased versions per example
        
    Returns:
        Augmented list of examples
    """
    augmented = list(examples)  # Keep originals
    
    for ex in examples:
        paraphrases = paraphrase_fn(ex.reasoning)[:num_variants]
        for para in paraphrases:
            augmented.append(CoTExample(
                prompt=ex.prompt,
                reasoning=para,
                answer=ex.answer,
                domain=ex.domain,
                difficulty=ex.difficulty
            ))
    
    return augmented


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Create sample data
    sample = CoTExample(
        prompt="What is 25 * 4?",
        reasoning="Step 1: Multiply 25 by 4.\nStep 2: 25 * 4 = 100.\nTherefore, the answer is 100.",
        answer="100",
        domain="math"
    )
    
    print("=== Formatted Example ===")
    print(format_training_example(sample))
    
    print("\n=== Inference Prompt ===")
    print(format_inference_prompt("What is 25 * 4?"))
