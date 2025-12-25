"""
Tunix Kaggle Hackathon - Checkpoint Export Utilities
=====================================================
Export model checkpoints in Kaggle-compatible format (non-safetensors).
"""

import os
import json
import shutil
from typing import Dict, Optional, List
from datetime import datetime


# =============================================================================
# Checkpoint Management
# =============================================================================

def get_checkpoint_path(
    base_dir: str,
    checkpoint_type: str = "sft",
    iteration: Optional[int] = None,
    metric_value: Optional[float] = None
) -> str:
    """
    Generate a checkpoint directory path.
    
    Args:
        base_dir: Base checkpoints directory
        checkpoint_type: 'sft' or 'rl'
        iteration: Optional iteration/epoch number
        metric_value: Optional metric value for naming
        
    Returns:
        Path to checkpoint directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    name_parts = [checkpoint_type]
    if iteration is not None:
        name_parts.append(f"iter{iteration:04d}")
    if metric_value is not None:
        name_parts.append(f"metric{metric_value:.3f}")
    name_parts.append(timestamp)
    
    checkpoint_name = "_".join(name_parts)
    return os.path.join(base_dir, checkpoint_type, checkpoint_name)


def save_checkpoint_metadata(
    checkpoint_dir: str,
    metadata: Dict
) -> str:
    """
    Save checkpoint metadata.
    
    Args:
        checkpoint_dir: Checkpoint directory
        metadata: Dict with training info
        
    Returns:
        Path to metadata file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add standard fields
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['format'] = 'tunix_gemma'
    
    path = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return path


def list_checkpoints(base_dir: str, checkpoint_type: str = "all") -> List[Dict]:
    """
    List available checkpoints.
    
    Args:
        base_dir: Base checkpoints directory
        checkpoint_type: 'sft', 'rl', or 'all'
        
    Returns:
        List of checkpoint info dicts
    """
    checkpoints = []
    
    types_to_check = ['sft', 'rl'] if checkpoint_type == 'all' else [checkpoint_type]
    
    for ctype in types_to_check:
        type_dir = os.path.join(base_dir, ctype)
        if not os.path.exists(type_dir):
            continue
        
        for name in os.listdir(type_dir):
            ckpt_dir = os.path.join(type_dir, name)
            if not os.path.isdir(ckpt_dir):
                continue
            
            info = {
                'name': name,
                'type': ctype,
                'path': ckpt_dir
            }
            
            # Load metadata if available
            meta_path = os.path.join(ckpt_dir, 'checkpoint_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    info['metadata'] = json.load(f)
            
            checkpoints.append(info)
    
    # Sort by timestamp (most recent first)
    checkpoints.sort(key=lambda x: x['name'], reverse=True)
    return checkpoints


def get_best_checkpoint(
    base_dir: str,
    checkpoint_type: str = "rl",
    metric_key: str = "composite_score"
) -> Optional[str]:
    """
    Find the best checkpoint by metric.
    
    Args:
        base_dir: Base checkpoints directory
        checkpoint_type: 'sft' or 'rl'
        metric_key: Key in metadata to sort by
        
    Returns:
        Path to best checkpoint, or None
    """
    checkpoints = list_checkpoints(base_dir, checkpoint_type)
    
    best = None
    best_score = -float('inf')
    
    for ckpt in checkpoints:
        meta = ckpt.get('metadata', {})
        score = meta.get(metric_key, 0)
        if score > best_score:
            best_score = score
            best = ckpt['path']
    
    return best


# =============================================================================
# Export Functions (Kaggle Compatible)
# =============================================================================

def export_for_kaggle(
    checkpoint_dir: str,
    output_dir: str,
    model_name: str = "gemma_reasoning_model"
) -> Dict:
    """
    Export checkpoint in Kaggle-compatible format.
    
    IMPORTANT: Kaggle requires non-safetensors format.
    This function ensures compatibility with Tunix Gemma loading code.
    
    Args:
        checkpoint_dir: Source checkpoint directory
        output_dir: Output directory for export
        model_name: Name for the exported model
        
    Returns:
        Dict with export info
    """
    os.makedirs(output_dir, exist_ok=True)
    
    export_path = os.path.join(output_dir, model_name)
    os.makedirs(export_path, exist_ok=True)
    
    files_copied = []
    
    # Copy model files (exclude safetensors)
    for filename in os.listdir(checkpoint_dir):
        src = os.path.join(checkpoint_dir, filename)
        dst = os.path.join(export_path, filename)
        
        # Skip safetensors files (Kaggle rules)
        if filename.endswith('.safetensors'):
            continue
        
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            files_copied.append(filename)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            files_copied.append(f"{filename}/")
    
    # Create export manifest
    manifest = {
        'model_name': model_name,
        'source_checkpoint': checkpoint_dir,
        'exported_at': datetime.now().isoformat(),
        'files': files_copied,
        'format': 'tunix_gemma_native',
        'kaggle_compatible': True
    }
    
    manifest_path = os.path.join(export_path, 'export_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    return {
        'export_path': export_path,
        'manifest': manifest
    }


def create_kaggle_model_card(
    export_dir: str,
    model_info: Dict
) -> str:
    """
    Create a Kaggle model card (README.md).
    
    Args:
        export_dir: Export directory
        model_info: Dict with model details
        
    Returns:
        Path to created file
    """
    card = f"""# {model_info.get('title', 'Gemma Reasoning Model')}

## Description
{model_info.get('description', 'Fine-tuned Gemma model for step-by-step reasoning.')}

## Training Details
- **Base Model:** {model_info.get('base_model', 'gemma3-1b')}
- **Training Method:** {model_info.get('method', 'SFT + GRPO')}
- **Training Data:** {model_info.get('training_data', 'GSM8K, StrategyQA')}
- **Epochs:** {model_info.get('epochs', 'N/A')}

## Usage

```python
from tunix import modeling

# Load model
model = modeling.Gemma.from_pretrained("{export_dir}")

# Generate with reasoning
prompt = "Q: What is 2+2?\\nA:"
output = model.generate(prompt, max_length=512)
# Output will have <reasoning>...</reasoning><answer>...</answer> format
```

## Output Format
```
<reasoning>step-by-step thinking</reasoning>
<answer>final answer</answer>
```

## Metrics
| Metric | Value |
|--------|-------|
| Accuracy | {model_info.get('accuracy', 'N/A')} |
| Format Compliance | {model_info.get('format_compliance', 'N/A')} |
| Trace Score | {model_info.get('trace_score', 'N/A')} |

## License
{model_info.get('license', 'Apache 2.0')}
"""
    
    path = os.path.join(export_dir, 'README.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(card)
    
    return path


# =============================================================================
# Validation
# =============================================================================

def validate_checkpoint(checkpoint_dir: str) -> Dict:
    """
    Validate a checkpoint directory.
    
    Checks:
    - Required files present
    - No safetensors (Kaggle rule)
    - Metadata present
    
    Returns:
        Dict with validation results
    """
    issues = []
    warnings = []
    
    if not os.path.exists(checkpoint_dir):
        return {'valid': False, 'issues': ['Checkpoint directory does not exist']}
    
    files = os.listdir(checkpoint_dir)
    
    # Check for safetensors (not allowed on Kaggle)
    safetensors = [f for f in files if f.endswith('.safetensors')]
    if safetensors:
        issues.append(f"Contains safetensors files (not Kaggle compatible): {safetensors}")
    
    # Check for metadata
    if 'checkpoint_metadata.json' not in files:
        warnings.append("Missing checkpoint_metadata.json")
    
    # Check for config
    if 'config.json' not in files:
        warnings.append("Missing config.json")
    
    # Check for model weights (should have some weight files)
    weight_files = [f for f in files if f.endswith('.bin') or f.endswith('.pkl') or f.endswith('.msgpack')]
    if not weight_files:
        issues.append("No model weight files found (.bin, .pkl, or .msgpack)")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'files': files
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint export utilities")
    parser.add_argument('command', choices=['list', 'export', 'validate', 'best'])
    parser.add_argument('--base-dir', default='checkpoints', help='Base checkpoint directory')
    parser.add_argument('--type', default='all', choices=['sft', 'rl', 'all'])
    parser.add_argument('--checkpoint', help='Specific checkpoint path')
    parser.add_argument('--output', help='Output directory for export')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        checkpoints = list_checkpoints(args.base_dir, args.type)
        print(f"Found {len(checkpoints)} checkpoints:")
        for ckpt in checkpoints:
            print(f"  - {ckpt['type']}/{ckpt['name']}")
    
    elif args.command == 'best':
        best = get_best_checkpoint(args.base_dir, args.type if args.type != 'all' else 'rl')
        if best:
            print(f"Best checkpoint: {best}")
        else:
            print("No checkpoints found")
    
    elif args.command == 'validate':
        if not args.checkpoint:
            print("--checkpoint required for validate")
        else:
            result = validate_checkpoint(args.checkpoint)
            print(f"Valid: {result['valid']}")
            if result['issues']:
                print("Issues:", result['issues'])
            if result['warnings']:
                print("Warnings:", result['warnings'])
    
    elif args.command == 'export':
        if not args.checkpoint or not args.output:
            print("--checkpoint and --output required for export")
        else:
            result = export_for_kaggle(args.checkpoint, args.output)
            print(f"Exported to: {result['export_path']}")
