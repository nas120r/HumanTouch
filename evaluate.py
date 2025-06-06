#!/usr/bin/env python3
"""
HumanTouch Evaluation Script
Evaluates the fine-tuned model on humanization quality.

Usage:
    python evaluate.py --model_path models/humantouch --test_dataset data/processed/hf_dataset
"""

import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel
from datasets import load_from_disk

# Optional evaluation metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: Install rouge-score and nltk for detailed metrics")
    METRICS_AVAILABLE = False


def load_model(model_path: str, base_model: str):
    """Load the fine-tuned model."""
    print(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    # Load PEFT model
    model = PeftModel.from_pretrained(
        base_model_obj,
        model_path,
        torch_dtype=torch.bfloat16
    )
    
    model.eval()
    print("Model loaded successfully")
    
    return model, tokenizer


def extract_texts_from_sample(sample: Dict) -> tuple:
    """Extract AI and human texts from a sample."""
    # Extract AI text from user message
    ai_text = ""
    human_text = ""
    
    if "messages" in sample:
        for message in sample["messages"]:
            if message["role"] == "user":
                content = message["content"]
                if "Humanize this AI text:" in content:
                    ai_text = content.replace("Humanize this AI text:", "").strip()
            elif message["role"] == "assistant":
                human_text = message["content"]
    
    return ai_text, human_text


def generate_humanized_text(model, tokenizer, ai_text: str, max_tokens: int = 1024) -> str:
    """Generate humanized text from AI text."""
    # Create prompt
    prompt = f"<|im_start|>system\nYou are an expert text humanizer. Convert AI-generated text into natural, human-like writing while preserving meaning and context.<|im_end|>\n<|im_start|>user\nHumanize this AI text: {ai_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=30000,
        truncation=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # Decode and extract
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|im_start|>assistant\n" in generated_text:
        assistant_start = generated_text.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        result = generated_text[assistant_start:].strip()
        
        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0].strip()
        
        return result
    
    return generated_text.strip()


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    
    if METRICS_AVAILABLE:
        # ROUGE scores
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = rouge_scorer_obj.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        metrics.update({
            'rouge1_f1': np.mean(rouge_scores['rouge1']),
            'rouge2_f1': np.mean(rouge_scores['rouge2']),
            'rougeL_f1': np.mean(rouge_scores['rougeL'])
        })
        
        # BLEU scores
        smoothing_function = SmoothingFunction().method1
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]
                bleu_score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing_function)
                bleu_scores.append(bleu_score)
            except:
                bleu_scores.append(0.0)
        
        metrics['bleu_score'] = np.mean(bleu_scores)
    
    # Basic metrics
    metrics.update({
        'num_samples': len(predictions),
        'avg_prediction_length': np.mean([len(p.split()) for p in predictions]),
        'avg_reference_length': np.mean([len(r.split()) for r in references])
    })
    
    return metrics


def save_results(results: Dict, predictions: List[str], references: List[str], ai_texts: List[str], output_dir: str):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save examples
    examples = []
    for i, (ai_text, pred, ref) in enumerate(zip(ai_texts, predictions, references)):
        examples.append({
            'sample_id': i,
            'ai_text': ai_text,
            'predicted_human': pred,
            'reference_human': ref
        })
    
    with open(f"{output_dir}/examples.json", "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    # Save CSV
    df = pd.DataFrame(examples)
    df.to_csv(f"{output_dir}/examples.csv", index=False)
    
    print(f"Results saved to {output_dir}")


def print_results(results: Dict):
    """Print formatted results."""
    print("\n" + "="*50)
    print("HumanTouch Evaluation Results")
    print("="*50)
    
    print(f"Samples evaluated: {results.get('num_samples', 'N/A')}")
    print(f"Avg prediction length: {results.get('avg_prediction_length', 'N/A'):.1f} words")
    print(f"Avg reference length: {results.get('avg_reference_length', 'N/A'):.1f} words")
    
    if 'rouge1_f1' in results:
        print("\nFluency Metrics:")
        print(f"ROUGE-1 F1: {results['rouge1_f1']:.4f}")
        print(f"ROUGE-2 F1: {results['rouge2_f1']:.4f}")
        print(f"ROUGE-L F1: {results['rougeL_f1']:.4f}")
        print(f"BLEU Score: {results['bleu_score']:.4f}")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HumanTouch model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--text", type=str, help="Single text for quick test")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # Quick test mode
    if args.quick_test and args.text:
        print("Quick test mode:")
        print(f"Input: {args.text}")
        result = generate_humanized_text(model, tokenizer, args.text)
        print(f"Output: {result}")
        return
    
    # Load test dataset
    print(f"Loading test dataset from {args.test_dataset}")
    dataset = load_from_disk(args.test_dataset)
    test_data = dataset["test"]
    
    if args.max_samples:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))
    
    print(f"Evaluating on {len(test_data)} samples")
    
    # Generate predictions
    predictions = []
    references = []
    ai_texts = []
    
    for sample in tqdm(test_data, desc="Generating predictions"):
        ai_text, human_text = extract_texts_from_sample(sample)
        
        if ai_text and human_text:
            try:
                pred = generate_humanized_text(model, tokenizer, ai_text, args.max_tokens)
                predictions.append(pred)
                references.append(human_text)
                ai_texts.append(ai_text)
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    print(f"Generated {len(predictions)} predictions")
    
    # Compute metrics
    results = compute_metrics(predictions, references)
    
    # Print and save results
    print_results(results)
    save_results(results, predictions, references, ai_texts, args.output_dir)


if __name__ == "__main__":
    main()