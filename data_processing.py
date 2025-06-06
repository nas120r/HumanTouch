#!/usr/bin/env python3
"""
HumanTouch Data Processing Script
Processes Kaggle AI vs Human text dataset for DoRA training.

Usage:
    python data_processing.py --input data/raw/dataset.csv --output data/processed
"""

import pandas as pd
import json
import os
import argparse
from typing import List, Dict
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset, DatasetDict


def load_kaggle_dataset(file_path: str) -> pd.DataFrame:
    """Load the Kaggle AI vs Human text dataset."""
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Generated distribution: {df['generated'].value_counts()}")
    
    return df


def filter_by_length(df: pd.DataFrame, max_length: int = 32768) -> pd.DataFrame:
    """Filter texts by length to fit within context window."""
    print("Filtering by length...")
    
    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
    df['estimated_tokens'] = df['text'].str.len() / 4
    
    # Keep texts that fit in context (leave room for prompt + response)
    max_input_tokens = max_length * 0.8
    df_filtered = df[df['estimated_tokens'] <= max_input_tokens].copy()
    
    print(f"Original samples: {len(df)}")
    print(f"After filtering: {len(df_filtered)}")
    
    return df_filtered.drop('estimated_tokens', axis=1)


def create_training_pairs(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Create AI->Human training pairs."""
    print("Creating training pairs...")
    
    # Separate AI and human texts
    ai_texts = df[df['generated'] == 1]['text'].tolist()
    human_texts = df[df['generated'] == 0]['text'].tolist()
    
    print(f"AI texts: {len(ai_texts)}")
    print(f"Human texts: {len(human_texts)}")
    
    # Create pairs by random sampling
    pairs = []
    min_length = min(len(ai_texts), len(human_texts))
    
    np.random.seed(42)
    ai_indices = np.random.permutation(len(ai_texts))[:min_length]
    human_indices = np.random.permutation(len(human_texts))[:min_length]
    
    for ai_idx, human_idx in zip(ai_indices, human_indices):
        ai_text = ai_texts[ai_idx]
        human_text = human_texts[human_idx]
        
        # Create conversation format
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert text humanizer. Convert AI-generated text into natural, human-like writing while preserving meaning and context."
                },
                {
                    "role": "user", 
                    "content": f"Humanize this AI text: {ai_text}"
                },
                {
                    "role": "assistant",
                    "content": human_text
                }
            ],
            "text": f"<|im_start|>system\nYou are an expert text humanizer. Convert AI-generated text into natural, human-like writing while preserving meaning and context.<|im_end|>\n<|im_start|>user\nHumanize this AI text: {ai_text}<|im_end|>\n<|im_start|>assistant\n{human_text}<|im_end|>"
        }
        pairs.append(sample)
    
    print(f"Created {len(pairs)} training pairs")
    return pairs


def split_and_save_dataset(data: List[Dict], output_dir: str):
    """Split dataset and save in multiple formats."""
    print("Splitting and saving dataset...")
    
    # Split data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON files
    with open(f"{output_dir}/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(f"{output_dir}/validation.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(f"{output_dir}/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # Save as Hugging Face Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    dataset_dict.save_to_disk(f"{output_dir}/hf_dataset")
    
    print(f"Dataset saved to {output_dir}")
    print("Files created:")
    print("- train.json, validation.json, test.json")
    print("- hf_dataset/ (Hugging Face format)")


def main():
    parser = argparse.ArgumentParser(description="Process Kaggle AI vs Human dataset")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--max_length", type=int, default=32768, help="Max sequence length")
    
    args = parser.parse_args()
    
    # Process data
    df = load_kaggle_dataset(args.input)
    df_filtered = filter_by_length(df, args.max_length)
    training_pairs = create_training_pairs(df_filtered)
    split_and_save_dataset(training_pairs, args.output)
    
    print("\nData processing complete!")
    print(f"Processed {len(training_pairs)} samples")
    print(f"Max sequence length: {args.max_length}")


if __name__ == "__main__":
    main()