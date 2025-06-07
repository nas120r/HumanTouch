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
import zipfile
from typing import List, Dict
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm


def load_kaggle_dataset(file_path: str) -> pd.DataFrame:
    """Load the Kaggle AI vs Human text dataset."""
    print(f"Loading dataset from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        if 'generated' in df.columns:
            print(f"  Generated distribution: {df['generated'].value_counts().to_dict()}")
        
        # Show sample data
        print(f"\nSample data:")
        print(df.head(2).to_string())
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text


def filter_by_length(df: pd.DataFrame, max_length: int = 32768) -> pd.DataFrame:
    """Filter texts by length to fit within context window."""
    print("\nFiltering texts by length...")
    
    # Clean texts first
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0].copy()
    
    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
    df['estimated_tokens'] = df['text'].str.len() / 4
    
    # Keep texts that fit in context (leave room for prompt + response)
    max_input_tokens = max_length * 0.6  # Conservative estimate
    df_filtered = df[df['estimated_tokens'] <= max_input_tokens].copy()
    
    # Additional quality filters
    df_filtered = df_filtered[df_filtered['text'].str.len() >= 50]  # Minimum length
    df_filtered = df_filtered[df_filtered['text'].str.len() <= 8192]  # Maximum length for quality
    
    print(f"  Original samples: {len(df):,}")
    print(f"  After length filtering: {len(df_filtered):,}")
    print(f"  Average text length: {df_filtered['text'].str.len().mean():.0f} characters")
    print(f"  Average estimated tokens: {df_filtered['estimated_tokens'].mean():.0f}")
    
    return df_filtered.drop('estimated_tokens', axis=1)


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Balance AI and human texts."""
    print("\nBalancing dataset...")
    
    ai_texts = df[df['generated'] == 1]
    human_texts = df[df['generated'] == 0]
    
    print(f"  AI texts: {len(ai_texts):,}")
    print(f"  Human texts: {len(human_texts):,}")
    
    # Take equal amounts of each
    min_count = min(len(ai_texts), len(human_texts))
    
    # Sample equal amounts
    ai_sample = ai_texts.sample(n=min_count, random_state=42)
    human_sample = human_texts.sample(n=min_count, random_state=42)
    
    balanced_df = pd.concat([ai_sample, human_sample], ignore_index=True)
    
    print(f"  Balanced dataset: {len(balanced_df):,} samples")
    print(f"  Final AI texts: {len(balanced_df[balanced_df['generated'] == 1]):,}")
    print(f"  Final human texts: {len(balanced_df[balanced_df['generated'] == 0]):,}")
    
    return balanced_df


def create_training_pairs(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Create AI->Human training pairs."""
    print("\nCreating training pairs...")
    
    # Separate AI and human texts
    ai_texts = df[df['generated'] == 1]['text'].tolist()
    human_texts = df[df['generated'] == 0]['text'].tolist()
    
    print(f"  AI texts available: {len(ai_texts):,}")
    print(f"  Human texts available: {len(human_texts):,}")
    
    # Create pairs by random sampling
    pairs = []
    min_length = min(len(ai_texts), len(human_texts))
    
    # Set random seed for reproducibility
    np.random.seed(42)
    ai_indices = np.random.permutation(len(ai_texts))[:min_length]
    human_indices = np.random.permutation(len(human_texts))[:min_length]
    
    print(f"  Creating {min_length:,} training pairs...")
    
    for i, (ai_idx, human_idx) in enumerate(tqdm(zip(ai_indices, human_indices), desc="Creating pairs")):
        ai_text = ai_texts[ai_idx]
        human_text = human_texts[human_idx]
        
        # Skip if either text is too short
        if len(ai_text.strip()) < 20 or len(human_text.strip()) < 20:
            continue
        
        # Create conversation format for training
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
            "text": f"<|im_start|>system\nYou are an expert text humanizer. Convert AI-generated text into natural, human-like writing while preserving meaning and context.<|im_end|>\n<|im_start|>user\nHumanize this AI text: {ai_text}<|im_end|>\n<|im_start|>assistant\n{human_text}<|im_end|>",
            "ai_text": ai_text,
            "human_text": human_text,
            "pair_id": i
        }
        pairs.append(sample)
    
    print(f"✓ Created {len(pairs):,} training pairs")
    return pairs


def analyze_dataset(data: List[Dict]) -> Dict:
    """Analyze the processed dataset."""
    print("\nAnalyzing processed dataset...")
    
    # Calculate statistics
    text_lengths = [len(sample['text']) for sample in data]
    ai_text_lengths = [len(sample['ai_text']) for sample in data]
    human_text_lengths = [len(sample['human_text']) for sample in data]
    
    stats = {
        "total_samples": int(len(data)),
        "avg_total_length": float(np.mean(text_lengths)),
        "max_total_length": int(np.max(text_lengths)),
        "min_total_length": int(np.min(text_lengths)),
        "avg_ai_length": float(np.mean(ai_text_lengths)),
        "avg_human_length": float(np.mean(human_text_lengths)),
        "estimated_tokens_avg": float(np.mean(text_lengths) / 4),
        "estimated_tokens_max": float(np.max(text_lengths) / 4)
    }
    
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Average total length: {stats['avg_total_length']:.0f} chars")
    print(f"  Average AI text length: {stats['avg_ai_length']:.0f} chars") 
    print(f"  Average human text length: {stats['avg_human_length']:.0f} chars")
    print(f"  Estimated average tokens: {stats['estimated_tokens_avg']:.0f}")
    print(f"  Estimated max tokens: {stats['estimated_tokens_max']:.0f}")
    
    return stats


def show_sample_data(data: List[Dict], num_samples: int = 2):
    """Show sample processed data."""
    print(f"\nSample processed data (showing {num_samples} examples):")
    print("=" * 80)
    
    for i in range(min(num_samples, len(data))):
        sample = data[i]
        print(f"\nSample {i+1}:")
        print(f"AI Text: {sample['ai_text'][:200]}...")
        print(f"Human Text: {sample['human_text'][:200]}...")
        print(f"Full formatted text length: {len(sample['text'])} characters")
        print("-" * 80)


def split_and_save_dataset(data: List[Dict], output_dir: str):
    """Split dataset and save in multiple formats."""
    print(f"\nSplitting and saving dataset to {output_dir}...")
    
    # Split data (80% train, 10% validation, 10% test)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, shuffle=True)
    
    print(f"  Train samples: {len(train_data):,}")
    print(f"  Validation samples: {len(val_data):,}")
    print(f"  Test samples: {len(test_data):,}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON files
    print("  Saving JSON files...")
    with open(f"{output_dir}/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(f"{output_dir}/validation.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(f"{output_dir}/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # Save as Hugging Face Dataset
    print("  Creating Hugging Face dataset...")
    try:
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        hf_output_dir = f"{output_dir}/hf_dataset"
        dataset_dict.save_to_disk(hf_output_dir)
        print(f"  ✓ Saved Hugging Face dataset to {hf_output_dir}")
        
    except Exception as e:
        print(f"  ✗ Error saving Hugging Face dataset: {e}")
    
    # Save dataset statistics
    stats = analyze_dataset(data)
    with open(f"{output_dir}/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Dataset saved to {output_dir}")
    print("Files created:")
    print("  - train.json, validation.json, test.json")
    print("  - hf_dataset/ (Hugging Face format)")
    print("  - dataset_stats.json")
    
    return len(train_data), len(val_data), len(test_data)


def create_download_package(output_dir: str):
    """Create a zip file with all processed data."""
    print("\nCreating download package...")
    
    zip_filename = f"{output_dir}/humantouch_processed_data.zip"
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from output directory
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.zip'):  # Skip existing zip files
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        
        print(f"✓ Created {zip_filename}")
        print("You can share or backup this complete dataset package")
        return zip_filename
        
    except Exception as e:
        print(f"✗ Failed to create download package: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Process Kaggle AI vs Human dataset")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--max_length", type=int, default=32768, help="Max sequence length")
    parser.add_argument("--create_zip", action="store_true", help="Create zip package for sharing")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HumanTouch Data Processing")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Max sequence length: {args.max_length}")
    print("=" * 80)
    
    # Load and process dataset
    df = load_kaggle_dataset(args.input)
    if df is None:
        print("\n❌ Failed to load dataset.")
        return
    
    # Filter and balance
    df_filtered = filter_by_length(df, args.max_length)
    if len(df_filtered) == 0:
        print("\n❌ No data remaining after filtering.")
        return
    
    df_balanced = balance_dataset(df_filtered)
    
    # Create training pairs
    training_pairs = create_training_pairs(df_balanced)
    if len(training_pairs) == 0:
        print("\n❌ No training pairs created.")
        return
    
    # Show sample data
    show_sample_data(training_pairs, num_samples=2)
    
    # Split and save
    train_count, val_count, test_count = split_and_save_dataset(training_pairs, args.output)
    
    # Create download package if requested
    if args.create_zip:
        create_download_package(args.output)
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 DATA PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"✓ Processed {len(training_pairs):,} training pairs")
    print(f"✓ Train: {train_count:,} samples")
    print(f"✓ Validation: {val_count:,} samples") 
    print(f"✓ Test: {test_count:,} samples")
    print(f"✓ Files saved to {args.output}/")
    print(f"✓ Max sequence length: {args.max_length}")
    if args.create_zip:
        print(f"✓ Download package created")
    print("\nNext steps:")
    print("1. Use the hf_dataset folder for training") 
    print("2. Run training with: python train.py --dataset_path data/processed/hf_dataset")
    print("=" * 80)


if __name__ == "__main__":
    main()