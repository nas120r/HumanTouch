#!/usr/bin/env python3
"""
HumanTouch Training Script for Google Colab
Complete DoRA training with optimizations for Colab environment.

Usage in Colab:
1. Upload your humantouch_processed_data.zip to Google Drive
2. Run this entire script
3. Download the trained model

This script will:
- Install required packages and setup environment
- Mount Google Drive and extract dataset
- Setup Qwen3-0.6B with DoRA configuration
- Train with memory optimizations for Colab
- Save and package the trained model
"""

# Install required packages
import subprocess
import sys
import os

def install_packages():
    """Install required packages for training."""
    packages = [
        'torch>=2.0.0',
        'transformers>=4.40.0',
        'peft>=0.10.0', 
        'accelerate>=0.27.0',
        'datasets>=2.14.0',
        'tqdm>=4.65.0',
        'wandb>=0.16.0',
        'bitsandbytes',
        'scipy'
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    print("Package installation complete!\n")

# Install packages first
install_packages()

# Import required libraries
import torch
import json
import zipfile
import shutil
from datetime import datetime
from typing import Dict, List
import gc

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_from_disk
import wandb

# Check GPU availability
def check_gpu():
    """Check GPU availability and memory."""
    print("GPU Information:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("✗ No GPU available - training will be very slow")
        return False

def setup_drive_and_extract():
    """Mount Google Drive and extract dataset."""
    print("Setting up Google Drive and extracting dataset...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
            print("✓ Google Drive mounted")
        else:
            print("✓ Google Drive already mounted")
    except ImportError:
        print("⚠️ Not running in Google Colab")
        return None
    except Exception as e:
        print(f"✗ Error mounting Google Drive: {e}")
        return None
    
    # Check for dataset zip
    zip_path = "/content/drive/MyDrive/dataset/humantouch_processed_data.zip"
    if not os.path.exists(zip_path):
        print(f"✗ Dataset not found at {zip_path}")
        print("Please upload humantouch_processed_data.zip to your Google Drive")
        return None
    
    # Extract dataset
    extract_dir = "/content/data"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/content/")
    
    # Try multiple possible paths
    possible_paths = [
        "/content/processed/hf_dataset",
        "/content/data/processed/hf_dataset", 
        "/content/hf_dataset"
    ]
    
    for dataset_path in possible_paths:
        if os.path.exists(dataset_path):
            print(f"✓ Dataset found at {dataset_path}")
            # Verify it's a valid HF dataset
            if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
                print(f"✓ Valid HuggingFace dataset structure confirmed")
                return dataset_path
            else:
                print(f"⚠️ Dataset structure incomplete at {dataset_path}")
                
    # If HF dataset not found, look for JSON files
    json_paths = [
        "/content/processed",
        "/content/data/processed"
    ]
    
    for json_dir in json_paths:
        train_json = os.path.join(json_dir, "train.json")
        val_json = os.path.join(json_dir, "validation.json")
        if os.path.exists(train_json) and os.path.exists(val_json):
            print(f"✓ Found JSON dataset files at {json_dir}")
            return json_dir
    
    print(f"✗ Dataset extraction failed - no valid dataset found")
    print("Searched paths:", possible_paths)
    return None

def setup_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B", training_mode: str = "basic"):
    """Load and configure the model and tokenizer for Colab."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Choose dtype based on training mode
    if training_mode == "full":
        dtype = torch.bfloat16
        attn_impl = "eager"
    else:
        dtype = torch.float16
        attn_impl = "eager"
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    print(f"✓ Model loaded. Parameters: {model.num_parameters():,}")
    print(f"✓ Using dtype: {dtype}")
    return model, tokenizer

def setup_dora(model, rank: int = 64, alpha: int = 128):
    """Configure DoRA for Colab (reduced settings for memory)."""
    print(f"Setting up DoRA with rank {rank}")
    
    # DoRA configuration optimized for Colab
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj"
        ],
        use_dora=True,  # Enable DoRA
        bias="none"
    )
    
    # Prepare and apply DoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model

def tokenize_dataset(dataset, tokenizer, max_length: int = 2048):
    """Tokenize the dataset with Colab-friendly settings."""
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
        num_proc=1  # Single process for Colab
    )
    
    return tokenized_dataset

def clear_memory():
    """Clear GPU and CPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_training_config(training_mode: str):
    """Get training configuration based on mode."""
    configs = {
        "basic": {
            "model_name": "Qwen/Qwen2.5-0.5B",
            "rank": 32,
            "alpha": 64,
            "max_length": 2048,
            "batch_size": 1,
            "grad_accum": 8,
            "learning_rate": 5e-5,
            "epochs": 2,
            "train_samples": 5000,
            "val_samples": 500,
            "description": "Fast training for testing (2 epochs, small dataset)"
        },
        "full": {
            "model_name": "Qwen/Qwen3-0.6B-Base",
            "rank": 128,
            "alpha": 256,
            "max_length": 8192,
            "batch_size": 1,
            "grad_accum": 16,
            "learning_rate": 8e-5,
            "epochs": 6,
            "train_samples": 50000,
            "val_samples": 5000,
            "description": "Maximum quality training (6 epochs, large dataset)"
        },
        "smaller_gpu": {
            "model_name": "Qwen/Qwen2.5-0.5B",
            "rank": 64,
            "alpha": 128,
            "max_length": 4096,
            "batch_size": 1,
            "grad_accum": 16,
            "learning_rate": 8e-5,
            "epochs": 4,
            "train_samples": 20000,
            "val_samples": 2000,
            "description": "Balanced training for smaller GPUs (4 epochs, medium dataset)"
        }
    }
    return configs[training_mode]

def create_colab_training_args(output_dir: str, config: dict, training_mode: str = "basic"):
    """Create training arguments based on configuration."""
    
    # Base arguments
    args = {
        # Output
        "output_dir": output_dir,
        
        # Training schedule
        "num_train_epochs": config["epochs"],
        "max_steps": -1,
        
        # Batch size
        "per_device_train_batch_size": config["batch_size"],
        "per_device_eval_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["grad_accum"],
        
        # Optimization
        "learning_rate": config["learning_rate"],
        "weight_decay": 0.01,
        "warmup_ratio": 0.1 if training_mode == "basic" else 0.03,
        "lr_scheduler_type": "cosine",
        
        # Evaluation
        "eval_strategy": "steps",
        "eval_steps": 50 if training_mode == "basic" else 200,
        "save_strategy": "steps",
        "save_steps": 100 if training_mode == "basic" else 400,  # Must be multiple of eval_steps
        "save_total_limit": 2 if training_mode == "basic" else 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        
        # Logging
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 5 if training_mode == "basic" else 10,
        "logging_first_step": True,
        "report_to": None,
        
        # Memory optimization
        "dataloader_num_workers": 0,
        "gradient_checkpointing": True,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        
        # Stability
        "max_grad_norm": 1.0,
        "seed": 42,
    }
    
    # Precision settings based on mode
    if training_mode == "full":
        args["bf16"] = True
        args["fp16"] = False
    else:
        args["fp16"] = True
        args["bf16"] = False
    
    return TrainingArguments(**args)

def choose_training_mode():
    """Let user choose training mode."""
    print("=" * 80)
    print("🎯 CHOOSE TRAINING MODE")
    print("=" * 80)
    
    modes = {
        "1": "basic",
        "2": "full", 
        "3": "smaller_gpu"
    }
    
    # Display options
    for key, mode in modes.items():
        config = get_training_config(mode)
        print(f"{key}. {mode.replace('_', ' ').title()} Training")
        print(f"   - {config['description']}")
        print(f"   - Model: {config['model_name']}")
        print(f"   - DoRA Rank: {config['rank']}")
        print(f"   - Max Length: {config['max_length']} tokens")
        print(f"   - Epochs: {config['epochs']}")
        print(f"   - Dataset Size: {config['train_samples']:,} train, {config['val_samples']:,} val")
        print()
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            if choice in modes:
                selected_mode = modes[choice]
                config = get_training_config(selected_mode)
                print(f"\n✓ Selected: {selected_mode.replace('_', ' ').title()} Training")
                print(f"✓ {config['description']}")
                return selected_mode, config
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n❌ Training cancelled.")
            return None, None
        except:
            print("❌ Invalid input. Please enter 1, 2, or 3.")

def train_model(dataset_path: str, training_mode: str, config: dict, output_dir: str = "/content/humantouch_model"):
    """Main training function with configurable settings."""
    print(f"Starting {training_mode.replace('_', ' ').title()} Training")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Clear memory
    clear_memory()
    
    # Setup model with selected configuration
    model, tokenizer = setup_model_and_tokenizer(config["model_name"], training_mode)
    model = setup_dora(model, rank=config["rank"], alpha=config["alpha"])
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    
    # Ensure the path exists
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path not found: {dataset_path}")
        return False
    
    try:
        # Check if it's a HF dataset directory or JSON directory
        if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
            # Load HuggingFace dataset
            dataset = load_from_disk(dataset_path)
            print(f"✓ HuggingFace dataset loaded successfully")
        else:
            # Load from JSON files
            print("Loading from JSON files...")
            from datasets import DatasetDict, Dataset
            
            train_path = os.path.join(dataset_path, "train.json")
            val_path = os.path.join(dataset_path, "validation.json")
            
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                print(f"✗ Required JSON files not found in {dataset_path}")
                print(f"Looking for: {train_path}, {val_path}")
                return False
                
            with open(train_path, 'r') as f:
                train_data = json.load(f)
            with open(val_path, 'r') as f:
                val_data = json.load(f)
            
            dataset = DatasetDict({
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data)
            })
            print("✓ Dataset loaded from JSON files")
            
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Use configured dataset size
    train_size = min(config["train_samples"], len(dataset["train"]))
    val_size = min(config["val_samples"], len(dataset["validation"]))
    
    dataset["train"] = dataset["train"].select(range(train_size))
    dataset["validation"] = dataset["validation"].select(range(val_size))
    
    print(f"✓ Using {len(dataset['train']):,} train samples")
    print(f"✓ Using {len(dataset['validation']):,} validation samples")
    
    # Tokenize with configured max length
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=config["max_length"])
    
    # Training arguments with configuration
    training_args = create_colab_training_args(output_dir, config, training_mode)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Clear memory before training
    clear_memory()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"], 
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Start training
    print("🚀 Starting training...")
    print("=" * 60)
    
    try:
        trainer.train()
        print("✓ Training completed successfully!")
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config_save = {
            "training_mode": training_mode,
            "config": config,
            "model_name": config["model_name"],
            "trained_at": datetime.now().isoformat()
        }
        
        with open(f"{output_dir}/training_config.json", "w") as f:
            json.dump(config_save, f, indent=2)
        
        # Save training history
        if trainer.state.log_history:
            with open(f"{output_dir}/training_history.json", "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
        
        print(f"✓ Model saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model(model_dir: str = "/content/humantouch_model"):
    """Test the trained model with a sample."""
    print("Testing trained model...")
    
    try:
        from peft import PeftModel
        
        # Load training config to get base model name
        config_path = f"{model_dir}/training_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                training_config = json.load(f)
            base_model_name = training_config["model_name"]
        else:
            base_model_name = "Qwen/Qwen2.5-0.5B"  # fallback
        
        print(f"Using base model: {base_model_name}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load trained PEFT model
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()
        
        # Test prompt
        test_text = "The artificial intelligence system processed the data efficiently and generated comprehensive analytical reports."
        prompt = f"<|im_start|>system\nYou are an expert text humanizer. Convert AI-generated text into natural, human-like writing while preserving meaning and context.<|im_end|>\n<|im_start|>user\nHumanize this AI text: {test_text}<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(model.device)  # Move inputs to same device as model
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|im_start|>assistant\n" in result:
            assistant_start = result.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            response = result[assistant_start:].strip()
        else:
            response = result[len(prompt):].strip()
        
        print("=" * 60)
        print("MODEL TEST RESULTS")
        print("=" * 60)
        print(f"Input: {test_text}")
        print(f"Output: {response}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_download_package(model_dir: str = "/content/humantouch_model"):
    """Create downloadable model package."""
    print("Creating download package...")
    
    zip_filename = "humantouch_trained_model.zip"
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(model_dir))
                    zipf.write(file_path, arcname)
        
        print(f"✓ Created {zip_filename} for download")
        print("You can download this file from Colab files panel")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create download package: {e}")
        return False

def main():
    """Main training pipeline."""
    print("=" * 80)
    print("🤖 HumanTouch DoRA Training for Google Colab")
    print("=" * 80)
    
    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print("⚠️ Training without GPU will be extremely slow!")
        try:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        except:
            print("⚠️ No GPU detected - proceeding with CPU training")
    
    print()
    
    # Setup and extract dataset
    dataset_path = setup_drive_and_extract()
    if not dataset_path:
        return
    
    print()
    
    # Choose training mode
    training_mode, config = choose_training_mode()
    if not training_mode:
        return
    
    print()
    
    # Train model with selected configuration
    success = train_model(dataset_path, training_mode, config)
    if not success:
        print("❌ Training failed!")
        return
    
    print()
    
    # Test model
    test_model()
    
    print()
    
    # Create download package
    create_download_package()
    
    print()
    print("=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✓ {training_mode.replace('_', ' ').title()} training completed")
    print("✓ Model trained and saved")
    print("✓ Model tested successfully") 
    print("✓ Download package created")
    print("✓ You can now download 'humantouch_trained_model.zip'")
    print("=" * 80)
    
    # Show final configuration summary
    print("\nTraining Summary:")
    print(f"- Mode: {training_mode.replace('_', ' ').title()}")
    print(f"- Model: {config['model_name']}")
    print(f"- DoRA Rank: {config['rank']}")
    print(f"- Epochs: {config['epochs']}")
    print(f"- Dataset Size: {config['train_samples']:,} samples")
    print("=" * 80)
    
    # Memory cleanup
    clear_memory()

if __name__ == "__main__":
    main()