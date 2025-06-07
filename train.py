#!/usr/bin/env python3
"""
HumanTouch Training Script
Fine-tunes Qwen models with DoRA for text humanization.

Usage:
    # Interactive mode selection
    python train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch
    
    # Direct mode specification
    python train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode full
"""

import os
import json
import torch
import wandb
import argparse
import gc
import time
from datetime import datetime
from typing import Dict, List
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
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import load_from_disk, Dataset, DatasetDict
import deepspeed


def check_gpu():
    """Check GPU availability and memory."""
    print("GPU Information:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        return True, gpu_memory
    else:
        print("✗ No GPU available - training will be very slow")
        return False, 0


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
            "deepspeed_stage": 2,
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
            "deepspeed_stage": 3,
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
            "deepspeed_stage": 3,
            "description": "Balanced training for smaller GPUs (4 epochs, medium dataset)"
        }
    }
    return configs[training_mode]


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


def setup_model_and_tokenizer(model_name: str, training_mode: str = "full"):
    """Load and configure the model and tokenizer."""
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
    else:
        dtype = torch.float16
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # No Flash Attention
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    print(f"✓ Model loaded. Parameters: {model.num_parameters():,}")
    print(f"✓ Using dtype: {dtype}")
    return model, tokenizer


def setup_dora(model, rank: int = 128, alpha: int = 256):
    """Configure DoRA for the model."""
    print(f"Setting up DoRA with rank {rank}")
    
    # DoRA configuration
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


def tokenize_dataset(dataset, tokenizer, max_length: int = 32768):
    """Tokenize the dataset."""
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
        num_proc=1  # Conservative for stability
    )
    
    return tokenized_dataset


def create_deepspeed_config(output_dir: str, stage: int = 3):
    """Create DeepSpeed configuration."""
    ds_config = {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "cpu_offload": True if stage == 3 else False
        },
        "gradient_clipping": 1.0,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }
    
    # Save config
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/ds_config.json", "w") as f:
        json.dump(ds_config, f, indent=2)
    
    return f"{output_dir}/ds_config.json"


def create_training_args(output_dir: str, config: dict, training_mode: str, use_wandb: bool = True):
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
        
        # Evaluation - fix the save/eval step alignment
        "eval_strategy": "steps",
        "eval_steps": 50 if training_mode == "basic" else 200,
        "save_strategy": "steps",
        "save_steps": 100 if training_mode == "basic" else 400,  # Multiple of eval_steps
        "save_total_limit": 2 if training_mode == "basic" else 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        
        # Logging
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 5 if training_mode == "basic" else 10,
        "logging_first_step": True,
        "report_to": "wandb" if use_wandb else None,
        
        # Memory optimization
        "dataloader_num_workers": 2,
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
    
    # Add DeepSpeed config
    deepspeed_config = create_deepspeed_config(output_dir, config["deepspeed_stage"])
    args["deepspeed"] = deepspeed_config
    
    return TrainingArguments(**args)


def load_and_prepare_dataset(dataset_path: str, config: dict):
    """Load and prepare dataset with size limits."""
    print(f"Loading dataset from {dataset_path}")
    
    # Ensure the path exists
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path not found: {dataset_path}")
        return None
    
    try:
        # Check if it's a HF dataset directory or JSON directory
        if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
            # Load HuggingFace dataset
            dataset = load_from_disk(dataset_path)
            print(f"✓ HuggingFace dataset loaded successfully")
        else:
            # Load from JSON files
            print("Loading from JSON files...")
            
            train_path = os.path.join(dataset_path, "train.json")
            val_path = os.path.join(dataset_path, "validation.json")
            
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                print(f"✗ Required JSON files not found in {dataset_path}")
                return None
                
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
        return None
    
    # Use configured dataset size
    train_size = min(config["train_samples"], len(dataset["train"]))
    val_size = min(config["val_samples"], len(dataset["validation"]))
    
    dataset["train"] = dataset["train"].select(range(train_size))
    dataset["validation"] = dataset["validation"].select(range(val_size))
    
    print(f"✓ Using {len(dataset['train']):,} train samples")
    print(f"✓ Using {len(dataset['validation']):,} validation samples")
    
    return dataset


def clear_memory():
    """Clear GPU and CPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def test_model(model_dir: str, base_model_name: str):
    """Test the trained model with a sample."""
    print("\nTesting trained model...")
    
    try:
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
        return False


def main():
    parser = argparse.ArgumentParser(description="Train HumanTouch model with DoRA")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    parser.add_argument("--mode", type=str, choices=["basic", "full", "smaller_gpu"], 
                      help="Training mode (if not specified, interactive selection)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--run_name", type=str, help="WandB run name")
    parser.add_argument("--test_model", action="store_true", help="Test model after training")
    
    # Override arguments (optional)
    parser.add_argument("--model_name", type=str, help="Override model name")
    parser.add_argument("--rank", type=int, help="Override DoRA rank")
    parser.add_argument("--alpha", type=int, help="Override DoRA alpha") 
    parser.add_argument("--max_length", type=int, help="Override max sequence length")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🤖 HumanTouch DoRA Training")
    print("=" * 80)
    
    # Check GPU
    has_gpu, gpu_memory = check_gpu()
    print()
    
    # Choose or get training mode
    if args.mode:
        training_mode = args.mode
        config = get_training_config(training_mode)
        print(f"🎯 Training Mode: {training_mode.replace('_', ' ').title()}")
        print(f"📝 {config['description']}")
    else:
        training_mode, config = choose_training_mode()
        if not training_mode:
            return
    
    print()
    
    # Apply any overrides
    if args.model_name:
        config["model_name"] = args.model_name
    if args.rank:
        config["rank"] = args.rank
    if args.alpha:
        config["alpha"] = args.alpha
    if args.max_length:
        config["max_length"] = args.max_length
    if args.epochs:
        config["epochs"] = args.epochs
    
    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project="humantouch-dora",
            name=args.run_name or f"{training_mode}-r{config['rank']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={**config, "training_mode": training_mode}
        )
    
    print(f"Starting {training_mode.replace('_', ' ').title()} Training")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Clear memory
    clear_memory()
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config["model_name"], training_mode)
    model = setup_dora(model, config["rank"], config["alpha"])
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(args.dataset_path, config)
    if dataset is None:
        print("❌ Failed to load dataset!")
        return
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, config["max_length"])
    
    # Training arguments
    training_args = create_training_args(args.output_dir, config, training_mode, not args.no_wandb)
    
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
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        print("✓ Training completed successfully!")
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training configuration
        config_save = {
            "training_mode": training_mode,
            "config": config,
            "model_name": config["model_name"],
            "training_time_hours": training_time / 3600,
            "trained_at": datetime.now().isoformat()
        }
        
        with open(f"{args.output_dir}/training_config.json", "w") as f:
            json.dump(config_save, f, indent=2)
        
        # Save training history
        if trainer.state.log_history:
            with open(f"{args.output_dir}/training_history.json", "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
        
        print(f"✓ Model saved to {args.output_dir}")
        print(f"✓ Training time: {training_time/3600:.1f} hours")
        
        # Test model if requested
        if args.test_model:
            test_model(args.output_dir, config["model_name"])
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 80)
        print(f"✓ {training_mode.replace('_', ' ').title()} training completed")
        print(f"✓ Model: {config['model_name']}")
        print(f"✓ DoRA Rank: {config['rank']}")
        print(f"✓ Epochs: {config['epochs']}")
        print(f"✓ Dataset Size: {config['train_samples']:,} samples")
        print(f"✓ Training Time: {training_time/3600:.1f} hours")
        print(f"✓ Model saved to: {args.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Memory cleanup
    clear_memory()


if __name__ == "__main__":
    main()