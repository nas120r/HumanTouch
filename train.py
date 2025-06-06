#!/usr/bin/env python3
"""
HumanTouch Training Script
Fine-tunes Qwen3-0.6B with DoRA for text humanization.

Usage:
    python train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch
"""

import os
import json
import torch
import wandb
import argparse
from datetime import datetime
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
import deepspeed


def setup_model_and_tokenizer(model_name: str):
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
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # No Flash Attention
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    print(f"Model loaded. Parameters: {model.num_parameters():,}")
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
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
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
        desc="Tokenizing"
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


def main():
    parser = argparse.ArgumentParser(description="Train HumanTouch model with DoRA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=128, help="DoRA rank")
    parser.add_argument("--alpha", type=int, default=256, help="DoRA alpha")
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=8e-5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--deepspeed_stage", type=int, default=3)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--run_name", type=str)
    
    args = parser.parse_args()
    
    print("=== HumanTouch DoRA Training ===")
    print(f"Model: {args.model_name}")
    print(f"DoRA Rank: {args.rank}")
    print(f"Max Length: {args.max_length}")
    print(f"Output: {args.output_dir}")
    
    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project="humantouch-dora",
            name=args.run_name or f"dora-r{args.rank}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args)
        )
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    model = setup_dora(model, args.rank, args.alpha)
    
    # Load and tokenize dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)
    
    print(f"Train samples: {len(tokenized_dataset['train'])}")
    print(f"Validation samples: {len(tokenized_dataset['validation'])}")
    
    # Create DeepSpeed config
    deepspeed_config = create_deepspeed_config(args.output_dir, args.deepspeed_stage)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        report_to="wandb" if not args.no_wandb else None,
        
        # Memory optimization
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        bf16=True,
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
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
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Training completed! Model saved to {args.output_dir}")
    
    # Save training history
    if trainer.state.log_history:
        with open(f"{args.output_dir}/training_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)


if __name__ == "__main__":
    main()