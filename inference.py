#!/usr/bin/env python3
"""
HumanTouch Inference Script
Use the trained model to humanize AI-generated text.

Usage:
    python inference.py --model_path models/humantouch --text "AI text to humanize"
    python inference.py --model_path models/humantouch --interactive
    python inference.py --model_path models/humantouch --input_file texts.json --output_file results.json
"""

import os
import json
import torch
import argparse
import time
from typing import List, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer
)
from peft import PeftModel


def load_model(model_path: str, base_model: str = "Qwen/Qwen3-0.6B-Base"):
    """Load the fine-tuned model."""
    print(f"Loading HumanTouch model from {model_path}")
    
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
    
    print("Model loaded successfully!")
    print(f"Device: {next(model.parameters()).device}")
    
    return model, tokenizer


def humanize_text(model, tokenizer, ai_text: str, max_tokens: int = 2048, temperature: float = 0.7, stream: bool = False) -> str:
    """Humanize AI-generated text."""
    # Create prompt
    system_prompt = "You are an expert text humanizer. Convert AI-generated text into natural, human-like writing while preserving meaning and context."
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nHumanize this AI text: {ai_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=30000,
        truncation=True
    ).to(model.device)
    
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    
    # Setup streamer
    streamer = None
    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            streamer=streamer
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|im_start|>assistant\n" in generated_text:
        assistant_start = generated_text.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        result = generated_text[assistant_start:].strip()
        
        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0].strip()
    else:
        result = generated_text[len(prompt):].strip()
    
    # Print stats
    output_tokens = outputs[0].shape[0] - inputs['input_ids'].shape[1]
    tokens_per_sec = output_tokens / generation_time if generation_time > 0 else 0
    print(f"Generated {output_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
    
    return result


def interactive_mode(model, tokenizer):
    """Run interactive humanization."""
    print("\n" + "="*60)
    print("HumanTouch Interactive Mode")
    print("Enter AI text to humanize (type 'quit' to exit)")
    print("Commands: 'stream' (toggle), 'temp X' (set temperature)")
    print("="*60)
    
    stream_mode = False
    temperature = 0.7
    
    while True:
        print(f"\n[Stream: {'ON' if stream_mode else 'OFF'}, Temp: {temperature}]")
        user_input = input("Enter AI text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'stream':
            stream_mode = not stream_mode
            print(f"Streaming mode: {'ON' if stream_mode else 'OFF'}")
            continue
        
        if user_input.startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                print(f"Temperature set to {temperature}")
                continue
            except:
                print("Invalid temperature. Use: temp 0.7")
                continue
        
        if not user_input:
            print("Please enter text to humanize.")
            continue
        
        print("\nHumanizing...")
        print("-" * 40)
        
        try:
            result = humanize_text(model, tokenizer, user_input, temperature=temperature, stream=stream_mode)
            
            if not stream_mode:
                print("Humanized text:")
                print(result)
        
        except Exception as e:
            print(f"Error: {e}")


def process_file(model, tokenizer, input_file: str, output_file: str):
    """Process file with multiple texts."""
    print(f"Processing file: {input_file}")
    
    # Load texts
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                texts = data
            elif isinstance(data, dict) and 'texts' in data:
                texts = data['texts']
            else:
                raise ValueError("JSON must be list of texts or dict with 'texts' key")
        else:
            content = f.read().strip()
            texts = [t.strip() for t in content.split('\n\n') if t.strip()]
    
    print(f"Found {len(texts)} texts to process")
    
    # Process all texts
    results = []
    for i, text in enumerate(texts):
        print(f"\nProcessing {i+1}/{len(texts)}")
        try:
            humanized = humanize_text(model, tokenizer, text)
            results.append({
                'id': i,
                'original': text,
                'humanized': humanized,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            print(f"Error processing text {i+1}: {e}")
            results.append({
                'id': i,
                'original': text,
                'humanized': f"[ERROR: {str(e)}]",
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="HumanTouch Text Humanization")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B-Base")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--text", type=str, help="Single text to humanize")
    mode_group.add_argument("--interactive", action="store_true", help="Interactive mode")
    mode_group.add_argument("--input_file", type=str, help="Input file for batch processing")
    
    # Generation parameters
    parser.add_argument("--output_file", type=str, help="Output file for batch processing")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # Single text mode
    if args.text:
        print("Input text:")
        print(args.text)
        print("\nHumanized text:")
        result = humanize_text(model, tokenizer, args.text, args.max_tokens, args.temperature, args.stream)
        print(result)
    
    # Interactive mode
    elif args.interactive:
        interactive_mode(model, tokenizer)
    
    # File processing mode
    elif args.input_file:
        if not args.output_file:
            args.output_file = f"humanized_{int(time.time())}.json"
        process_file(model, tokenizer, args.input_file, args.output_file)
    
    else:
        print("Please specify --text, --interactive, or --input_file")
        print("Use --help for more information")


if __name__ == "__main__":
    main()