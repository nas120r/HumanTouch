# HumanTouch: AI Text Humanization with DoRA

Transform AI-generated text into natural, human-like writing using DoRA fine-tuned Qwen3-0.6B with 32k context support.

## 🎯 What is HumanTouch?

HumanTouch solves the problem of robotic, repetitive AI-generated content by converting it into natural, flowing human-like text while preserving the original meaning and context.

**Key Features:**
- 🧠 **DoRA Fine-tuning** - Advanced weight decomposition (rank 128) for superior quality
- 📝 **32k Context** - Handle long documents without truncation  
- 🎯 **High Quality** - No quantization or Flash Attention compromises
- 🚀 **Multi-GPU Support** - Scale training across 2-8+ GPUs with DeepSpeed
- ⚡ **Easy to Use** - Simple Python scripts for training and inference
- 📈 **Linear Scaling** - Near-linear speedup with multiple GPUs

**The Problem:** AI text often sounds robotic, repetitive, and lacks natural human flow.

**The Solution:** HumanTouch learns from 500K human vs AI text pairs to add natural variations, improve flow, and maintain context coherence.

## 🚀 Multi-GPU Training Benefits

**Why Multi-GPU?**
- **Faster Training**: 2-4x speedup with 4 GPUs, 4-8x speedup with 8 GPUs
- **Larger Models**: Train bigger models that don't fit on single GPU
- **Better Quality**: Larger effective batch sizes improve convergence
- **Cost Efficiency**: Reduce training time and cloud costs significantly

**Automatic Optimizations:**
- **Learning Rate Scaling**: Automatic sqrt scaling for stable multi-GPU training
- **Gradient Accumulation**: Smart adjustment to maintain effective batch size
- **DeepSpeed Integration**: ZeRO-2/3 optimization based on GPU count
- **Communication Efficient**: Optimized data transfers between GPUs

## 📋 Requirements

### Hardware

**Single-GPU Setup:**
- **GPU**: A100 80GB, H100, or RTX 4090 24GB+ 
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space

**Multi-GPU Setup (2-8 GPUs):**
- **GPU**: 2-8x A100 80GB, H100, RTX 4090 24GB+, or RTX A6000
- **RAM**: 64GB+ system memory (scales with GPU count)
- **Storage**: 200GB+ free space
- **Network**: High-speed interconnect (NVLink, InfiniBand) recommended for 4+ GPUs
- **PSU**: Adequate power supply for multiple high-end GPUs

### Software
- **Python**: 3.10 or higher
- **CUDA**: 11.8+ or 12.1+ (for GPU support)
- **UV**: Fast Python package manager

## 🚀 Manual Setup (Windows/Linux/Mac)

### Step 1: Install UV Package Manager

**Windows:**
```cmd
# PowerShell (as Administrator)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

**Linux/Mac:**
```bash
# Via curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip  
pip install uv
```

**Verify installation:**
```bash
uv --version
```

### Step 2: Clone and Setup Project

```bash
# Clone repository
git clone https://github.com/your-username/HumanTouch.git
cd HumanTouch

# Create virtual environment
uv venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### Step 3: Install Dependencies

**Install PyTorch (choose your platform):**

```bash
# CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1  
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
uv pip install torch torchvision torchaudio
```

**Install project dependencies:**
```bash
uv pip install -r requirements.txt
```

**Install DeepSpeed:**
```bash
uv pip install deepspeed
```

**Verify GPU setup:**
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Step 4: Create Directories

```bash
# Windows
mkdir data\raw data\processed data\evaluation models logs

# Linux/Mac  
mkdir -p data/{raw,processed,evaluation} models logs
```

### Step 5: Download Dataset

1. **Go to Kaggle**: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
2. **Download** the dataset (CSV file)
3. **Place** the CSV file in `data/raw/` folder
4. **Rename** it to `AI_Human_Text.csv` (if needed)

**Alternative - Kaggle API:**
```bash
# Install Kaggle API
uv pip install kaggle

# Setup API key (see Kaggle docs)
kaggle datasets download -d shanegerami/ai-vs-human-text -p data/raw/
```

### Step 6: Process Dataset

**Basic processing:**
```bash
python data_processing.py --input data/raw/AI_Human_Text.csv --output data/processed
```

**With optional zip package creation:**
```bash
python data_processing.py --input data/raw/AI_Human_Text.csv --output data/processed --create_zip
```

**Expected output:**
```
================================================================================
HumanTouch Data Processing
================================================================================
Input file: data/raw/AI_Human_Text.csv
Output directory: data/processed
Max sequence length: 32768
================================================================================
✓ Dataset loaded successfully!
  Shape: (487235, 2)
  Columns: ['text', 'generated']
  Generated distribution: {0.0: 305797, 1.0: 181438}

Filtering texts by length...
  Original samples: 487,231
  After length filtering: 487,182
  Average text length: 2263 characters
  Average estimated tokens: 566

Balancing dataset...
  AI texts: 181,419
  Human texts: 305,763
  Balanced dataset: 362,838 samples
  Final AI texts: 181,419
  Final human texts: 181,419

Creating training pairs...
  AI texts available: 181,419
  Human texts available: 181,419
  Creating 181,419 training pairs...
Creating pairs: 100%|████████████| 181419/181419 [00:01<00:00, 109313.90it/s]
✓ Created 181,419 training pairs

Sample processed data (showing 2 examples):
================================================================================
Sample 1:
AI Text: The artificial intelligence system processed...
Human Text: Our AI tool crunched through the data...
Full formatted text length: 5832 characters

Splitting and saving dataset to data/processed...
  Train samples: 145,135
  Validation samples: 18,142
  Test samples: 18,142
  Saving JSON files...
  Creating Hugging Face dataset...
  ✓ Saved Hugging Face dataset to data/processed/hf_dataset

✓ Dataset saved to data/processed
Files created:
  - train.json, validation.json, test.json
  - hf_dataset/ (Hugging Face format)
  - dataset_stats.json

🎉 DATA PROCESSING COMPLETE!
================================================================================
✓ Processed 181,419 training pairs
✓ Train: 145,135 samples
✓ Validation: 18,142 samples
✓ Test: 18,142 samples
✓ Files saved to data/processed/
✓ Max sequence length: 32768
================================================================================
```

### Step 7: Train Model

**🎯 Single-GPU Training (Interactive Mode Selection):**
```bash
python train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch
```

**🚀 Multi-GPU Training with DeepSpeed:**
```bash
# 2 GPUs
deepspeed --num_gpus=2 train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode full

# 4 GPUs  
deepspeed --num_gpus=4 train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode full

# 8 GPUs (for high-end setups)
deepspeed --num_gpus=8 train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode full

# Interactive mode selection with multi-GPU
deepspeed --num_gpus=4 train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch
```

**Training Mode Options:**
Both single-GPU and multi-GPU training offer 3 optimized configurations:

```
🎯 CHOOSE TRAINING MODE
================================================================================
🚀 Multi-GPU Mode: 4 GPUs Detected
================================================================================
1. Basic Training
   - Fast training for testing (2 epochs, small dataset) (Multi-GPU: 4 GPUs)
   - Model: Qwen/Qwen2.5-0.5B
   - DoRA Rank: 32
   - Max Length: 2048 tokens
   - Epochs: 2
   - Dataset Size: 5,000 train, 500 val
   - Learning Rate (scaled): 1.00e-04
   - Grad Accumulation (adjusted): 2

2. Full Training
   - Maximum quality training (6 epochs, large dataset) (Multi-GPU: 4 GPUs)
   - Model: Qwen/Qwen3-0.6B-Base
   - DoRA Rank: 128
   - Max Length: 8192 tokens
   - Epochs: 6
   - Dataset Size: 50,000 train, 5,000 val
   - Learning Rate (scaled): 1.60e-04
   - Grad Accumulation (adjusted): 4

3. Smaller Gpu Training
   - Balanced training for smaller GPUs (4 epochs, medium dataset) (Multi-GPU: 4 GPUs)
   - Model: Qwen/Qwen2.5-0.5B
   - DoRA Rank: 64
   - Max Length: 4096 tokens
   - Epochs: 4
   - Dataset Size: 20,000 train, 2,000 val
   - Learning Rate (scaled): 1.60e-04
   - Grad Accumulation (adjusted): 4

Enter your choice (1-3):
```

**🚀 Direct Mode Specification:**

**Single-GPU:**
```bash
# Basic training (fast, for testing)
python train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode basic

# Full training (maximum quality)
python train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode full

# Smaller GPU training (balanced)
python train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode smaller_gpu
```

**Multi-GPU with DeepSpeed:**
```bash
# Basic training (2 GPUs)
deepspeed --num_gpus=2 train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode basic

# Full training (4 GPUs)
deepspeed --num_gpus=4 train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode full

# Smaller GPU training (2 GPUs)
deepspeed --num_gpus=2 train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch --mode smaller_gpu
```

**⚙️ Advanced Options with Overrides:**

**Single-GPU:**
```bash
# Custom configuration with testing
python train.py \
    --dataset_path data/processed/hf_dataset \
    --output_dir models/humantouch \
    --mode full \
    --epochs 8 \
    --rank 256 \
    --test_model \
    --run_name "custom-experiment"

# Training with specific model
python train.py \
    --dataset_path data/processed/hf_dataset \
    --output_dir models/humantouch \
    --mode smaller_gpu \
    --model_name "Qwen/Qwen2.5-1.5B" \
    --max_length 8192
```

**Multi-GPU:**
```bash
# Custom multi-GPU configuration
deepspeed --num_gpus=4 train.py \
    --dataset_path data/processed/hf_dataset \
    --output_dir models/humantouch \
    --mode full \
    --epochs 8 \
    --rank 256 \
    --test_model \
    --run_name "multi-gpu-experiment"

# High-end 8-GPU setup
deepspeed --num_gpus=8 train.py \
    --dataset_path data/processed/hf_dataset \
    --output_dir models/humantouch \
    --mode full \
    --model_name "Qwen/Qwen2.5-1.5B" \
    --max_length 8192
```

**📊 Training Features:**
- **Multi-GPU Support**: Automatic multi-GPU detection and DeepSpeed integration (2-8+ GPUs)
- **GPU Detection**: Automatic GPU memory and capability detection
- **Learning Rate Scaling**: Automatic sqrt scaling for multi-GPU setups
- **Gradient Accumulation**: Smart adjustment to maintain effective batch size
- **Progress Monitoring**: Real-time training progress with WandB integration
- **Memory Optimization**: Automatic DeepSpeed configuration based on mode and GPU count
- **Communication Optimization**: Efficient data transfers for multi-GPU training
- **Error Handling**: Comprehensive error recovery and debugging
- **Model Testing**: Built-in model testing after training completion
- **Configuration Saving**: All training settings saved for reproducibility

**⏱️ Expected Training Times:**

**Single-GPU:**
- **Basic Mode**: 30-60 minutes (testing/prototyping)
- **Smaller GPU Mode**: 8-15 hours (RTX 4090, etc.)
- **Full Mode**: 15-25 hours (A100/H100)

**Multi-GPU (with linear scaling):**
- **Basic Mode**: 15-30 minutes (2-4 GPUs)
- **Smaller GPU Mode**: 2-8 hours (2-4 GPUs)
- **Full Mode**: 4-12 hours (2-4 GPUs), 2-6 hours (4-8 GPUs)

**💡 Training Tips:**

**Single-GPU:**
```bash
# Monitor GPU usage
nvidia-smi

# Training with custom WandB project
python train.py --dataset_path data/processed --output_dir models/humantouch --mode full --run_name "production-v1"

# Disable WandB logging
python train.py --dataset_path data/processed --output_dir models/humantouch --mode basic --no_wandb

# Test model after training
python train.py --dataset_path data/processed --output_dir models/humantouch --mode basic --test_model
```

**Multi-GPU:**
```bash
# Monitor all GPUs
watch -n 1 nvidia-smi

# Multi-GPU training with WandB
deepspeed --num_gpus=4 train.py --dataset_path data/processed --output_dir models/humantouch --mode full --run_name "multi-gpu-v1"

# Check DeepSpeed installation
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"

# Multi-GPU test after training
deepspeed --num_gpus=2 train.py --dataset_path data/processed --output_dir models/humantouch --mode basic --test_model
```

### Step 8: Evaluate Model

**Built-in model testing (during training):**
```bash
# Test model automatically after training
python train.py --dataset_path data/processed --output_dir models/humantouch --mode basic --test_model
```

**Comprehensive evaluation:**
```bash
# Full evaluation suite
python evaluate.py \
    --model_path models/humantouch \
    --test_dataset data/processed/hf_dataset \
    --max_samples 500 \
    --output_dir evaluation_results

# Quick single text test
python evaluate.py \
    --model_path models/humantouch \
    --quick_test \
    --text "The artificial intelligence system processed the data efficiently and generated comprehensive analytical reports."
```

**Expected test output:**
```
============================================================
MODEL TEST RESULTS
============================================================
Input: The artificial intelligence system processed the data efficiently...
Output: Our AI tool crunched through the data pretty efficiently and came up with some detailed analysis reports.
============================================================
```

### Step 9: Use for Inference

**Interactive mode:**
```bash
python inference.py --model_path models/humantouch --interactive
```

**Single text:**
```bash
python inference.py \
    --model_path models/humantouch \
    --text "Artificial intelligence algorithms demonstrate significant capabilities in processing and analyzing large datasets."
```

**Batch processing:**
```bash
# Create input file
echo ["Text 1 to humanize", "Text 2 to humanize"] > input_texts.json

# Process batch
python inference.py \
    --model_path models/humantouch \
    --input_file input_texts.json \
    --output_file humanized_results.json
```

## 📁 Project Structure

```
HumanTouch/
├── data_processing.py      # Enhanced dataset preparation with progress tracking
├── train.py               # Interactive DoRA training with 3 modes
├── evaluate.py            # Model evaluation script
├── inference.py           # Text humanization script
├── requirements.txt       # Python dependencies
├── configs/
│   └── dora_config.yaml   # DoRA configuration settings
├── colab_ready/           # Google Colab versions
│   ├── data_processing_colab.py
│   └── train_colab.py
├── data/
│   ├── raw/              # Original Kaggle dataset
│   ├── processed/        # Processed training data
│   └── evaluation/       # Evaluation results
├── models/               # Trained model checkpoints
└── logs/                 # Training logs
```

## ⚙️ Configuration Options

### 🎯 Training Modes (Interactive)

The new interactive training system provides three optimized configurations:

```bash
# Interactive mode selection (recommended)
python train.py --dataset_path data/processed --output_dir models/humantouch

# Direct mode selection
python train.py --dataset_path data/processed --output_dir models/humantouch --mode [basic|full|smaller_gpu]
```

**Mode Comparison:**

| Mode | Model | Rank | Length | Epochs | Time | Quality |
|------|-------|------|--------|--------|------|----------|
| **Basic** | Qwen2.5-0.5B | 32 | 2048 | 2 | 30-60min | Testing |
| **Smaller GPU** | Qwen2.5-0.5B | 64 | 4096 | 4 | 8-15hrs | Good |
| **Full** | Qwen3-0.6B | 128 | 8192 | 6 | 15-25hrs | Best |

### 🔧 Advanced Configuration

**Custom parameter overrides:**
```bash
# Override specific parameters
python train.py \
    --dataset_path data/processed \
    --output_dir models/humantouch \
    --mode full \
    --rank 256 \
    --epochs 8 \
    --max_length 16384 \
    --model_name "Qwen/Qwen2.5-1.5B"

# Memory-optimized training
python train.py \
    --dataset_path data/processed \
    --output_dir models/humantouch \
    --mode smaller_gpu \
    --max_length 2048 \
    --rank 32
```

### Inference Parameters

```bash
# High quality (slower)
python inference.py --model_path models/humantouch --max_tokens 2048 --temperature 0.7

# Faster generation
python inference.py --model_path models/humantouch --max_tokens 1024 --temperature 0.8

# More creative
python inference.py --model_path models/humantouch --temperature 0.9

# More conservative  
python inference.py --model_path models/humantouch --temperature 0.5
```

## 🛠️ Troubleshooting

### GPU Memory Issues

**Single-GPU - CUDA out of memory during training:**
```bash
# Solution 1: Use Basic mode for testing
python train.py --dataset_path data/processed --output_dir models/humantouch --mode basic

# Solution 2: Reduce parameters manually
python train.py --dataset_path data/processed --output_dir models/humantouch --mode smaller_gpu --max_length 2048 --rank 32

# Solution 3: Check GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
```

**Multi-GPU - Memory or communication issues:**
```bash
# Solution 1: Check all GPUs
nvidia-smi

# Solution 2: Test with fewer GPUs
deepspeed --num_gpus=2 train.py --dataset_path data/processed --output_dir models/humantouch --mode basic

# Solution 3: Check GPU memory across all devices
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB') for i in range(torch.cuda.device_count())]"

# Solution 4: Use smaller model for multi-GPU
deepspeed --num_gpus=4 train.py --dataset_path data/processed --output_dir models/humantouch --mode smaller_gpu
```

### Training Issues

**Single-GPU - Training fails or is very slow:**
```bash
# Check system status
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"

# Use Basic mode for quick testing
python train.py --dataset_path data/processed --output_dir models/test --mode basic --test_model

# Check DeepSpeed installation
python -c "import deepspeed; print('DeepSpeed OK')"
```

**Multi-GPU - Training fails or communication issues:**
```bash
# Check DeepSpeed installation with multi-GPU support
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"

# Test basic multi-GPU functionality
deepspeed --num_gpus=2 train.py --dataset_path data/processed --output_dir models/test --mode basic

# Check GPU topology and communication
nvidia-smi topo -m

# Verify all GPUs are visible
python -c "import torch; print(f'Total GPUs: {torch.cuda.device_count()}')"

# Run with debugging enabled
NCCL_DEBUG=INFO deepspeed --num_gpus=2 train.py --dataset_path data/processed --output_dir models/test --mode basic
```

### Data Processing Issues

**Problem:** Dataset processing fails
```bash
# Check input file
head -5 data/raw/AI_Human_Text.csv

# Process with debugging
python data_processing.py --input data/raw/AI_Human_Text.csv --output data/processed --create_zip

# Verify output
ls -la data/processed/
head -2 data/processed/train.json
```

### Quality Optimization

**Problem:** Generated text quality needs improvement
```bash
# Use Full mode for best quality
python train.py --dataset_path data/processed --output_dir models/humantouch --mode full

# Increase training with custom parameters
python train.py --dataset_path data/processed --output_dir models/humantouch --mode full --epochs 8 --rank 256

# Test different models
python train.py --dataset_path data/processed --output_dir models/humantouch --mode full --model_name "Qwen/Qwen2.5-1.5B"
```

### Installation Issues

**Problem:** Package installation fails
```bash
# Update UV
uv self update

# Clear cache
uv cache clean

# Try pip instead
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.10+
```

### Model Loading Issues

**Problem:** Cannot load trained model
```bash
# Verify model files exist
# Windows: dir models\humantouch
# Linux/Mac: ls -la models/humantouch/

# Check PEFT installation
python -c "from peft import PeftModel; print('PEFT OK')"

# Test base model loading
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base')"
```

## 📊 Expected Results

### Training Metrics
- **Final Loss**: < 1.5
- **Eval Loss**: < 2.0
- **Training Time**: 48-120 hours
- **Model Size**: ~2-5GB

### Quality Metrics
- **ROUGE-L F1**: 0.55-0.65
- **BLEU Score**: 0.35-0.45  
- **Human Rating**: 8/10+
- **Perplexity**: 2.0-3.0

### Example Results
```
Input: "The artificial intelligence system processed the data efficiently and generated comprehensive analytical reports."

Output: "Our AI tool crunched through the data pretty efficiently and came up with some detailed analysis reports."
```

## 🎯 Use Cases

- **Content Creation**: Humanize AI-generated articles, blogs, marketing copy
- **Academic Writing**: Improve AI-assisted research papers and reports
- **Documentation**: Make technical documentation more readable
- **Creative Writing**: Add human touch to AI-generated stories
- **Business**: Natural-sounding AI communication and reports

## 🔗 Advanced Features

### 📁 Google Colab Support

For Google Colab users, we provide specialized scripts:

```bash
# Use the Colab-ready scripts
colab_ready/
├── data_processing_colab.py  # Auto-installs packages, Drive integration
└── train_colab.py           # Memory-optimized for Colab
```

**Colab Features:**
- 📦 **Auto-installation** of all dependencies
- 💾 **Google Drive integration** for dataset storage
- 🧠 **Memory optimization** for free/pro Colab
- 📊 **Progress tracking** with download packages
- ⚙️ **Same 3 training modes** as local version

### 📈 Dataset Requirements

**Input CSV Format:**
- `text`: The text content
- `generated`: 1 for AI text, 0 for human text
- `prompt`: (optional) prompt used to generate text

**Output Structure:**
```
data/processed/
├── train.json              # Training data (80%)
├── validation.json         # Validation data (10%)
├── test.json               # Test data (10%)
├── dataset_stats.json      # Dataset statistics
├── hf_dataset/             # HuggingFace format
└── humantouch_processed_data.zip  # Complete package
```

### 📉 WandB Integration

```bash
# Setup WandB (optional)
uv pip install wandb
wandb login

# Training automatically logs to WandB
python train.py --dataset_path data/processed --output_dir models/humantouch --mode full

# Disable WandB if needed
python train.py --dataset_path data/processed --output_dir models/humantouch --mode full --no_wandb

# Custom run names
python train.py --dataset_path data/processed --output_dir models/humantouch --mode full --run_name "production-v1"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test them
4. Format code: `black . && isort .` (optional)
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team** - Qwen3 base model
- **NVIDIA Research** - DoRA methodology  
- **Hugging Face** - PEFT and Transformers
- **Microsoft** - DeepSpeed optimization
- **Kaggle** - AI vs Human text dataset

## 📞 Support

- **Issues**: GitHub Issues for bug reports
- **Questions**: GitHub Discussions for help
- **Documentation**: This README + code comments

---

## 🚀 Quick Start Summary

```bash
# 1. Setup environment
git clone https://github.com/your-username/HumanTouch.git
cd HumanTouch
uv venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt

# 2. Process your dataset
python data_processing.py --input data/raw/AI_Human_Text.csv --output data/processed

# 3. Train with interactive mode selection
python train.py --dataset_path data/processed --output_dir models/humantouch

# 4. Test your model
python train.py --dataset_path data/processed --output_dir models/humantouch --mode basic --test_model
```

**🎉 Ready to start?** The enhanced interactive system will guide you through the entire process!