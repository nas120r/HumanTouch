# HumanTouch: AI Text Humanization with DoRA

Transform AI-generated text into natural, human-like writing using DoRA fine-tuned Qwen3-0.6B with 32k context support.

## 🎯 What is HumanTouch?

HumanTouch solves the problem of robotic, repetitive AI-generated content by converting it into natural, flowing human-like text while preserving the original meaning and context.

**Key Features:**
- 🧠 **DoRA Fine-tuning** - Advanced weight decomposition (rank 128) for superior quality
- 📝 **32k Context** - Handle long documents without truncation  
- 🎯 **High Quality** - No quantization or Flash Attention compromises
- ⚡ **Easy to Use** - Simple Python scripts for training and inference

**The Problem:** AI text often sounds robotic, repetitive, and lacks natural human flow.

**The Solution:** HumanTouch learns from 500K human vs AI text pairs to add natural variations, improve flow, and maintain context coherence.

## 📋 Requirements

### Hardware
- **GPU**: A100 80GB, H100, or RTX 4090 24GB+ 
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space

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

```bash
python data_processing.py --input data/raw/AI_Human_Text.csv --output data/processed
```

**Expected output:**
```
Loading dataset from data/raw/AI_Human_Text.csv
Dataset shape: (500000, 3)
Columns: ['text', 'generated', 'prompt']
Generated distribution: 0    250000, 1    250000
Filtering by length...
Original samples: 500000
After filtering: 450000
Creating training pairs...
AI texts: 225000
Human texts: 225000
Created 225000 training pairs
Splitting and saving dataset...
Train: 180000
Validation: 22500
Test: 22500
Dataset saved to data/processed
Files created:
- train.json, validation.json, test.json
- hf_dataset/ (Hugging Face format)

Data processing complete!
Processed 225000 samples
Max sequence length: 32768
```

### Step 7: Train Model

**Basic training command:**
```bash
python train.py --dataset_path data/processed/hf_dataset --output_dir models/humantouch
```

**Full training with all options:**
```bash
python train.py \
    --dataset_path data/processed/hf_dataset \
    --output_dir models/humantouch \
    --rank 128 \
    --alpha 256 \
    --max_length 32768 \
    --batch_size 1 \
    --grad_accum 16 \
    --learning_rate 8e-5 \
    --epochs 6 \
    --run_name "my-experiment"
```

**For smaller GPUs (RTX 4090, etc.):**
```bash
python train.py \
    --dataset_path data/processed/hf_dataset \
    --output_dir models/humantouch \
    --rank 64 \
    --alpha 128 \
    --max_length 16384 \
    --batch_size 1 \
    --grad_accum 32 \
    --learning_rate 8e-5 \
    --epochs 4
```

**Training progress:**
- Monitor with WandB (automatic if installed)
- Check GPU usage: `nvidia-smi`
- Expected time: 48-120 hours depending on GPU

### Step 8: Evaluate Model

```bash
# Comprehensive evaluation
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
├── data_processing.py      # Dataset preparation script
├── train.py               # DoRA training script  
├── evaluate.py            # Model evaluation script
├── inference.py           # Text humanization script
├── requirements.txt       # Python dependencies
├── configs/
│   └── dora_config.yaml   # DoRA configuration settings
├── data/
│   ├── raw/              # Original Kaggle dataset
│   ├── processed/        # Processed training data
│   └── evaluation/       # Evaluation results
├── models/               # Trained model checkpoints
└── logs/                 # Training logs
```

## ⚙️ Configuration Options

### Training Parameters

```bash
# Maximum Quality (slow, best results)
python train.py --rank 128 --alpha 256 --max_length 32768 --epochs 6

# Balanced (faster, good results)  
python train.py --rank 64 --alpha 128 --max_length 16384 --epochs 4

# Fast Prototype (quick testing)
python train.py --rank 32 --alpha 64 --max_length 8192 --epochs 2
```

### Memory Optimization

**For 24GB GPU (RTX 4090):**
```bash
python train.py \
    --rank 64 \
    --max_length 16384 \
    --batch_size 1 \
    --grad_accum 32
```

**For 16GB GPU (RTX 4080):**
```bash
python train.py \
    --rank 32 \
    --max_length 8192 \
    --batch_size 1 \
    --grad_accum 32
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

### Out of Memory Errors

**Problem:** CUDA out of memory during training
```bash
# Solution 1: Reduce sequence length
python train.py --max_length 16384

# Solution 2: Reduce batch size (already at minimum 1)
python train.py --grad_accum 32

# Solution 3: Reduce DoRA rank
python train.py --rank 64

# Solution 4: Use smaller model
# Edit train.py and change model_name to a smaller variant
```

### Slow Training

**Problem:** Training is very slow
```bash
# Check GPU utilization
nvidia-smi

# Verify CUDA is being used
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check if DeepSpeed is working - look for "DeepSpeed" in logs
```

### Poor Quality Results

**Problem:** Generated text quality is low
```bash
# Solution 1: Increase DoRA rank
python train.py --rank 128

# Solution 2: More training epochs
python train.py --epochs 8

# Solution 3: Lower learning rate
python train.py --learning_rate 5e-5

# Solution 4: Better data filtering
# Edit data_processing.py to add quality filters
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

## 🔧 Advanced Configuration

### Custom Dataset Format

Your CSV should have these columns:
- `text`: The text content
- `generated`: 1 for AI text, 0 for human text
- `prompt`: (optional) prompt used to generate text

### WandB Monitoring

```bash
# Install and login to WandB
uv pip install wandb
wandb login

# Training will automatically log to WandB
# View at: https://wandb.ai/your-username/humantouch-dora
```

### Multi-GPU Training

```bash
# For multiple GPUs
python -m torch.distributed.launch --nproc_per_node=2 train.py \
    --dataset_path data/processed/hf_dataset \
    --output_dir models/humantouch-multigpu
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

**Ready to start?** Follow the setup steps above and you'll be humanizing AI text in no time!