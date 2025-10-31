# GPT-2 Medium Fine-tuning with LoRA

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)

A production-ready implementation for fine-tuning GPT-2 Medium (354M parameters) using **LoRA (Low-Rank Adaptation)** on WikiText datasets. Supports multi-GPU training with distributed data parallelism, achieving significant parameter efficiency (1.74% trainable parameters) while maintaining strong performance.

## 📊 Results

| Configuration | Dataset | Final PPL | Training Time | Trainable Params |
|--------------|---------|-----------|---------------|------------------|
| LoRA r=16 | WikiText-2 | **20.72** | 1.82h (2x T4) | 6.3M (1.74%) |
| LoRA r=16 | WikiText-103 | ~18 | ~6-8h (2x T4) | 6.3M (1.74%) |

**Key Achievements:**
- ✅ 98.26% parameter reduction using LoRA
- ✅ Multi-GPU training with optimized NCCL configuration
- ✅ Early stopping at epoch 2.69 with best validation perplexity
- ✅ Production-ready checkpointing and model export

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA-capable GPU(s) recommended
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gpt2-medium-lora-training.git
cd gpt2-medium-lora-training

# Install dependencies
pip install -r requirements.txt
```

### Basic Training (Single GPU)

```bash
# Fast experimentation with WikiText-2
python scripts/train.py \
    --use_wikitext_2 \
    --lora_r 16 \
    --learning_rate 3e-4 \
    --num_epochs 5 \
    --batch_size 16 \
    --output_dir ./outputs/wikitext2_run
```

### Multi-GPU Training (Recommended)

```bash
# Configure accelerate for your setup
accelerate config

# Launch distributed training
accelerate launch --config_file configs/accelerate_multi_gpu.yaml scripts/train.py \
    --use_wikitext_2 \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 3e-4 \
    --num_epochs 5 \
    --batch_size 16 \
    --grad_accum 4 \
    --save_steps 250 \
    --eval_steps 250
```

---

## 📁 Project Structure

```
gpt2-medium-lora-training/
│
├── configs/                          # Training configurations
│   ├── accelerate_multi_gpu.yaml    # Multi-GPU distributed setup
│   ├── default_config.yaml          # Default hyperparameters
│   ├── wikitext2_config.yaml        # WikiText-2 specific config
│   └── wikitext103_config.yaml      # WikiText-103 specific config
│
├── scripts/                         # Training and evaluation scripts
│   ├── train.py                     # Main training script
│   ├── evaluate.py                  # Model evaluation
│   ├── generate.py                  # Text generation
│   └── push_to_hub.py              # HuggingFace Hub upload
│
├── src/                            # Source code
│   ├── data/                       # Data loading utilities
│   ├── models/                     # Model architecture
│   ├── training/                   # Training utilities
│   └── utils/                      # Helper functions
│       ├── __init__.py
│       └── config.py
│
├── outputs/                        # Training outputs (gitignored)
│   ├── checkpoints/               # Model checkpoints
│   ├── final_model/               # Final trained model
│   └── logs/                      # Training logs
│
├── tests/                         # Unit tests
│
├── KAGGLE NOTEBOOK/              # Kaggle-specific files
│   └── peft-lora-gpt-medium.ipynb
│
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔧 Configuration

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_wikitext_2` | False | Use WikiText-2 (faster) instead of WikiText-103 |
| `--max_length` | 512 | Maximum sequence length |
| `--lora_r` | 16 | LoRA rank (8, 16, 32, 64) |
| `--lora_alpha` | 32 | LoRA alpha (typically 2x rank) |
| `--lora_dropout` | 0.05 | LoRA dropout rate |
| `--learning_rate` | 3e-4 | Peak learning rate |
| `--num_epochs` | 5 | Number of training epochs |
| `--batch_size` | 8 | Per-device batch size |
| `--grad_accum` | 8 | Gradient accumulation steps |
| `--weight_decay` | 0.01 | AdamW weight decay |
| `--warmup_ratio` | 0.05 | Learning rate warmup ratio |
| `--scheduler` | cosine | LR scheduler (linear/cosine/constant) |
| `--patience` | 3 | Early stopping patience |
| `--save_steps` | 250 | Checkpoint save frequency |
| `--eval_steps` | 250 | Evaluation frequency |
| `--output_dir` | auto | Output directory |
| `--resume` | None | Resume from checkpoint path |

### Hyperparameter Recommendations

**For WikiText-2 (Experimentation):**
```bash
--lora_r 16 --learning_rate 3e-4 --num_epochs 5
```

**For WikiText-103 (Production):**
```bash
--lora_r 32 --learning_rate 2e-4 --num_epochs 3 --batch_size 8
```

**Low VRAM (<16GB):**
```bash
--batch_size 4 --grad_accum 16 --max_length 256
```

---

## 🎯 LoRA Configuration Details

### Architecture
- **Target Modules**: `c_attn`, `c_proj`, `c_fc` (attention and feed-forward layers)
- **LoRA Rank (r)**: 16 (1.74% trainable parameters)
- **LoRA Alpha**: 32 (scaling factor)
- **Dropout**: 0.05
- **RSLoRA**: Enabled for improved stability

### Why LoRA?
- **Efficiency**: Only 6.3M trainable parameters vs 354M total
- **Speed**: Faster training and lower memory footprint
- **Modularity**: Easy to swap adapters for different tasks
- **Quality**: Achieves comparable performance to full fine-tuning

---

## 💻 Multi-GPU Setup

### NCCL Configuration (Critical for Cloud/Kaggle)

The training script includes optimized NCCL settings to prevent timeout issues:

```python
os.environ['NCCL_TIMEOUT'] = '3600'           # 1 hour timeout
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1' # Better error handling
os.environ['NCCL_DEBUG'] = 'WARN'             # Reduced logging
os.environ['NCCL_IB_DISABLE'] = '1'           # Disable InfiniBand
os.environ['NCCL_P2P_DISABLE'] = '1'          # Disable P2P transfers
```

### Accelerate Configuration

Create `configs/accelerate_multi_gpu.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2  # Number of GPUs
rdzv_backend: static
same_network: true
use_cpu: false
```

### Effective Batch Size Calculation

```
Effective Batch Size = per_device_batch × grad_accum × num_gpus
Example: 16 × 4 × 2 = 128
```

---

## 📈 Training Process

### Phase 1: Setup & Initialization
1. **Environment**: CUDA detection, GPU enumeration
2. **Authentication**: HuggingFace token, W&B API key
3. **Model Loading**: GPT-2 Medium (354M params) in FP16
4. **LoRA Setup**: Inject adapters (6.3M trainable params)
5. **Dataset**: Load & tokenize WikiText

### Phase 2: Training Loop
- **Optimizer**: AdamW with fused kernels
- **Scheduler**: Cosine annealing with warmup
- **Mixed Precision**: FP16 training
- **Gradient Checkpointing**: Memory optimization
- **Early Stopping**: Monitor validation loss

### Phase 3: Evaluation & Export
- **Test Set Evaluation**: Final perplexity calculation
- **Model Export**: Save LoRA adapters
- **Checkpointing**: Best model preservation
- **Logging**: W&B integration for metrics

---

## 🧪 Evaluation & Generation

### Evaluate Model

```bash
python scripts/evaluate.py \
    --model_path ./outputs/gpt2-finetuned-20251028/final_model \
    --dataset wikitext-2-raw-v1
```

### Generate Text

```bash
python scripts/generate.py \
    --model_path ./outputs/gpt2-finetuned-20251028/final_model \
    --prompt "The history of artificial intelligence began" \
    --max_length 100 \
    --temperature 0.8 \
    --top_p 0.9
```

### Load Model Programmatically

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "gpt2-medium",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "./outputs/gpt2-finetuned-20251028/final_model"
)

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

# Generate
prompt = "In the field of machine learning"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🌐 HuggingFace Hub Integration

### Push Model to Hub

```bash
python scripts/push_to_hub.py \
    --model_path ./outputs/gpt2-finetuned-20251028/final_model \
    --repo_id yourusername/gpt2-medium-wikitext2-lora \
    --private False
```

### Load from Hub

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = PeftModel.from_pretrained(
    "gpt2-medium",
    "yourusername/gpt2-medium-wikitext2-lora"
)
```

---

## 🐛 Troubleshooting

### NCCL Timeout Errors
**Issue**: `WorkNCCL timeout after 600000 milliseconds`

**Solution**: Already handled in `train.py` with extended timeouts. If persistent:
```bash
export NCCL_TIMEOUT=7200  # 2 hours
export NCCL_P2P_DISABLE=1
```

### Out of Memory (OOM)
**Solutions**:
- Reduce `--batch_size` (try 4 or 2)
- Increase `--grad_accum` to maintain effective batch size
- Reduce `--max_length` (try 256 or 384)
- Enable gradient checkpointing (already enabled)

### Slow Training
**Optimizations**:
- Use `adamw_torch_fused` optimizer (already enabled)
- Enable mixed precision FP16 (already enabled)
- Increase `--dataloader_num_workers` (default: 4)
- Use faster storage for datasets

### Model Not Learning
**Debugging**:
- Check learning rate (try 5e-4 for WikiText-2)
- Verify dataset loading (check `--use_wikitext_2` flag)
- Monitor gradient norms in logs
- Increase LoRA rank (`--lora_r 32`)

---

## 📊 W&B Integration

Training automatically logs to Weights & Biases when API key is provided:

```bash
# Set W&B API key
export WANDB_API_KEY=your_key_here

# Or use Kaggle secrets
# Add WANDB_API_KEY to Kaggle secrets
```

**Logged Metrics**:
- Training loss (per step)
- Evaluation loss & perplexity
- Learning rate schedule
- Gradient norms
- GPU memory usage

---

## 🧰 Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Code formatting
black src/ scripts/
isort src/ scripts/

# Linting
flake8 src/ scripts/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{gpt2-lora-finetuning,
  author = {Your Name},
  title = {GPT-2 Medium Fine-tuning with LoRA},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/gpt2-medium-lora-training}
}
```

### References
- **GPT-2**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **WikiText**: [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers and PEFT libraries
- **Microsoft** for the LoRA methodology
- **Salesforce** for the WikiText datasets
- **Kaggle** for providing free GPU resources
- **Weights & Biases** for experiment tracking

---

## 📧 Contact

**Project Maintainer**: Your Name  
**Email**: your.email@example.com  
**GitHub**: [@yourusername](https://github.com/yourusername)

---

**Built with ❤️ for the ML community**