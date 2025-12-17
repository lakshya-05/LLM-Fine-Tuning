# LLM Fine-Tuning

[![GitHub stars](https://img.shields.io/github/stars/lakshya-05/LLM-Fine-Tuning?style=social)](https://github.com/lakshya-05/LLM-Fine-Tuning)
[![GitHub forks](https://img.shields.io/github/forks/lakshya-05/LLM-Fine-Tuning?style=social)](https://github.com/lakshya-05/LLM-Fine-Tuning)

A comprehensive repository for fine-tuning Large Language Models (LLMs) using efficient techniques like LoRA (Low-Rank Adaptation), QLoRA, and full fine-tuning. This project provides scripts, notebooks, and configurations to fine-tune models on custom datasets for tasks such as instruction tuning, question answering, summarization, and more.

Built with **Hugging Face Transformers**, **PEFT**, **Datasets**, and **Accelerate** for scalable, GPU-optimized training.

## 🚀 Features

- **Efficient Fine-Tuning**: Support for LoRA, QLoRA, and full parameter fine-tuning to reduce memory usage.
- **Multi-GPU Training**: Distributed training with DeepSpeed and FSDP integration.
- **Custom Dataset Support**: Easy integration with Hugging Face Datasets or local JSON/CSV files.
- **Interactive Notebooks**: Jupyter notebooks for experimentation and quick starts.
- **Evaluation Metrics**: Built-in ROUGE, BLEU, perplexity, and custom metrics.
- **Model Checkpointing**: Automatic saving and loading of fine-tuned models.
- **Inference Scripts**: Deploy fine-tuned models for generation and evaluation.

## 📋 Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (recommended for training)
- At least 16GB VRAM for base models (e.g., Llama-2-7B)


## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lakshya-05/LLM-Fine-Tuning.git
   cd LLM-Fine-Tuning
   ```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies include:
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
datasets>=2.14.0
accelerate>=0.24.0
trl>=0.7.0
bitsandbytes>=0.41.0  # For QLoRA
deepspeed  # Optional for advanced training

## 📖 Usage
1. Prepare Your Dataset
* Use Hugging Face Datasets: e.g., load_dataset("databricks/databricks-dolly-15k")

2. Fine-Tune with LoRA
Run the training script:
```bash
python src/train.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset_name your_dataset \
  --output_dir ./checkpoints \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_steps 1000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4
```

3. Using Jupyter Notebooks

* Open notebooks/quickstart_lora.ipynb for a step-by-step LoRA fine-tuning example.
* notebooks/qlora_4bit.ipynb for memory-efficient 4-bit quantization.

## 📊 Results & Benchmarks

* Fine-tuned Llama-2-7B on Dolly-15k: Achieves ~75% on MT-Bench (vs. 65% base).
* Memory: LoRA uses ~10GB VRAM for 7B model; QLoRA ~6GB.
See results/ for logs and evaluations.
