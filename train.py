from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# Small model for learning
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load dataset
dataset = load_dataset(
    "json",
    data_files="data/dataset.json"
)
