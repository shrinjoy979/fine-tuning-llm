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

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset(
    "json",
    data_files="data/dataset.json"
)

# Convert dataset into prompt format
def formatting(example):
    return {
        "text": f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    }

dataset = dataset.map(formatting)
