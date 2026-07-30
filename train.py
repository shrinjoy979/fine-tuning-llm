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

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model("fine_tuned_model")