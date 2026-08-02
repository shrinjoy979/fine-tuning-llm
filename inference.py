from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(MODEL)

prompt = """
### Instruction:
Translate English to French

### Input:
Good Night

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=30
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))