from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load your custom dataset
dataset = load_dataset("csv", data_files={"train": "essays.csv"})

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
