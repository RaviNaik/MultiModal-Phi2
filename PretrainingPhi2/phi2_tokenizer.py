import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, AutoTokenizer, AutoConfig

# Fix the context length and create an instance of Phi2 Tokenizer
context_length = 256
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.pad_token = tokenizer.eos_token

# The mapper function to create Iterable tokenized sample from Interleved dataset
def tokenize_mapper(element):
    outputs = tokenizer(
        element["text"],
        max_length=context_length,
        truncation=True,
        return_overflowing_tokens=True,
        return_length=True,
    )
    return {"input_ids": outputs["input_ids"]}