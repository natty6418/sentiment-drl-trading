#!/usr/bin/env python3
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


# Config

MODEL_NAME = "meta-llama/Llama-2-7b-hf"  
MAX_SEQ_LEN = 512
SEED = 42
OUTPUT_DIR = "./finetuned_model"

LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}
INSTRUCTION = "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."



def make_prompt(example: dict) -> str:
    return (
        f"Instruction: {INSTRUCTION}\n"
        f"Input: {example['text']}\n"
        "Answer: "
    )

def load_tfns():
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    ds = ds.map(lambda x: {"label_text": LABEL_MAP[int(x["label"])]})
    return ds

ds = load_tfns().shuffle(seed=SEED)

splits = ds.train_test_split(test_size=0.2, seed=SEED)



# Tokenizer / Config

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preprocess: build input_ids + masked labels

def preprocess(example: dict):
    prompt = make_prompt(example)
    target = example["label_text"]

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LEN)
    target_ids = tokenizer.encode(target, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)

    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    input_ids = input_ids[:MAX_SEQ_LEN]

    # Mask prompt tokens so loss is only on the answer
    # HF expects -100 for ignored label positions
    prompt_len = min(len(prompt_ids), MAX_SEQ_LEN)
    labels = [-100] * (prompt_len - 1) + input_ids[(prompt_len - 1):]
    labels = labels[:MAX_SEQ_LEN]

    return {"input_ids": input_ids, "labels": labels}

train_ds = splits["train"].map(preprocess, remove_columns=splits["train"].column_names)
eval_ds = splits["test"].map(preprocess, remove_columns=splits["test"].column_names)


# Data collator (pad batch)

def collate(features):
    max_len = max(len(f["input_ids"]) for f in features)

    input_ids = []
    labels = []
    for f in features:
        ids = f["input_ids"]
        lab = f["labels"]

        pad_len = max_len - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        labels.append(lab + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# Model: 4-bit QLoRA + LoRA

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=bnb_config,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# -------------------------
# Training
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=200,                 
    logging_steps=50,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    fp16=True,
    remove_unused_columns=False,
    report_to="tensorboard",         
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate,
)

trainer.train()

# Save adapter weights
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
