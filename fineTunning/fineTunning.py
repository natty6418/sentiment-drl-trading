#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datasets
from datasets import load_dataset


# In[2]:


from ipywidgets import FloatProgress
import json
from tqdm import tqdm
from typing import List, Dict, Optional
import torch
from loguru import logger
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    AutoConfig
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    #prepare_model_for_int8_training,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


# In[3]:


# !jupyter nbextension enable --py widgetsnbextension
# !jupyter labextension install @jupyter-widgets/jupyterlab-manager


# In[4]:


import os
import shutil

jsonl_path = "./data/dataset_new.jsonl"
save_path = './data/dataset_new'


if os.path.exists(jsonl_path):
    os.remove(jsonl_path)

if os.path.exists(save_path):
    shutil.rmtree(save_path)

directory = "./data"
if not os.path.exists(directory):
    os.makedirs(directory)


# In[5]:


dic = {
    0:"negative",
    1:'positive',
    2:'neutral',
}

tfns = load_dataset('zeroshot/twitter-financial-news-sentiment')
tfns = tfns['train']
tfns = tfns.to_pandas()
tfns['label'] = tfns['label'].apply(lambda x:dic[x])
tfns['instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
tfns.columns = ['input', 'output', 'instruction']
tfns = datasets.Dataset.from_pandas(tfns)
tfns


# In[6]:


tmp_dataset = datasets.concatenate_datasets([tfns]*2)
train_dataset = tmp_dataset
print(tmp_dataset.num_rows)

all_dataset = train_dataset.shuffle(seed = 42)
all_dataset.shape


# In[7]:


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


data_list = []
for item in all_dataset.to_pandas().itertuples():
    tmp = {}
    tmp["instruction"] = item.instruction
    tmp["input"] = item.input
    tmp["output"] = item.output
    data_list.append(tmp)


# save to a jsonl file
with open("./data/dataset_new.jsonl", 'w') as f:
    for example in tqdm(data_list, desc="formatting.."):
        f.write(json.dumps(format_example(example)) + '\n')


# In[8]:


model_name = "THUDM/chatglm3-6b"
jsonl_path = "./data/dataset_new.jsonl"  # updated path
save_path = './data/dataset_new'  # updated path
max_seq_length = 512
skip_overlength = True


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

# The read_jsonl function reads each line from the JSONL file, preprocesses it using the preprocess function,
# and then yields each preprocessed example.
def read_jsonl(path, max_seq_length, skip_overlength=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


# In[9]:


save_path = './data/dataset_new'

dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(jsonl_path, max_seq_length, skip_overlength)
    )
dataset.save_to_disk(save_path)


# In[19]:


training_args = TrainingArguments(
    output_dir='./finetuned_model',
    logging_steps=500,
    num_train_epochs=2,
   # max_steps=2000,  # Limit training length
    per_device_train_batch_size=4,  # Reduce batch size
    gradient_accumulation_steps=8,  # Increase accumulation to compensate
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    save_steps=500,
    fp16=True,
    torch_compile=False,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    remove_unused_columns=False,
    #gradient_checkpointing=True,  # Enable memory-efficient training
    #optim="adamw_bnb_8bit",  # Use memory-efficient optimizer
)


# In[20]:


q_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16
                                )


# In[17]:


#torch.cuda.empty_cache()


# In[21]:


model_name = "THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
        model_name,
        quantization_config=q_config,
        trust_remote_code=True,
   #     load_in_4bit=True,
  #      device_map="auto"
        )
#device = torch.device("cuda")
#if not getattr(model, "is_quantized", False):
 #   model.to(device)


#if not hasattr(model, "hf_device_map"):
 #   model.to(device)  # Only move if not loaded with accelerate
#else:
 #   print("Model is already assigned to devices by `accelerate`.")
#model.to(device)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


# In[ ]:


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



# LoRA
target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules,
    bias='none',
)
model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

resume_from_checkpoint = None
if resume_from_checkpoint is not None:
    checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, 'adapter_model.bin'
        )
        resume_from_checkpoint = False
    if os.path.exists(checkpoint_name):
        logger.info(f'Restarting from {checkpoint_name}')
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        logger.info(f'Checkpoint {checkpoint_name} not found')

model.print_trainable_parameters()


# In[ ]:


dataset = datasets.load_from_disk("./data/dataset_new")
dataset = dataset.train_test_split(0.2, shuffle=True, seed = 42)


# In[ ]:


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def prediction_step(self, model: torch.nn.Module, inputs, prediction_loss_only: bool, ignore_keys = None):
        with torch.no_grad():
            res = model(
                input_ids=inputs["input_ids"].to(model.device),
                labels=inputs["labels"].to(model.device),
            ).loss
        return (res, None, None)

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [tokenizer.pad_token_id] * (seq_len - 1) + ids[(seq_len - 1) :] + [tokenizer.pad_token_id] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

# Train
# Took about 10 compute units
# Took 40 mins to train
writer = SummaryWriter()
trainer = ModifiedTrainer(
    model=model,
    args=training_args,             # Trainer args
    train_dataset=dataset["train"], # Training set
    eval_dataset=dataset["test"],   # Testing set
    data_collator=data_collator,    # Data Collator
    callbacks=[TensorBoardCallback(writer)],
)
trainer.train()
writer.close()
# save model
model.save_pretrained(training_args.output_dir)

