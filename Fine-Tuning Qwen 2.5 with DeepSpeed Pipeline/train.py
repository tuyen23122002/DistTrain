from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import os

# --- Step 1: Load tokenizer & EOS token ---
# Sửa nhỏ: Tên model chính thức là Qwen2
model_name = "Qwen/Qwen2.5-3B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
EOS_TOKEN = tokenizer.eos_token

# --- Step 2: Prompt formatting function (Giữ nguyên) ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}""" + EOS_TOKEN

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_, output)
        texts.append(text)
    return {"text": texts}

# --- Step 3: Load dataset & preprocess (Giữ nguyên) ---
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

def tokenize(example):
    # Sửa nhỏ: Thêm return_overflowing_tokens=True để xử lý các mẫu dài
    return tokenizer(example["text"], truncation=True, max_length=512)
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)


# --- Step 6: Sửa lại TrainingArguments cho Full Fine-Tuning ---
training_args = TrainingArguments(
    output_dir="/home/tuyennt/working/Multi-GPU-Fine-Training-LLMs/Fine-Tuning Qwen 2.5 with DeepSpeed Pipeline/qwen2-7b-pipeline-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    warmup_steps=100,
    max_grad_norm=1.0,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    deepspeed="/home/tuyennt/working/Multi-GPU-Fine-Training-LLMs/Fine-Tuning Qwen 2.5 with DeepSpeed Pipeline/ds_config.json",
    report_to="wandb",                         # << Bật wandb
    run_name="qwen2.5-finetune-pipeline",      # << Tên run
    gradient_checkpointing=True,
)

# --- Step 7: Trainer (Giữ nguyên) ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Bắt đầu huấn luyện Full-Tuning với DeepSpeed Pipeline...")
trainer.train()