from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# --- Step 1: Load tokenizer & EOS token ---
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
EOS_TOKEN = tokenizer.eos_token

# --- Step 2: Prompt formatting function ---
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

# --- Step 3: Load dataset & preprocess ---
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# --- Step 4: Load Qwen model ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    # device_map="auto" không được dùng khi có DeepSpeed
)

# Bật gradient checkpointing để tiết kiệm bộ nhớ (Rất quan trọng cho full fine-tuning)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# --- Step 5: BỎ QUA PHẦN ÁP DỤNG LORA ---
# Không cần peft_config, get_peft_model, hay enable_input_require_grads nữa.
# Toàn bộ mô hình sẽ được huấn luyện.

# --- Step 6: TrainingArguments (Đã điều chỉnh cho full fine-tuning) ---
training_args = TrainingArguments(
    # Đổi tên thư mục output để phản ánh đúng phương pháp
    output_dir="/home/tuyennt/working/qwen2.5b-full-finetune-alpaca",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    # Giảm learning rate cho full fine-tuning để huấn luyện ổn định hơn
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    bf16=False,
    fp16=True,
    # DeepSpeed là chìa khóa để full fine-tune thành công trên các GPU thông thường
    deepspeed=r"/home/tuyennt/working/Multi-GPU-Fine-Training-LLMs/Fine-Tuning Qwen 2.5 with DeepSpeed_Zero3/ds_config_zero3.json", 
    report_to="wandb",                         # << Bật wandb
    run_name="qwen2.5-finetune-pipeline_zero3",  
    # Gradient checkpointing vẫn rất quan trọng, phải giữ lại
    gradient_checkpointing=True,
)

# --- Step 7: Trainer ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Bắt đầu quá trình full fine-tuning
trainer.train()