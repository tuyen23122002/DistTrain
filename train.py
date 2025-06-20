from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

# --- Step 1: Load tokenizer & EOS token ---
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Sửa lỗi nhỏ: Thêm pad_token nếu nó chưa tồn tại để tránh lỗi padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN = tokenizer.eos_token # Giữ nguyên

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
# Không cần set_format, Trainer sẽ tự xử lý

# --- Step 4: Load Qwen model ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    #device_map="auto",  # Rất tốt, đã comment dòng này
)

# =========================================================================
# === SỬA LỖI QUAN TRỌNG 1: BẬT GRADIENT CHECKPOINTING TRÊN MODEL TRỰC TIẾP ===
# =========================================================================
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# --- Step 5: Apply LoRA ---
# =========================================================================
# === SỬA LỖI QUAN TRỌNG 2: SỬA LẠI TARGET MODULES CHO ĐÚNG VỚI QWEN2 ===
# =========================================================================
peft_config = LoraConfig(
    r=16,  # Tăng r lên một chút để có kết quả tốt hơn
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# =========================================================================
# === SỬA LỖI QUAN TRỌNG 3: BẬT INPUT GRADIENTS CHO PEFT MODEL ===
# =========================================================================
model.enable_input_require_grads()

# In ra để kiểm tra
model.print_trainable_parameters()

# --- Step 6: TrainingArguments (Giữ nguyên, vì bạn đã thêm gradient_checkpointing=True) ---
training_args = TrainingArguments(
    output_dir="./qwen2.5b-lora-alpaca",
    num_train_epochs=1,
    per_device_train_batch_size=4, # batch size này khá lớn, nếu OOM thì giảm xuống
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=False,
    fp16=True,
    deepspeed= r"/home/tuyennt/working/Multi-GPU-Fine-Training-LLMs/ds_config_zero3.json", 
    report_to="none",
    gradient_checkpointing=True, # Dòng này vẫn rất quan trọng, phải giữ lại
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

trainer.train()