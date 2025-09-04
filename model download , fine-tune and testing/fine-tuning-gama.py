from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
import json
import os

# Hugging Face Token
os.environ["HF_TOKEN"] = "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"

# Model ID
model_id = "google/gemma-2b-it"  # CPU dostu, 3B yerine 2B önerilir

# Tokenizer ve model yükleme (CPU için)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=os.environ["HF_TOKEN"])
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": "cpu"},
    trust_remote_code=True,
    token=os.environ["HF_TOKEN"]
)
tokenizer.pad_token = tokenizer.eos_token

# LoRA ayarı (quantization YOK)
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False  # Eğitim için
)

model = get_peft_model(model, lora_config)

# Veri yükle
def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            instruction = item['instruction']
            input_text = item.get('input', '')
            output = item['output']
            prompt = f"Soru: {instruction}\nGirdi: {input_text}\nYanıt: {output}"
            data.append({"text": prompt})
    return Dataset.from_list(data)

dataset = load_jsonl_dataset("erp_finetune_dataset.jsonl")

# Tokenize et
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./gemma2b-finetuned-cpu",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=False,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Trainer başlat
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Eğitimi başlat
trainer.train()

# Kaydet
model.save_pretrained("./gemma2b-finetuned-cpu")
tokenizer.save_pretrained("./gemma2b-finetuned-cpu")
