import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset

def load_jsonl_dataset(path):
    # .jsonl içeriğini load_dataset ile yükle
    dataset = load_dataset('json', data_files=path, split='train')

    # instruction + input + output'u birleştir
    def format_example(example):
        prompt = f"Soru: {example['instruction']}\n"
        if example.get("input"):
            prompt += f"Girdi: {example['input']}\n"
        prompt += f"Cevap: {example['output']}"
        return {"text": prompt}

    return dataset.map(format_example)

def fine_tune_model(dataset_path, model_dir, epochs, batch_size):
    # Model ve tokenizer yükle
    print("🚀 Model yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

    # PEFT LoRA yapılandırması
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Dataset yükle ve tokenize et
    print("📚 Veri seti yükleniyor...")
    dataset = load_jsonl_dataset(dataset_path)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Eğitim argümanları
    output_dir = "fine_tune/output_model"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_total_limit=1,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Eğitim başlat
    print("🎯 Fine-tuning başlıyor...")
    trainer.train()

    # Model ve tokenizer kaydet
    print(f"💾 Model kaydediliyor: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Fine-tuning tamamlandı!")

if  __name__ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    fine_tune_model(args.dataset, args.model_dir, args.epochs, args.batch_size)