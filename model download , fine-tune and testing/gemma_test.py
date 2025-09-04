import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ✅ Cihaz kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {device}")

# ✅ Model ve tokenizer yolları
base_model_path = "./local_models/gemma-2b-it"
finetuned_model_path = "./gemma-2b-it-lora-finetuned"

# ✅ Tokenizer ve Base model yükle
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True).to(device)

# ✅ PEFT modelini yükle (doğru yöntem)
model = PeftModel.from_pretrained(base_model, finetuned_model_path)
model = model.merge_and_unload()  # LoRA katmanlarını entegre et (inference için optimize)
model.eval()

# ✅ Prompt
instruction = "Fatura ile irsaliyenin ne farkı var?"
input_text = ""
if input_text:
    prompt = f"Soru: {instruction}\nGirdi: {input_text}\nYanıt:"
else:
    prompt = f"Soru: {instruction}\nYanıt:"

# ✅ Tokenize et
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# ✅ Yanıt üret
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# ✅ Çıktıyı yazdır
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n📢 Model Yanıtı:")
print(generated_text)
