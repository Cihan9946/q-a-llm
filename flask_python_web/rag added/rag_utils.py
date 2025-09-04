import psycopg2
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# 🔹 Embedding modeli (384 boyutlu)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Veritabanı bağlantı ayarları
DB_CONFIG = {
    "dbname": "ragdb",
    "user": "postgres",
    "password": "secret",  # kendi şifreni kullan
    "host": "localhost",
    "port": "5432"
}

# 🔹 Ollama ayarları
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "turkcell-custom"  # model ismini değiştir

def get_similar_chunks(question, top_k=3):
    try:
        # Embedding oluştur
        embedding = model.encode(question, convert_to_tensor=True).cpu().numpy().tolist()
        
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Geliştirilmiş sorgu
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) as similarity 
            FROM documents
            WHERE 1 - (embedding <=> %s::vector) > 0.3
            ORDER BY similarity DESC
            LIMIT %s;
        """, (embedding, embedding, top_k))

        results = [row[0] for row in cur.fetchall()]
        print(f"🔍 Bulunan {len(results)} benzer doküman")
        return results
        
    except Exception as e:
        print(f"❌ Veritabanı Hatası: {str(e)}")
        return []

def ask_ollama(payload_or_prompt):
    try:
        if isinstance(payload_or_prompt, dict):
            payload = payload_or_prompt
        else:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": payload_or_prompt,
                "stream": False
            }

        print("📤 [ask_ollama] Prompt gönderiliyor...")
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            result = response.json().get("response", "")
            print("✅ [ask_ollama] Yanıt alındı:", result[:100])
            return result
        else:
            print("❌ [ask_ollama] HTTP Hatası:", response.status_code)
            return f"⚠️ Ollama Hatası: {response.status_code}"
    except Exception as e:
        print("❌ [ask_ollama] HATA:", str(e))
        return f"⚠️ Ollama Hatası: {str(e)}"

def rag_ask(question):
    """Tüm RAG zincirini çalıştır."""
    print("\n🔍 Soru alındı:", question)
    chunks = get_similar_chunks(question)
    if not chunks:
        print("⚠️ Hiç benzer içerik bulunamadı.")
        return
    prompt = build_prompt(question, chunks)
    print("\n📋 Oluşturulan prompt:\n", prompt[:600], "...\n")
    answer = ask_ollama(prompt)
    print("🤖 Cevap:\n", answer)


# rag_utils.py'ye bu fonksiyonu ekleyin
def build_prompt(question, context_chunks):
    """RAG için prompt oluşturur"""
    context = "\n".join([f"- {chunk}" for chunk in context_chunks])
    
    return f"""<|system|>
Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtlayın. 
Eğer cevap verilen bilgilerde yoksa "Bu konuda yeterli bilgi bulunmamaktadır." yazın.

Bağlam Bilgileri:
{context}

Soru: {question}
Cevap:"""



# 🔹 Terminalden çalıştırmak için
if __name__ == "_main_":
    while True:
        question = input("\n👤 Soru (çıkmak için 'q'): ")
        if question.lower() == "q":
            break
        rag_ask(question)