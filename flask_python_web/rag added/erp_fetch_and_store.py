# ✅ erp_fetch_and_store.py — Sadece veri çekip veritabanına ekler

import psycopg2
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔹 Embedding modeli (384 boyutlu)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Veritabanı bağlantı ayarları
DB_CONFIG = {
    "dbname": "ragdb",
    "user": "postgres",
    "password": "secret",  # kendi şifreni yaz
    "host": "localhost",
    "port": "5432"
}

def fetch_live_erp_data(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = "\n".join(p.text for p in soup.find_all("p"))
        return text
    except Exception as e:
        print(f"⚠️ Veri çekme hatası: {e}")
        return ""

def insert_to_pgvector(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    for chunk, emb in zip(chunks, embeddings):
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (chunk, emb.tolist())
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ {len(chunks)} yeni içerik eklendi.")

if __name__ == "__main__":
    url = "https://www.linkbilgisayar.com.tr/link-cloud"
    content = fetch_live_erp_data(url)
    if content:
        insert_to_pgvector(content)
    else:
        print("❌ Veri alınamadı.")
