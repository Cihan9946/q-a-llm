import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔹 1. PDF'den metni al
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# 🔹 2. Metni chunk'lara böl
def split_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# 🔹 3. Başla
pdf_path = "ersin_tevatiroglu_tez.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = split_text(text)

print(f"📄 PDF'den {len(chunks)} parça (chunk) üretildi.")

# 🔹 4. Embedding modeli (MiniLM - 384 boyutlu)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# 🔹 5. PostgreSQL veritabanına bağlan
conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="secret",   # Şifren neyse onu yaz
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# 🔹 6. Her chunk'ı embedding ile birlikte veritabanına yaz
for chunk, vector in zip(chunks, embeddings):
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (chunk, vector.tolist())
    )

conn.commit()
cur.close()
conn.close()

print("✅ PDF başarıyla vektörleştirildi ve veritabanına kaydedildi.")

import csv

with open("chunk_embeddings.csv", mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Chunk", "Embedding (first 10 dims)"])
    for chunk, vector in zip(chunks, embeddings):
        writer.writerow([chunk, vector[:10]])
