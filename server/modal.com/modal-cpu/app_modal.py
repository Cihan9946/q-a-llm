import os
import json
import faiss
import numpy as np
import modal
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# Modal App
app = modal.App("link-cloud-llamacpp")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "python3-dev", "git")
    .pip_install_from_requirements("requirements.txt")
)

# GGUF model volume
gguf_vol = modal.Volume.from_name("gguf-models", create_if_missing=False)

# Kaynak ayarları (7B Q4 için optimize)
CPU_CORES = float(os.environ.get("CPU_CORES", "12"))      # 12 vCPU
MEM_MB = int(os.environ.get("MEM_MB", "16384"))           # 16 GB RAM
THREADS = int(os.environ.get("LLM_THREADS", str(int(CPU_CORES))))  # CPU core sayısı kadar thread
CTX = int(os.environ.get("LLM_CTX", "8192"))              # 32K değil, 8K context
N_BATCH = int(os.environ.get("LLM_N_BATCH", "64"))        # küçük batch CPU için hızlı

MODEL_PATH = "/models/models/main_dposuz_q4_0.gguf"
VECTORS_FILE = "/models/models/kesin_cevaplar_data_vectors_v3.jsonl"

# Global değişkenler - artık class içinde yönetilecek
# EMBED_MODEL, INDEX, questions, answers, metadata artık LlamaWorker içinde

# ---------------- Llama Worker ----------------
@app.cls(
    image=image,
    volumes={"/models": gguf_vol},
    cpu=CPU_CORES,
    memory=MEM_MB,
    min_containers=1,
    max_containers=3
)
class LlamaWorker:
    llm: Any = None
    embed_model: Any = None
    index: Any = None
    questions: List[str] = []
    answers: List[str] = []
    metadata: List[Dict] = []

    def initialize_embeddings(self):
        """Embedding modelini ve FAISS indexini başlat"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
            
            if os.path.exists(VECTORS_FILE):
                print("🔄 Vektörlü JSONL yükleniyor...")
                embeddings = []
                self.questions, self.answers, self.metadata = [], [], []
                
                with open(VECTORS_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        self.questions.append(data["instruction"])
                        self.answers.append(data["output"])
                        self.metadata.append(data.get("metadata", {}))
                        
                        if "embedding" in data:
                            emb = np.array(data["embedding"], dtype=np.float32)
                        else:
                            emb = self.embed_model.encode(data["instruction"], convert_to_numpy=True).astype("float32")
                        
                        embeddings.append(emb)
                
                if embeddings:
                    embeddings = np.vstack(embeddings).astype("float32")
                    print(f"📏 Embedding boyutu: {embeddings.shape}")
                    
                    # Tüm embeddingleri normalize et
                    faiss.normalize_L2(embeddings)
                    
                    self.index = faiss.IndexFlatIP(embeddings.shape[1])
                    self.index.add(embeddings)
                    print(f"✅ FAISS index yüklendi: {self.index.ntotal} kayıt")
                else:
                    print("⚠️ JSONL dosyasında embedding bulunamadı")
            else:
                print("⚠️ JSONL dosyası bulunamadı, yalnızca LLM kullanılacak")
                
        except ImportError as e:
            print(f"❌ Import hatası: {e}")
        except Exception as e:
            print(f"❌ Embedding başlatma hatası: {e}")

    def search_in_jsonl(self, query, top_k=5, min_sim=0.85):
        if self.index is None or self.embed_model is None:
            print("❌ Index veya Embedding modeli yüklenmemiş")
            return []
        
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)

        print(f"📊 Arama sonuçları: {len(I[0])} kayıt bulundu")
        print(f"📈 Benzerlik skorları: {[f'{score:.4f}' for score in D[0]]}")
        
        results = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
            sim = float(score)
            question_preview = self.questions[idx][:100] + "..." if len(self.questions[idx]) > 100 else self.questions[idx]
            print(f"  {rank}. Index: {idx}, Benzerlik: {sim:.4f}, Soru: {question_preview}")
            
            if sim >= min_sim:
                results.append({
                    "soru": self.questions[idx],
                    "cevap": self.answers[idx],
                    "benzerlik": round(sim * 100, 2),
                    "metadata": self.metadata[idx]
                })
            else:
                print(f"  ⚠️  {rank}. Benzerlik {sim:.4f} < {min_sim} - atlandı")
        
        print(f"✅ {len(results)} uygun sonuç bulundu")
        return results

    @modal.enter()
    def load_model(self):
        from llama_cpp import Llama
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"GGUF model bulunamadı: {MODEL_PATH}")

        self.llm = Llama(
            model_path=MODEL_PATH,
            n_threads=THREADS,
            n_ctx=CTX,
            n_batch=N_BATCH,
            verbose=False,
        )
        
        # Embedding modelini başlat
        self.initialize_embeddings()

    def _run(self, message: str, history: list) -> str:
        system_prompt = (
            "Sadece Link-Cloud ERP hakkında konuş . "
            "Sadece Link-Cloud ERP hakkında konuşan asistansın. . "
            "Eğer soru ERP ile ilgiliyse kısa ve net yanıt ver. "
            "Eğer soru ERP dışındaysa şu sabit yanıtı ver:\n\n"
            "Üzgünüm, ben sadece Link-Cloud ERP hakkında yardımcı olabilirim."
        )

        short_history = history[-2:] if history else []

        prompt = f"<|system|>\n{system_prompt}\n"
        for h in short_history:
            prompt += f"Kullanıcı: {h['user']}\nAsistan: {h['bot']}\n"
        prompt += f"Kullanıcı: {message}\nAsistan:"

        # Streaming ile token token üret
        output = ""
        for token in self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            stop=["Kullanıcı:", "Asistan:", "<|im_end|>"],
            stream=True
        ):
            output += token["choices"][0]["text"]
        return output.strip()

    @modal.method()
    def generate(self, message: str, history: list = None) -> dict:
        # Önce JSONL araması
        hits = self.search_in_jsonl(message, top_k=1, min_sim=0.90)
        if hits:
            print(f"✅ JSONL içinde {len(hits)} uygun cevap bulundu, en iyisi döndürülüyor")
            print(f"📝 En iyi benzerlik: {hits[0]['benzerlik']}%")
            return {
                "response": hits[0]["cevap"], 
                "source": "jsonl", 
                "similarity": hits[0]["benzerlik"],
                "total_matches": len(hits)
            }

        # JSONL eşleşme yoksa Llama çalıştır
        print("❌ JSONL'da yeterli eşleşme bulunamadı, LLM'e yönlendiriliyor...")
        text = self._run(message, history or [])
        return {
            "response": text, 
            "source": "llm",
            "similarity": 0.0,
            "total_matches": 0
        }

    @modal.method()
    def test_search(self, message: str, top_k: int = 10, min_sim: float = 0.0) -> dict:
        """Sadece arama sonuçlarını test etmek için"""
        hits = self.search_in_jsonl(message, top_k=top_k, min_sim=min_sim)
        
        return {
            "query": message,
            "results": hits,
            "total_found": len(hits),
            "search_params": {
                "top_k": top_k,
                "min_similarity": min_sim
            }
        }


# ---------------- FastAPI ----------------
api = FastAPI(title="Link-Cloud ERP LLM API")

class GenReq(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

class TestSearchReq(BaseModel):
    message: str
    top_k: Optional[int] = 10
    min_sim: Optional[float] = 0.0

@app.function(image=image, cpu=0.25, memory=512)
@modal.asgi_app()
def fastapi_app():
    return api

@api.post("/generate")
def generate_endpoint(body: GenReq):
    worker = LlamaWorker()
    result = worker.generate.remote(body.message, body.history or [])
    return result

@api.post("/test-search")
def test_search_endpoint(body: TestSearchReq):
    worker = LlamaWorker()
    result = worker.test_search.remote(body.message, body.top_k, body.min_sim)
    return result

# Health check endpoint
@api.get("/health")
def health_check():
    # Basit bir health check, her seferinde worker oluşturmak yerine
    return {"status": "healthy", "service": "link-cloud-llm-api"}

# Debug endpoint
@api.get("/debug")
def debug_info():
    worker = LlamaWorker()
    # Debug bilgilerini almak için özel bir method ekleyebilirsiniz
    return {"status": "use /test-search for debugging"}