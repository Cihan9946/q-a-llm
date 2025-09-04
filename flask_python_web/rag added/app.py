from flask import Flask, render_template, request, jsonify, session
import requests
import uuid
import os
from db import init_db, init_settings, save_selected_model, get_selected_model


from db import init_db, save_message, get_history, get_all_sessions, reset_session

from rag_bridge import rag_answer  # yukarıya ekle
from rag_utils import build_prompt , get_similar_chunks,ask_ollama,rag_ask

app = Flask(__name__)
app.secret_key = os.urandom(24)

OLLAMA_URL = "http://localhost:11434/api/generate"

# Modelleri bir yapı içinde tutalım
MODELS = {
    "turkcell-custom": "Turkcell FineTune Model",
    "turkcell-unsloth":"Turkcell FineTune unsloth Model",
    "koc-custom": "Koç FineTune Model",
    "commencis-custom": "commencis FineTune Model",
    "trendyol-custom": "trendyol FineTune Model"
    
    # Buraya yeni modelleri ekleyebilirsiniz
}

DB_CONFIG = {
    "dbname": "ragdb",
    "user": "postgres",
    "password": "secret",  # kendi şifreni kullan
    "host": "localhost",
    "port": "5432"
}

@app.route("/")
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    sid = session['session_id']
    history = get_history(sid)
    all_sessions = get_all_sessions()
    return render_template("chat.html", history=history, all_sessions=all_sessions, models=MODELS)

@app.route("/send_message", methods=["POST"])
def send_message():
    user_msg = request.json.get("message")
    selected_model = request.json.get("model", "turkcell-custom")  # Seçilen model, default "turkcell-custom"
    sid = session.get('session_id')

    history = get_history(sid)
    bot_msg = generate_response(user_msg, history, selected_model)

    save_message(sid, user_msg, bot_msg)
    return jsonify({"response": bot_msg})

@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get('session_id')
    reset_session(sid)
    return jsonify({"status": "ok"})

@app.route("/switch_session", methods=["POST"])
def switch_session():
    new_sid = request.json.get("session_id")
    session['session_id'] = new_sid
    return jsonify({"status": "switched"})

@app.route("/train_model", methods=["GET", "POST"])
def train_model():
    if request.method == "POST":
        dataset = request.files.get("dataset")
        model_files = request.files.getlist("model_files")
        epochs = request.form.get("epochs")
        batch_size = request.form.get("batch_size")
        
        # 💡 Dosyaları ve parametreleri kontrol etmek için log at
        print("✅ Fine-tune başlatıldı")
        print("Dataset:", dataset.filename)
        print("Epochs:", epochs)
        print("Batch Size:", batch_size)
        print("Toplam model dosyası:", len(model_files))

        # Burada eğitim işlemini başlatabilirsin

        return "🚀 Model eğitimi başlatıldı!"
    
    # Eğer GET isteği gelirse, formu göster
    return render_template("train_model.html")

def generate_response(message, history, model):
    system_prompt = (
    "Sen yalnızca Link-Cloud ERP hakkında konuşan bir yapay zeka asistanısın. "
    "Link-Cloud, Link Bilgisayar tarafından geliştirilen bir ERP yazılımıdır. "
  
    "Sana gelen her mesajı önce dikkatlice analiz et. Eğer mesaj Link-Cloud ERP ile ilgiliyse cevap ver. "
    "Ama mesaj ERP ile ilgili değilse kesinlikle cevap üretme. Bu durumda sadece aşağıdaki cevabı ver:\n\n"
    "'Üzgünüm, ben sadece Link-Cloud ERP hakkında yardımcı olabilirim. Yardımcı olabileceğim farklı bir konu hakkında sorabilirsiniz "
    "216 522 00 00 numaralı telefondan veya info@linkbilgisayar.com.tr adresinden ulaşabilirsiniz. "
    "Link-Cloud ERP sistemi ile ilgili sorularınızı memnuniyetle yanıtlayabilirim.'\n\n"
    " Yanıltıcı veya alakasız konulara cevap verme. Aşağıdaki örnekler ERP dışı konulara örnektir:\n"
    "- Matematik soruları (örneğin: 2+2 kaç eder?)\n"
    "- Film önerisi, dizi tavsiyesi\n"
    "- Hava durumu, sağlık bilgisi, genel kültür\n"
    "- Eğlence amaçlı sorular\n\n"
    "Bu tür sorular geldiğinde sadece yukarıdaki sabit mesajı ver. Sakın başka bir cevap üretme."
    "çözüm odaklı ve yol gösterici ol"
    )

    # 🔁 Chat geçmişini oluştur
    prompt = f"<|system|>\n{system_prompt}\n"
    for item in history:
        prompt += f"Kullanıcı: {item['user']}\nAsistan: {item['bot']}\n"
    prompt += f"Kullanıcı: {message}\nAsistan:"

    payload = {
        "model": model,  # Seçilen model buraya ekleniyor
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "").strip()
    else:
        return "⚠️ Sunucu hatası: Yanıt alınamadı."
    
    
@app.route("/api_settings", methods=["GET", "POST"])
def api_settings():
    current_model = get_selected_model()

    if request.method == "POST":
        selected_model = request.form.get("selected_model")
        save_selected_model(selected_model)
        current_model = selected_model
        return render_template("api_settings.html", models=MODELS, current_model=current_model, message="Model kaydedildi.")

    return render_template("api_settings.html", models=MODELS, current_model=current_model)






@app.route("/rag_chat", methods=["POST"])
def rag_chat():
    try:
        data = request.json
        user_msg = data.get("message")
        model = data.get("model", "turkcell-custom")
        rag_active = data.get("rag_active", True)  # Default True

        sid = session.get('session_id')  # 🔹 Oturumu al

        print(f"\n🔵 RAG İSTEĞİ ALINDI (Durum: {'AKTİF' if rag_active else 'PASİF'})")
        print(f"📩 Mesaj: {user_msg}")
        print(f"🤖 Model: {model}")

        # Veritabanından chunk'ları al
        chunks = get_similar_chunks(user_msg)
        print(f"📚 Bulunan chunk sayısı: {len(chunks)}")

        if not chunks:
            response = "⚠️ RAG: İlgili bilgi bulunamadı"
            print(response)
            save_message(sid, user_msg, response)  # ❗ Kaydet
            return jsonify({"response": response})

        # Basit prompt template
        context = "\n".join([f"• {chunk[:200]}..." for chunk in chunks[:3]])  # İlk 3 chunk'ın kısaltılmış hali
        prompt = f"""<|system|>
Aşağıdaki bilgileri kullanarak soruyu cevaplayın. Bilgi yoksa "Bilgi bulunamadı" deyin.

Bağlam:
{context}

Soru: {user_msg}
Cevap:"""

        print(f"\n📝 Prompt (kısaltılmış):\n{prompt[:300]}...\n")

        # Ollama'ya istek
        response = ask_ollama({
            "model": model,
            "prompt": prompt,
            "stream": False
        })

        # Yanıtı işaretle
        marked_response = f"{response}\n\n🔍 [RAG ile oluşturuldu - Kaynaklar: {len(chunks)}]"
        print(f"🟢 Yanıt: {response[:200]}...")

        # 🔸 Sohbet geçmişine kaydet (tıpkı send_message gibi)
        save_message(sid, user_msg, marked_response)

        return jsonify({"response": marked_response})

    except Exception as e:
        error_msg = f"RAG Hatası: {str(e)}"
        print(f"🔴 {error_msg}")
        return jsonify({"response": error_msg}), 500



@app.route("/test_rag", methods=["GET"])
def test_rag():
    try:
        test_question = "Stok takibi nasıl yapılır?"
        print(f"\n🔍 Test sorusu: {test_question}")
        
        # 1. Embedding oluştur
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(test_question)
        print(f"✅ Vektör oluşturuldu (boyut: {len(embedding)})")
        
        # 2. Veritabanı bağlantısı
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # 3. Basit sorgu (doküman sayısı)
        cur.execute("SELECT COUNT(*) FROM documents")
        doc_count = cur.fetchone()[0]
        print(f"📊 Veritabanında {doc_count} doküman bulundu")
        
        # 4. Benzerlik sorgusu
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s) as similarity 
            FROM documents 
            ORDER BY similarity DESC 
            LIMIT 3
        """, (embedding.tolist(),))
        
        results = cur.fetchall()
        print(f"🔎 {len(results)} benzer doküman bulundu")
        
        if not results:
            return jsonify({
                "status": "error",
                "message": "Benzer doküman bulunamadı",
                "suggestion": "Vektör veritabanını kontrol edin"
            })
        
        return jsonify({
            "status": "success",
            "question": test_question,
            "documents_found": doc_count,
            "top_match": {
                "similarity": round(results[0][1], 4),
                "content_preview": results[0][0][:100] + "..."
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "critical_error",
            "error": str(e),
            "solution": "PostgreSQL bağlantısını ve embedding modelini kontrol edin"
        })

if __name__ == "__main__":
    init_db()
    init_settings()

    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
