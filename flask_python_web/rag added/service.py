from flask import Flask, request, jsonify, session
import requests
import uuid
import os
from db import get_selected_model  # DB'den seçilen modeli al
from db import init_settings  


app = Flask(__name__)
app.secret_key = os.urandom(24)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Basit session-temelli history tutucu (RAM içi)
user_histories = {}

# Sabit prompt
SYSTEM_PROMPT = (
    "Sen yalnızca Link-Cloud ERP hakkında konuşan bir yapay zeka asistanısın. "
    "Link-Cloud, Link Bilgisayar tarafından geliştirilen bir ERP yazılımıdır. "
    "Sana gelen her mesajı önce dikkatlice analiz et. Eğer mesaj Link-Cloud ERP ile ilgiliyse cevap ver. "
    "Ama mesaj ERP ile ilgili değilse kesinlikle cevap üretme. Bu durumda sadece aşağıdaki cevabı ver:\n\n"
    "'Üzgünüm, ben sadece Link-Cloud ERP hakkında yardımcı olabilirim. Yardımcı olabileceğim farklı bir konu hakkında sorabilirsiniz "
    "216 522 00 00 numaralı telefondan veya info@linkbilgisayar.com.tr adresinden ulaşabilirsiniz. "
    "Link-Cloud ERP sistemi ile ilgili sorularınızı memnuniyetle yanıtlayabilirim.'\n\n"
    "Yanıltıcı veya alakasız konulara cevap verme. Aşağıdaki örnekler ERP dışı konulara örnektir:\n"
    "- Matematik soruları (örneğin: 2+2 kaç eder?)\n"
    "- Film önerisi, dizi tavsiyesi\n"
    "- Hava durumu, sağlık bilgisi, genel kültür\n"
    "- Eğlence amaçlı sorular\n\n"
    "Bu tür sorular geldiğinde sadece yukarıdaki sabit mesajı ver. Sakın başka bir cevap üretme."
   
)

@app.before_request
def ensure_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        user_histories[session['session_id']] = []

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    message = data.get("message")
    
    if not message:
        return jsonify({"error": "Mesaj boş olamaz."}), 400

    # 💾 Kullanıcının en son seçtiği modeli veritabanından al
    model = get_selected_model()

    sid = session['session_id']
    history = user_histories.get(sid, [])

    # 🔁 Prompt oluştur
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n"
    for item in history:
        prompt += f"Kullanıcı: {item['user']}\nAsistan: {item['bot']}\n"
    prompt += f"Kullanıcı: {message}\nAsistan:"

    # Ollama API çağrısı
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        bot_reply = result.get("response", "").strip()

        # ➕ History güncelle
        user_histories[sid].append({
            "user": message,
            "bot": bot_reply
        })

        return jsonify({"response": bot_reply})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Ollama hatası", "details": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset_history():
    sid = session.get("session_id")
    if sid in user_histories:
        user_histories[sid] = []
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    init_settings()
    
    from waitress import serve
    serve(app, host="0.0.0.0", port=6000)
