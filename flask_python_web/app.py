from flask import Flask, render_template, request, jsonify, session
import requests
import uuid
import os
from db import init_db, init_settings, save_selected_model, get_selected_model

from db import init_db, save_message, get_history, get_all_sessions, reset_session

app = Flask(__name__)
app.secret_key = os.urandom(24)

OLLAMA_URL = "http://localhost:11434/api/generate"

# Modelleri bir yapı içinde tutalım
MODELS = {
    "turkcell-custom": "Turkcell FineTune Model",
    "koc-custom": "Koç FineTune Model",
    "commencis-custom": "commencis FineTune Model",
    "trendyol-custom": "trendyol FineTune Model"
    # Buraya yeni modelleri ekleyebilirsiniz
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

if __name__ == "__main__":
    init_db()
    init_settings()

    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
