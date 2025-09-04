from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__, static_folder="static", template_folder="templates")

# Modal API endpoint
MODAL_API = "https://incirmustafa23--link-cloud-llamacpp-fastapi-app.modal.run/generate"

@app.route("/")
def index():
    return render_template("main.html")   # artık templates klasöründen render ediyor

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    try:
        res = requests.post(MODAL_API, json={"message": user_msg}, timeout=60)
        print(">>> API STATUS:", res.status_code)
        print(">>> API TEXT:", res.text)
        return jsonify(res.json())
    except Exception as e:
        import traceback
        print(">>> ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
