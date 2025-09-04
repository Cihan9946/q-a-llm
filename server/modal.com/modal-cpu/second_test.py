import requests
import time
import json
from typing import List, Dict

URL = "https://incirmustafa23--link-cloud-llamacpp-fastapi-app.modal.run/generate"
HEADERS = {"Content-Type": "application/json"}
JSONL_PATH = "50-questions.jsonl"
TIMEOUT = 120
SPACING_SECONDS = 1
MAX_REQUESTS = 50

def load_messages(path: str, limit: int = 50) -> List[Dict]:
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "message" in obj and isinstance(obj["message"], str):
                    messages.append({"message": obj["message"]})
                    if len(messages) >= limit:
                        break
            except json.JSONDecodeError:
                print(f"⚠️  JSONL satırı ayrıştırılamadı, atlandı: {line[:120]}...")
    return messages

def send_request(payload: Dict, idx: int):
    try:
        t0 = time.perf_counter()
        r = requests.post(URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"\n[{idx:02d}] HTTP {r.status_code} | {dt_ms:.1f} ms")
        print(f"[{idx:02d}] Raw response: {r.text}")
        try:
            data = r.json()
            print(f"[{idx:02d}] Parsed JSON:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
        except Exception:
            print(f"[{idx:02d}] ⚠️ JSON parse edilemedi.")
    except requests.exceptions.RequestException as e:
        print(f"\n[{idx:02d}] ❌ RequestException: {e}")

def main():
    msgs = load_messages(JSONL_PATH, MAX_REQUESTS)
    if not msgs:
        print(f"❌ '{JSONL_PATH}' dosyasından uygun mesaj bulunamadı.")
        return
    if len(msgs) < MAX_REQUESTS:
        print(f"ℹ️  Dosyada {len(msgs)} mesaj var, {MAX_REQUESTS} yerine bunlar gönderilecek.")
        
    overall_start = time.perf_counter()
    
    for i, payload in enumerate(msgs, start=1):
        # İlk istekten sonra bekle
        if i > 1:
            time.sleep(SPACING_SECONDS)
        send_request(payload, i)

    total_sec = time.perf_counter() - overall_start
    print(f"\n✅ Toplam süre: {total_sec:.2f} saniye | Gönderilen istek sayısı: {len(msgs)}")

if __name__ == "__main__":
    main()