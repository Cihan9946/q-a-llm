# test.py
import requests, time, json

URL = "https://incirmustafa23--link-cloud-llamacpp-fastapi-app.modal.run/generate"
PAYLOAD = {"message": "Link-Cloud ERP nedir?"}
HEADERS = {"Content-Type": "application/json"}

def main():
    try:
        t0 = time.perf_counter()
        r = requests.post(URL, json=PAYLOAD, headers=HEADERS, timeout=120)
        dt = (time.perf_counter() - t0) * 1000
        print(f"HTTP {r.status_code} | {dt:.1f} ms")
        # Ham metni daima göster
        print("Raw response:", r.text)
        # JSON ise ayrıştır
        try:
            data = r.json()
            print("Parsed JSON:", json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            print("⚠️ JSON parse edilemedi.")
    except requests.exceptions.RequestException as e:
        print("❌ RequestException:", e)

if __name__ == "__main__":
    main()
