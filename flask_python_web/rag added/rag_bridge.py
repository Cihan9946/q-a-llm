# rag_bridge.py
# Bu dosya app.py'den izole RAG işlemleri yapar

# rag_bridge.py'deki import satırını değiştirin
from rag_utils import get_similar_chunks, build_prompt , ask_ollama

def rag_answer(user_message):
    try:
        print("\n" + "="*50)
        print("🔄 RAG Sistemi Aktif | Gelen Mesaj:", user_message)
        
        chunks = get_similar_chunks(user_message)
        print(f"📚 Bulunan Bilgi Parçaları ({len(chunks)} adet):")
        for i, chunk in enumerate(chunks[:3]):  # İlk 3 chunk'ı göster
            print(f"{i+1}. {chunk[:100]}...")
        
        if not chunks:
            print("❌ Uygun bilgi bulunamadı!")
            return "RAG: Konuyla ilgili kaynak bulunamadı"
        
        prompt = build_prompt(user_message, chunks)
        print("\n💡 Oluşturulan Prompt:")
        print(prompt[:500] + "...")  # Uzun prompt'lar için kısalt
        
        response = ask_ollama(prompt).strip()
        print("\n✅ RAG Yanıtı:", response[:200] + "..." if len(response) > 200 else response)hf_
        
        return response + "\n\n[RAG Sistemi ile oluşturulmuştur]"
        
    except Exception as e:
        print("🔥 RAG Hatası:", str(e))
        return f"RAG Sistemi Hatası: {str(e)}"