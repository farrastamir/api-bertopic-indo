from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from bertopic import BERTopic
import pandas as pd

# 1. Inisiasi Aplikasi FastAPI
app = FastAPI(title="API BERTopic Bahasa Indonesia")

# 2. Setup CORS (Agar web Vercel Anda tidak diblokir saat meminta data)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan akses dari semua web (termasuk Vercel & localhost)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Format Input Data
class DataInput(BaseModel):
    documents: List[str]

# 4. Inisiasi BERTopic (Menggunakan model Multilingual yang bagus untuk Indo)
# Model akan didownload otomatis saat aplikasi pertama kali dijalankan
print("Mempersiapkan Model AI... (Mungkin butuh waktu beberapa saat)")
topic_model = BERTopic(
    language="multilingual", 
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    min_topic_size=3 # Ubah angka ini jika data Anda sedikit (misal jadi 2 atau 3)
)

# 5. Endpoint Utama untuk Analisis
@app.post("/analyze")
async def analyze_text(data: DataInput):
    docs = data.documents
    
    if len(docs) < 5:
        raise HTTPException(status_code=400, detail="Minimal masukkan 5 dokumen untuk dianalisis.")

    try:
        # Menjalankan algoritma BERTopic
        topics, probs = topic_model.fit_transform(docs)
        
        # Mengambil info topik yang terbentuk
        topic_info = topic_model.get_topic_info()
        
        # Menyusun output JSON agar rapi
        result = []
        for index, row in topic_info.iterrows():
            topic_id = row['Topic']
            # Ambil 5 kata kunci teratas untuk topik ini
            keywords = [word[0] for word in topic_model.get_topic(topic_id)] if topic_id != -1 else ["outlier", "noise"]
            
            # Cari dokumen mana saja yang masuk ke topik ini
            docs_in_topic = [docs[i] for i, t in enumerate(topics) if t == topic_id]
            
            result.append({
                "topic_id": int(topic_id),
                "topic_name": f"Topik {topic_id}" if topic_id != -1 else "Data Tidak Relevan (Noise)",
                "keywords": keywords,
                "count": len(docs_in_topic),
                "documents": docs_in_topic
            })
            
        return {"status": "success", "total_data": len(docs), "clusters": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pengecekan apakah server hidup
@app.get("/")
def read_root():
    return {"message": "Server BERTopic Aktif dan Berjalan!"}