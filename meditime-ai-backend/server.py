from fastapi import FastAPI, File, UploadFile                 # API çatısı ve dosya yükleme tipleri
from fastapi.responses import JSONResponse                    #JSON yanıt döndürmek için (opsiyonel ama netlik sağlar)
from fastapi.middleware.cors import CORSMiddleware            # Flutter/web istekleri için CORS izni
import numpy as np                                            # byte -> numpy dizisi (OpenCV bununla çalışır)
import cv2                                                    # OpenCV: resim işleme 

app = FastAPI(title="MediTime AI Backend", version="0.1.0")    # Uygulama nesnemiz

# (Geliştirme kolaylığı için) CORS: her kaynağa izin veriyoruz.
# Üretimde allow_origins'i kendi domain’inle sınırlandır.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # DEV: * (her yer). PROD: ["https://senin-domainin.com"]
    allow_credentials=True,
    allow_methods=["*"],             # GET, POST, v.s. hepsi
    allow_headers=["*"],             # Authorization, Content-Type v.s.
)

@app.get("/")      # Sağlık kontrolü: “server ayakta mı?”
def root():
    return {"status": "ok", "message": "MediTime AI Server çalışıyor."}

@app.post("/predict-image")   # Flutter’dan/istemciden resim gönderilecek uç nokta
async def predict_image(file: UploadFile = File(...)):
    # 1) Dosyayı ham byte olarak oku
    content = await file.read()
    
    # 2) Byte -> numpy uint8 dizisine çevir (OpenCV buna ihtiyaç duyar)
    nparr =  np.frombuffer(content, np.uint8)
    
    # 3) Numpy buffer -> OpenCV BGR imaj (cv2.imdecode, dosya uzantısına göre çözer: JPG/PNG)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 4) Decode başarısızsa hata dön
    if img is None:
        return JSONResponse({"status":  "error", "detail": "Image decode edilemedi"}, status_code=400)
    
    # 5) Örnek “işleme”: boyut ölç, griye çevir, ortalama parlaklık hesapla
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))   # Python float'a çevir (JSON serileştirme için)
    
    # 6) Buraya ileride “ilaç kutusu algılama / OCR” modeli eklenecek.
    #    Örn: kutu tespit -> ROI kırp -> OCR ile yazıyı oku -> sonuçları dön
    
    # 7) JSON yanıt: istemci kolayca parse etsin
    return{
        "status": "ok",
        "width": int(w),
        "height": int(h),
        "mean_brightness": round(mean_brightness, 2),
        "info": "Model entegresyonu icin hazir (kutuyu algilama/OCR burda calisacak)."
    }
    
    # Bu blok sayesinde: `python server.py` dersen uvicorn’u programatik başlatır.
    # (İstersen terminalde `uvicorn server:app --reload` da kullanabilirsin.)
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)