import os
import requests
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine Final Bypass Active"}

@app.get("/predict")
async def predict_astrology(q: str):
    api_key = os.environ.get("GEMINI_KEY")
    # මෙතනදී අපි කෙලින්ම Stable v1 URL එක පාවිච්චි කරනවා
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": f"ඔබ දක්ෂ ලාංකීය ජෝතිෂ්‍යවේදියෙක් ලෙස මෙයට පිළිතුරු දෙන්න: {q}"}]
        }]
    }

    try:
        response = requests.post(url, json=payload)
        data = response.json()
        
        # ප්‍රතිඵලය ලබා ගැනීම
        if "candidates" in data:
            prediction = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"prediction": prediction}
        else:
            return {"prediction": f"API Error: {data.get('error', {}).get('message', 'Unknown Error')}"}
            
    except Exception as e:
        return {"prediction": f"Connection Error: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
