import os
import requests
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine - Groq Latest Model Active"}

@app.get("/predict")
async def predict_astrology(q: str):
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # දැනට තියෙන අලුත්ම සහ ස්ථාවරම මොඩල් එක: llama-3.3-70b-versatile
    payload = {
        "model": "llama-3.3-70b-versatile", 
        "messages": [
            {
                "role": "system", 
                "content": "ඔබ දක්ෂ ලාංකීය ජෝතිෂ්‍යවේදියෙක්. කරුණාකර ඔබගේ පිළිතුර ඉතා කෙටියෙන්, පැහැදිලිව සිංහලෙන් ලබා දෙන්න. උපරිම වචන 150කට සීමා කරන්න."
            },
            {"role": "user", "content": q}
        ],
        "temperature": 0.7,
        "max_tokens": 400 
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        
        if "choices" in data:
            prediction = data["choices"][0]["message"]["content"]
            return {"prediction": prediction}
        else:
            # මොඩල් එකේ ප්‍රශ්නයක් ආවොත් ලේසි මොඩල් එකකට (Fallback) මාරු වෙමු
            fallback_payload = payload.copy()
            fallback_payload["model"] = "llama3-8b-8192"
            res_fallback = requests.post(url, json=fallback_payload, headers=headers)
            data_fb = res_fallback.json()
            if "choices" in data_fb:
                return {"prediction": data_fb["choices"][0]["message"]["content"]}
            
            return {"prediction": f"Groq Error: {data.get('error', {}).get('message', 'Unknown Error')}"}
            
    except Exception as e:
        return {"prediction": f"Network Error: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
