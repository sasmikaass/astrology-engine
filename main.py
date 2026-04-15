import os
import requests
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine - Groq High Speed Mode Active"}

@app.get("/predict")
async def predict_astrology(q: str):
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192", # ඉතාමත් බුද්ධිමත් සහ දක්ෂ මොඩල් එකක්
        "messages": [
            {
                "role": "system", 
                "content": "ඔබ දක්ෂ ලාංකීය ජෝතිෂ්‍යවේදියෙක්. ඔබ සැමවිටම ජෝතිෂ්‍ය කරුණු ඇසුරින් ඉතාමත් පැහැදිලිව සිංහලෙන් පිළිතුරු දිය යුතුය."
            },
            {"role": "user", "content": q}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        
        if "choices" in data:
            prediction = data["choices"][0]["message"]["content"]
            return {"prediction": prediction}
        else:
            return {"prediction": f"Groq API Error: {data.get('error', {}).get('message', 'Unknown Error')}"}
            
    except Exception as e:
        return {"prediction": f"Network Error: {str(e)}"}

if __name__ == "__main__":
    # Railway එකේ Port එකට ලොක් කිරීම
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
