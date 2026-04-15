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
    # Railway Variables වලින් Groq Key එක ලබා ගැනීම
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192", 
        "messages": [
            {
                "role": "system", 
                "content": "ඔබ දක්ෂ ලාංකීය ජෝතිෂ්‍යවේදියෙක්. කරුණාකර ඔබගේ පිළිතුර ඉතා කෙටියෙන්, පැහැදිලිව සිංහලෙන් ලබා දෙන්න. උපරිම වචන 150-200ක් පමණක් භාවිතා කරන්න."
            },
            {"role": "user", "content": q}
        ],
        "temperature": 0.7,
        "max_tokens": 500  # උත්තරේ දිග සීමා කිරීමට (Telegram error එක වැලැක්වීමට)
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        
        if "choices" in data:
            prediction = data["choices"][0]["message"]["content"]
            return {"prediction": prediction}
        else:
            error_msg = data.get('error', {}).get('message', 'Unknown Error')
            return {"prediction": f"Groq API Error: {error_msg}"}
            
    except Exception as e:
        return {"prediction": f"Network Error: {str(e)}"}

if __name__ == "__main__":
    # Railway එකේ PORT එකට අනුකූලව සර්වර් එක ක්‍රියාත්මක කිරීම
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
