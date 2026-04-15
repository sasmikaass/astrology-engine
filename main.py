import os
import requests
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine Diagnostics Mode"}

@app.get("/predict")
async def predict_astrology(q: str):
    api_key = os.environ.get("GEMINI_KEY")
    
    # 1. මුලින්ම පරීක්ෂා කරමු ඔයාගේ Key එකට තියෙන මොඩල් මොනවද කියලා
    list_url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
    models_data = requests.get(list_url).json()
    
    # වැඩ කරන පළවෙනි මොඩල් එක තෝරාගමු
    model_name = "models/gemini-1.5-flash" # Default
    if "models" in models_data:
        for m in models_data["models"]:
            if "generateContent" in m["supportedGenerationMethods"]:
                model_name = m["name"]
                break
    
    # 2. දැන් ඒ හොයාගත්තු මොඩල් එකෙන් උත්තරය ගමු
    predict_url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": f"ජෝතිෂ්‍යවේදියෙකු ලෙස පිළිතුරු දෙන්න: {q}"}]}]
    }

    try:
        response = requests.post(predict_url, json=payload)
        data = response.json()
        
        if "candidates" in data:
            return {
                "prediction": data["candidates"][0]["content"]["parts"][0]["text"],
                "model_used": model_name
            }
        else:
            return {"prediction": f"මොඩල් එක හමු වුණා ({model_name}), නමුත් වැරැද්දක් ආවා: {data.get('error', {}).get('message')}"}
            
    except Exception as e:
        return {"prediction": f"සම්බන්ධතා දෝෂයකි: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
