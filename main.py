from fastapi import FastAPI
import os
import google.generativeai as genai
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine Direct Mode Live"}

@app.get("/predict")
async def predict_astrology(q: str):
    try:
        # Gemini Setup
        genai.configure(api_key=os.environ.get("GEMINI_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Direct Prompting (No Embeddings needed)
        prompt = f"ඔබ දක්ෂ ජෝතිෂ්‍යවේදියෙක් ලෙස පිළිතුරු දෙන්න. ප්‍රශ්නය: {q}"
        response = model.generate_content(prompt)
        
        return {"prediction": response.text}
        
    except Exception as e:
        return {"prediction": f"Technical Error: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
