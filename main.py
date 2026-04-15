from fastapi import FastAPI
import os
import google.generativeai as genai
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine is Scanning Your Key..."}

@app.get("/predict")
async def predict_astrology(q: str):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_KEY"))
        
        # 1. වැඩ කරන හැම මොඩල් එකක්ම ලැයිස්තුගත කරගමු
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not available_models:
            return {"prediction": "ඔබේ API Key එක සඳහා කිසිදු මොඩල් එකක් සොයාගත නොහැක."}

        # 2. Flash මොඩල් එකක් තියෙනවා නම් ඒක තෝරාගමු, නැත්නම් ලැයිස්තුවේ පළවෙනි එක ගමු
        selected_model = next((m for m in available_models if "flash" in m), available_models[0])
        
        # 3. Response එක ලබාගමු
        model = genai.GenerativeModel(selected_model)
        response = model.generate_content(f"ජෝතිෂ්‍යවේදියෙකු ලෙස පිළිතුරු දෙන්න: {q}")
        
        return {
            "prediction": response.text,
            "system_info": f"Used Model: {selected_model}" # අපි දැනගන්න පාවිච්චි කරපු මොඩල් එකත් එවන්නම්
        }
        
    except Exception as e:
        return {"prediction": f"System Error: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
