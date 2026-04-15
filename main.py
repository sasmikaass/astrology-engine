from fastapi import FastAPI
import os
import google.generativeai as genai
import uvicorn

app = FastAPI()

# සර්වර් එක පටන් ගන්නකොටම එක පාරක් Configuration එක කරමු
genai.configure(api_key=os.environ.get("GEMINI_KEY"))

@app.get("/")
def home():
    return {"status": "Alpha Engine is Fast and Ready"}

@app.get("/predict")
async def predict_astrology(q: str):
    try:
        # කෙලින්ම වඩාත් ස්ථාවර මොඩල් එකට යමු (Timeout නොවී ඉන්න)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # පිළිතුර ඉක්මනින් ලබාගැනීමට stream=False (default) භාවිතා කරමු
        response = model.generate_content(f"ජෝතිෂ්‍යවේදියෙකු ලෙස කෙටියෙන් පිළිතුරු දෙන්න: {q}")
        
        return {"prediction": response.text}
        
    except Exception as e:
        # මොකක් හරි වැරදුනොත් පරණ version එකත් try කරමු
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(q)
            return {"prediction": response.text}
        except:
            return {"prediction": f"Error: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
