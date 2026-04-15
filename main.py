from fastapi import FastAPI
import os
import google.generativeai as genai
from pinecone import Pinecone
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine is Ready"}

@app.get("/predict")
async def predict_astrology(q: str):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_KEY"))
        pc = Pinecone(api_key=os.environ.get("PINECONE_KEY"))
        index = pc.Index("astrology-knowledge")

        # 1. වැඩ කරන මොඩල් ලැයිස්තුව ගමු
        all_models = [m.name for m in genai.list_models()]
        
        # 2. Embedding මොඩල් එක තෝරාගමු
        embed_model = "models/text-embedding-004" if "models/text-embedding-004" in all_models else "models/embedding-001"
        
        # 3. Generate කරන මොඩල් එක තෝරාගමු (Flash)
        # ලැයිස්තුවේ තියෙන ඕනෑම flash මොඩල් එකක් ගමු
        flash_models = [m for m in all_models if "gemini-1.5-flash" in m]
        gen_model_name = flash_models[0] if flash_models else "models/gemini-pro"

        # Embed Content
        embedding_res = genai.embed_content(
            model=embed_model,
            content=q,
            task_type="retrieval_query" if "004" in embed_model else None
        )
        emb = embedding_res["embedding"]

        # Pinecone Search
        res = index.query(vector=emb, top_k=3, include_metadata=True)
        context_text = " ".join([match['metadata'].get('text', '') for match in res['matches']])
        
        # Generate Response
        model = genai.GenerativeModel(gen_model_name)
        prompt = f"පහත සඳහන් ජෝතිෂ්‍ය දත්ත ඇසුරින් ප්‍රශ්නයට පිළිතුරු දෙන්න. දත්ත: {context_text}\nප්‍රශ්නය: {q}"
        response = model.generate_content(prompt)
        
        return {"prediction": response.text}
        
    except Exception as e:
        return {"prediction": f"Technical Update Required: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
