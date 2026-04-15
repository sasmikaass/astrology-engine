from fastapi import FastAPI
import os
import google.generativeai as genai
from pinecone import Pinecone
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine is Scanning Models"}

@app.get("/predict")
async def predict_astrology(q: str):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_KEY"))
        pc = Pinecone(api_key=os.environ.get("PINECONE_KEY"))
        index = pc.Index("astrology-knowledge")

        # වැඩ කරන Embedding මොඩල් එක හොයාගැනීම
        available_models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
        
        # අපිට අවශ්‍ය මොඩල් එක ලැයිස්තුවේ තියෙනවද බලමු
        target_model = "models/text-embedding-004" if "models/text-embedding-004" in available_models else available_models[0]

        embedding_res = genai.embed_content(
            model=target_model,
            content=q,
            task_type="retrieval_query" if "004" in target_model else None
        )
        emb = embedding_res["embedding"]

        res = index.query(vector=emb, top_k=3, include_metadata=True)
        context_text = " ".join([match['metadata'].get('text', '') for match in res['matches']])
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"දත්ත: {context_text}\nප්‍රශ්නය: {q}")
        
        return {"prediction": response.text}
        
    except Exception as e:
        return {"prediction": f"Final technical error: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
