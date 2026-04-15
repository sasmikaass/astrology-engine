from fastapi import FastAPI
import os
import google.generativeai as genai
from pinecone import Pinecone
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine Final Version Live"}

@app.get("/predict")
async def predict_astrology(q: str):
    try:
        # Initialize
        genai.configure(api_key=os.environ.get("GEMINI_KEY"))
        pc = Pinecone(api_key=os.environ.get("PINECONE_KEY"))
        index = pc.Index("astrology-knowledge")

        # 1. Get Embeddings using the most stable config
        try:
            embedding_res = genai.embed_content(
                model="models/text-embedding-004",
                content=q,
                task_type="retrieval_query"
            )
        except:
            # Fallback for older keys
            embedding_res = genai.embed_content(
                model="models/embedding-001",
                content=q
            )
        
        emb = embedding_res["embedding"]

        # 2. Search Pinecone
        res = index.query(vector=emb, top_k=3, include_metadata=True)
        context_text = ""
        if res['matches']:
            context_text = " ".join([match['metadata'].get('text', '') for match in res['matches']])
        
        # 3. Generate Content using the most stable model name
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # 'latest' tag added for stability
        
        prompt = f"පහත සඳහන් ජෝතිෂ්‍ය දත්ත ඇසුරින් ප්‍රශ්නයට පිළිතුරු දෙන්න. දත්ත: {context_text}\nප්‍රශ්නය: {q}"
        response = model.generate_content(prompt)
        
        return {"prediction": response.text}
        
    except Exception as e:
        return {"prediction": f"Final Technical Error: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
