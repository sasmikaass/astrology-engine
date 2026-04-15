from fastapi import FastAPI
import os
import google.generativeai as genai
from pinecone import Pinecone
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Alpha Engine is Live and Ready"}

@app.get("/predict")
async def predict_astrology(q: str):
    try:
        # Configuration
        genai.configure(api_key=os.environ.get("GEMINI_KEY"))
        pc = Pinecone(api_key=os.environ.get("PINECONE_KEY"))
        index = pc.Index("astrology-knowledge")

        # Get Embeddings using the latest model
        embedding_res = genai.embed_content(
            model="models/text-embedding-004",
            content=q,
            task_type="retrieval_query"
        )
        emb = embedding_res["embedding"]

        # Search Pinecone
        res = index.query(vector=emb, top_k=3, include_metadata=True)
        
        # Build Context
        context_text = ""
        if res['matches']:
            context_text = " ".join([match['metadata'].get('text', '') for match in res['matches']])
        
        # Generate Response
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"පහත සඳහන් ජෝතිෂ්‍ය දත්ත ඇසුරින් ප්‍රශ්නයට පිළිතුරු දෙන්න. දත්ත: {context_text}\nප්‍රශ්නය: {q}"
        response = model.generate_content(prompt)
        
        return {"prediction": response.text}
        
    except Exception as e:
        return {"prediction": f"Error: {str(e)}"}

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
