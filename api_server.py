from fastapi import FastAPI, Request
import faiss
import numpy as np
from embedding_and_faiss import embed_text
from llm_api import call_llm

app = FastAPI()

index = faiss.read_index('faiss_index.bin')
with open('chunks.txt', 'r') as f:
    chunks = [line.strip() for line in f]

def get_relevant_chunks(question, top_k=3):
    question_embedding = embed_text(question).reshape(1, -1)
    D, I = index.search(question_embedding, top_k)
    return [chunks[i] for i in I[0]]

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data['question']
    relevant_chunks = get_relevant_chunks(question)
    
    prompt = "Answer the question based on the following context:\n"
    prompt += "\n\n".join(relevant_chunks)
    prompt += f"\n\nQuestion: {question}\nAnswer:"

    answer = call_llm(prompt)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
