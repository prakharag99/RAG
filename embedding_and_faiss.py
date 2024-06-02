from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

if __name__ == "__main__":
    with open('luke_skywalker.txt', 'r') as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = np.vstack([embed_text(chunk) for chunk in chunks])

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, 'faiss_index.bin')
    with open('chunks.txt', 'w') as f:
        for chunk in chunks:
            f.write("%s\n" % chunk)
