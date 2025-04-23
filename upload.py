import json
from dependencies import generate_embeddings, upsert_documents

with open("Combined_Insurance_Products.json", "r") as f:
    product_data = json.load(f)

pinecone_vectors = []

for i, item in enumerate(product_data, start=1):
    description = item.get("Description", "")
    product_id = f"Product_{i}"

    vector = generate_embeddings(description)
    if vector:
        pinecone_vectors.append({
            "id": product_id,
            "values": vector,
            "metadata": {
                k: v for k, v in item.items() if k != "Description"
            }
        })

upsert_documents(pinecone_vectors)