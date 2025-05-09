from dependencies import generate_embeddings, upsert_documents
from pinecone import Pinecone
import openai
import json
from config import PRODUCT_INDEX_NAME, REVIEW_INDEX_NAME, PINECONE_API_KEY
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

pc = Pinecone(api_key=PINECONE_API_KEY)
products_index = pc.Index(PRODUCT_INDEX_NAME)
reviews_index = pc.Index(REVIEW_INDEX_NAME)

query = "Should I buy AIA lift insurance? Can you research on some of it review?"
embedding_ada = generate_embeddings(query, model="text-embedding-ada-002")
embedding_babbage = generate_embeddings(query, model="text-embedding-3-small", dimensions=768)

product_matches = products_index.query(vector=embedding_ada, top_k=5, include_metadata=True)
review_matches = reviews_index.query(vector=embedding_babbage, top_k=5, include_metadata=True)

# Format contexts
def format_context(matches):
    return "\n\n".join([match.metadata.get("Description", str(match.metadata)) for match in matches])

product_context = format_context(product_matches["matches"])
review_context = format_context(review_matches["matches"])

# Relevance scoring
def relevance_score(similarity):
    if similarity >= 0.90:
        return 5
    elif similarity >= 0.75:
        return 4
    elif similarity >= 0.60:
        return 3
    elif similarity >= 0.40:
        return 2
    else:
        return 1

def get_similarity_score(text, query):
    text_embed = generate_embeddings(text)
    query_embed = generate_embeddings(query)
    similarity = cosine_similarity([text_embed], [query_embed])[0][0]
    return similarity, relevance_score(similarity)

product_sim, product_rel_score = get_similarity_score(product_context, query)
review_sim, review_rel_score = get_similarity_score(review_context, query)

print(f"\n--- Context Relevance Scores ---")
print(f"Product Context Similarity: {product_sim:.2f} → Score: {product_rel_score}/5")
print(f"Review Context Similarity: {review_sim:.2f} → Score: {review_rel_score}/5")

# Prompts
prompt_stuffing = f"""
You are an insurance assistant. Based on the information below, answer the question clearly and helpfully.

Context:
{product_context}
{review_context}

Question: {query}
"""

prompt_cot = f"""
You are an expert insurance assistant. Think step-by-step and explain clearly.

Context:
{product_context}

Question: {query}
"""

prompt_react = f"""
You are a helpful agent. Think step-by-step and decide if you need to act (search or calculate).

Context:
{product_context}
{review_context}

Question: {query}
"""

prompt_map_reduce = [
    f"Answer this question based on product data:\nContext:\n{product_context}\n\nQuestion: {query}",
    f"Answer this question based on review data:\nContext:\n{review_context}\n\nQuestion: {query}"
]


prompt_react_cot = f"""
You are a professional assistant. Start by thinking step-by-step. Then explain your answer with reasoning and take action if needed.

Context:
{product_context}
{review_context}

Question: {query}
"""

prompt_stuff_summarise = f"""
You are an expert assistant. Read all the context and provide a concise summary and answer.

Context:
{product_context}
{review_context}

Question: {query}
"""

prompt_no_rag = f"""
You are an expert insurance assistant.

Question: {query}
"""

def run_prompt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful insurance assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Print answers
print("\n--- Stuffing ---\n", run_prompt(prompt_stuffing))
print("\n--- Chain of Thought ---\n", run_prompt(prompt_cot))
print("\n--- ReAct ---\n", run_prompt(prompt_react))
print("\n--- ReAct + CoT ---\n", run_prompt(prompt_react_cot))
print("\n--- Stuffing + Summarisation ---\n", run_prompt(prompt_stuff_summarise))
print("\n--- No RAG (Direct LLM) ---\n", run_prompt(prompt_no_rag))

map_reduce_results = [run_prompt(p) for p in prompt_map_reduce]
combined_answer = "\n".join(map_reduce_results)
print("\n--- Map-Reduce (Combined Result) ---\n", combined_answer)