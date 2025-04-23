from dependencies import generate_embeddings, upsert_documents
from pinecone import Pinecone
import openai
import json
from config import PRODUCT_INDEX_NAME, REVIEW_INDEX_NAME, PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
products_index = pc.Index(PRODUCT_INDEX_NAME)
reviews_index = pc.Index(REVIEW_INDEX_NAME)

query = "What does AIA Life Insurance cover?"
embedding_ada = generate_embeddings(query, model="text-embedding-ada-002")
embedding_babbage = generate_embeddings(
    query,
    model="text-embedding-3-small",
    dimensions=768
)

product_matches = products_index.query(vector=embedding_ada, top_k=5, include_metadata=True)
review_matches = reviews_index.query(vector=embedding_babbage, top_k=5, include_metadata=True)

#Format contexts
def format_context(matches):
    return "\n\n".join([match.metadata.get("Description", str(match.metadata)) for match in matches])

product_context = format_context(product_matches["matches"])
review_context = format_context(review_matches["matches"])

#Prompting Strategies

#Stuffing
prompt_stuffing = f"""
You are an insurance assistant. Based on the information below, answer the question clearly and helpfully.

Context:
{product_context}
{review_context}

Question: {query}
"""

#Chain-of-Thought
prompt_cot = f"""
You are an expert insurance assistant. Think step-by-step and explain clearly.

Context:
{product_context}

Question: {query}
"""

#ReAct
prompt_react = f"""
You are a helpful agent. Think step-by-step and decide if you need to act (search or calculate).

Context:
{product_context}
{review_context}

Question: {query}
"""

#Map-Reduce (we simulate multiple prompts)
prompt_map_reduce = [
    f"What is the payout for AIA Crisis Recovery?\nContext:\n{product_context}",
    f"What conditions trigger AIA Crisis Recovery?\nContext:\n{product_context}",
    f"Does the review mention Crisis Recovery?\nContext:\n{review_context}"
]

#ReAct + CoT
prompt_react_cot = f"""
You are a professional assistant. Start by thinking step-by-step. Then explain your answer with reasoning and take action if needed.

Context:
{product_context}
{review_context}

Question: {query}
"""

#Stuffing + Summarisation
prompt_stuff_summarise = f"""
You are an expert assistant. Read all the context and provide a concise summary and answer.

Context:
{product_context}
{review_context}

Question: {query}
"""
def run_prompt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful insurance assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

print("\n--- Stuffing ---\n", run_prompt(prompt_stuffing))
print("\n--- Chain of Thought ---\n", run_prompt(prompt_cot))
print("\n--- ReAct ---\n", run_prompt(prompt_react))
print("\n--- ReAct + CoT ---\n", run_prompt(prompt_react_cot))
print("\n--- Stuffing + Summarisation ---\n", run_prompt(prompt_stuff_summarise))

map_reduce_results = [run_prompt(p) for p in prompt_map_reduce]
print("\n--- Map-Reduce ---\n", "\n---\n".join(map_reduce_results))