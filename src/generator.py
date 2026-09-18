"""LLM answer generation with citations."""
from typing import List, Dict
import google.generativeai as genai
from groq import Groq

from .utils import get_api_key


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Answer only using information from the context below.
2. If the answer is not in the context, say: "I couldn't find this in the provided documents."
3. Always cite your sources using the format [Source: filename, Page: X].
4. Be concise and accurate.
5. Do not make up information.

Context:
{context}
"""


def format_context(hits: List[Dict]) -> str:
    """Format retrieved chunks into a context string."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[{i}] Source: {hit['source']}, Page: {hit['page']}\n{hit['text']}"
        )
    return "\n\n---\n\n".join(parts)


def answer_with_gemini(query: str, hits: List[Dict], model: str = "gemini-2.0-flash") -> str:
    """Generate answer using Google Gemini."""
    genai.configure(api_key=get_api_key("gemini"))
    context = format_context(hits)

    model_obj = genai.GenerativeModel(model)
    prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nQuestion: {query}\n\nAnswer:"
    response = model_obj.generate_content(prompt)
    return response.text


def answer_with_groq(query: str, hits: List[Dict], model: str = "llama-3.3-70b-versatile") -> str:
    """Generate answer using Groq (Llama)."""
    client = Groq(api_key=get_api_key("groq"))
    context = format_context(hits)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def generate_answer(query: str, hits: List[Dict], provider: str = "gemini") -> str:
    """Route to the correct LLM provider."""
    if not hits:
        return "I couldn't find any relevant content in the documents to answer this question."

    provider = provider.lower()
    if provider == "gemini":
        return answer_with_gemini(query, hits)
    elif provider == "groq":
        return answer_with_groq(query, hits)
    else:
        raise ValueError(f"Unknown provider: {provider}")