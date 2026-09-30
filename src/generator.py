"""LLM answer generation with citations, streaming, and conversation memory."""
from typing import List, Dict, Iterator, Optional
from google import genai
from groq import Groq

from .utils import get_api_key


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Answer only using information from the context below.
2. If the answer is not in the context, say: "I couldn't find this in the provided documents."
3. Always cite your sources using the format [Source: filename, Page: X].
4. Be concise and accurate.
5. Do not make up information.
6. If the user asks a follow-up question, use the conversation history for context but still ground your answer in the provided document context.

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


def format_history(history: List[Dict], max_turns: int = 4) -> str:
    """Format last N conversation turns into a compact string."""
    if not history:
        return ""

    recent = history[-max_turns * 2:]  # user + assistant pairs
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_prompt(query: str, hits: List[Dict], history: Optional[List[Dict]] = None) -> str:
    """Build the full prompt with context and optional history."""
    context = format_context(hits)
    base = SYSTEM_PROMPT.format(context=context)

    if history:
        hist_str = format_history(history)
        if hist_str:
            base += f"\n\nConversation so far:\n{hist_str}"

    base += f"\n\nQuestion: {query}\n\nAnswer:"
    return base


# ═══════════════════════════════════════════════════════════════════
#  GEMINI
# ═══════════════════════════════════════════════════════════════════

def answer_with_gemini(
    query: str,
    hits: List[Dict],
    history: Optional[List[Dict]] = None,
    model: str = "gemini-3.6-flash",
) -> str:
    """Generate answer using Google Gemini (non-streaming)."""
    client = genai.Client(api_key=get_api_key("gemini"))
    prompt = _build_prompt(query, hits, history)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def stream_with_gemini(
    query: str,
    hits: List[Dict],
    history: Optional[List[Dict]] = None,
    model: str = "gemini-3.6-flash",
) -> Iterator[str]:
    """Stream answer from Google Gemini."""
    client = genai.Client(api_key=get_api_key("gemini"))
    prompt = _build_prompt(query, hits, history)

    for chunk in client.models.generate_content_stream(model=model, contents=prompt):
        if chunk.text:
            yield chunk.text


# ═══════════════════════════════════════════════════════════════════
#  GROQ
# ═══════════════════════════════════════════════════════════════════

def _build_groq_messages(query: str, hits: List[Dict], history: Optional[List[Dict]]) -> List[Dict]:
    context = format_context(hits)
    system_msg = SYSTEM_PROMPT.format(context=context)
    if history:
        hist_str = format_history(history)
        if hist_str:
            system_msg += f"\n\nConversation so far:\n{hist_str}"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ]


def answer_with_groq(
    query: str,
    hits: List[Dict],
    history: Optional[List[Dict]] = None,
    model: str = "openai/gpt-oss-120b",
) -> str:
    """Generate answer using Groq (non-streaming)."""
    client = Groq(api_key=get_api_key("groq"))
    messages = _build_groq_messages(query, hits, history)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


def stream_with_groq(
    query: str,
    hits: List[Dict],
    history: Optional[List[Dict]] = None,
    model: str = "openai/gpt-oss-120b",
) -> Iterator[str]:
    """Stream answer from Groq."""
    client = Groq(api_key=get_api_key("groq"))
    messages = _build_groq_messages(query, hits, history)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ═══════════════════════════════════════════════════════════════════
#  ROUTERS
# ═══════════════════════════════════════════════════════════════════

def generate_answer(
    query: str,
    hits: List[Dict],
    provider: str = "gemini",
    history: Optional[List[Dict]] = None,
) -> str:
    """Route to the correct LLM provider (non-streaming)."""
    if not hits:
        return "I couldn't find any relevant content in the documents to answer this question."

    provider = provider.lower()
    if provider == "gemini":
        return answer_with_gemini(query, hits, history)
    elif provider == "groq":
        return answer_with_groq(query, hits, history)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def stream_answer(
    query: str,
    hits: List[Dict],
    provider: str = "gemini",
    history: Optional[List[Dict]] = None,
) -> Iterator[str]:
    """Route to the correct LLM provider (streaming)."""
    if not hits:
        yield "I couldn't find any relevant content in the documents to answer this question."
        return

    provider = provider.lower()
    if provider == "gemini":
        yield from stream_with_gemini(query, hits, history)
    elif provider == "groq":
        yield from stream_with_groq(query, hits, history)
    else:
        raise ValueError(f"Unknown provider: {provider}")