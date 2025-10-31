import os
from openai import OpenAI

# LM Studio defaults: http://localhost:1234/v1 with any non-empty API key.
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
API_KEY  = os.getenv("OPENAI_API_KEY", "lm-studio")  # placeholder string is fine
MODEL    = os.getenv("OPENAI_MODEL", "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")  # set to the model name shown in LM Studio

_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = (
    "Answer only with facts found in the provided context. "
    "If the answer is not present, reply: 'Not in view.'"
)

def _build_prompt(context_snippets, query):
    context = "\n".join(f"- {c}" for c in context_snippets[:8])
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

def answer(query, context_snippets):
    if not context_snippets:
        return "Not in view."
    prompt = _build_prompt(context_snippets, query)

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=128,
    )
    return resp.choices[0].message.content.strip()