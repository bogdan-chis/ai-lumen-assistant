import os
from openai import OpenAI

# Defaults for local LM Studio server
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
API_KEY  = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL    = os.getenv("OPENAI_MODEL", "rollama3-8b-instruct-imat")

_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """
Ești un asistent pentru persoane cu deficiențe de vedere.
Primești ca intrare fragmente de text extrase din mediul înconjurător, uneori cu greșeli sau incomplete.
Scopul tău este să construiești o propoziție scurtă care descrie cât mai clar și util ce se află în fața utilizatorului.

REGULI:
- Folosește DOAR textul furnizat. Nu inventa informații lipsă.
- Fii concis și ușor de rostit. Limita este de 140 de caractere, dacă nu se specifică altfel.
- Prioritizează informațiile utile pentru un utilizator nevăzător, în această ordine:
  1) Avertismente, pericole, direcții, numere de autobuz/tren
  2) Denumiri de magazine, titluri de pagini, firme sau etichete vizibile
  3) Prețuri, totaluri, date, ore, termene, numere de telefon
  4) Orar, intervale orare, indicații de orientare

- Elimină duplicatele, caracterele inutile și suprapunerile parțiale.
- Normalizează majusculele, spațiile și semnele de punctuație.
- Extinde abrevierile comune doar dacă sensul este clar (ex: „L-V” → „Luni-Vineri”).
- Dacă nu există nimic relevant sau inteligibil: răspunde exact cu „Nimic citibil.”

FORMAT:
- O singură propoziție, fără punct la final, fără ghilimele, fără emoji.
- Răspunsul trebuie să fie clar, informativ și rostit natural.
"""

def _build_prompt(context_snippets, query):
    context = "\n".join(f"- {c}" for c in context_snippets[:8])
    return f"Context:\n{context}\n\nÎntrebare: {query}\nRăspuns:"

def answer(query, context_snippets):
    if not context_snippets:
        return "Nimic citibil aici."
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
