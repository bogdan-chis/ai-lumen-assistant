from sentence_transformers import SentenceTransformer, util

_model = SentenceTransformer("all-MiniLM-L6-v2")
_emb_cache = {}

def embed(text):
    if text in _emb_cache: return _emb_cache[text]
    _emb_cache[text] = _model.encode(text, normalize_embeddings=True)
    return _emb_cache[text]

def retrieve(query, corpus_texts, k=5):
    if not corpus_texts: return []
    q = embed(query)
    cs = [embed(t) for t in corpus_texts]
    sims = util.cos_sim(q, cs).tolist()[0]
    ranked = sorted(zip(corpus_texts, sims), key=lambda x: x[1], reverse=True)
    return [t for t,_ in ranked[:k]]
