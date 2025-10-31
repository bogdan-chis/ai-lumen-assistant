# No TF/Transformers. Pure scikit-learn.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self):
        self.texts = []
        self.vectorizer = None
        self.matrix = None

    def build(self, texts):
        # texts: list[str]
        self.texts = texts[:]
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1,2),
            max_features=20000
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def query(self, q, k=5):
        if not self.texts:
            return []
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.matrix).ravel()
        idx = sims.argsort()[::-1][:k]
        return [self.texts[i] for i in idx if sims[i] > 0]

# simple module-level instance
_retriever = TfidfRetriever()

def retrieve(query, corpus_texts, k=5):
    # rebuild if corpus changed size
    if _retriever.matrix is None or len(corpus_texts) != len(_retriever.texts):
        _retriever.build(corpus_texts)
    return _retriever.query(query, k=k)
