from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOC_DIR = Path(__file__).parent / "company_documents"

class PolicyRetriever:
    def __init__(self):
        self.documents = []
        for path in sorted(DOC_DIR.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            for part in content.split("\n## "):
                self.documents.append({"source": path.name, "text": part.strip()})
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(d["text"] for d in self.documents)

    def search(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        best = scores.argsort()[::-1][:k]
        return [{**self.documents[i], "score": float(scores[i])} for i in best if scores[i] > 0]

