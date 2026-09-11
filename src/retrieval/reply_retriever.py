"""
Retrieves similar historical customer complaints and their resolved AppleSupport replies, 
to ground the reply generator (RAG).

Uses TF-IDF cosine similarity over customer_text — simple, no API cost,
good enough for retrieval (unlike generation, which needs a real LLM).

Usage:
    python -m src.retrieval.reply_retriever --query "my battery is draining fast"
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "data" / "processed" / "retrieval_corpus.csv"


class ReplyRetriever:
    def __init__(self, corpus_path: Path):
        self.df = pd.read_csv(corpus_path)
        self.df = self.df.dropna(subset=["customer_text", "apple_reply_text"])
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.df["customer_text"])

    def retrieve(self, query: str, k: int = 3):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for idx in top_idx:
            results.append({
                "similarity": float(sims[idx]),
                "historical_customer_text": self.df.iloc[idx]["customer_text"],
                "historical_reply": self.df.iloc[idx]["apple_reply_text"],
            })
        return results


def main(args):
    retriever = ReplyRetriever(Path(args.corpus))
    results = retriever.retrieve(args.query, k=args.k)
    for r in results:
        print(f"\n[sim={r['similarity']:.3f}]")
        print(f"  historical customer msg: {r['historical_customer_text'][:150]}")
        print(f"  historical reply:        {r['historical_reply'][:150]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=3)
    main(parser.parse_args())