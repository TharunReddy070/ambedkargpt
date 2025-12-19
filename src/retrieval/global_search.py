from sklearn.metrics.pairwise import cosine_similarity


class GlobalGraphSearch:
    def __init__(self, embedder):
        self.embedder = embedder

    def search(
        self,
        query: str,
        community_summaries: dict,
        top_k: int = 3,
    ) -> list[int]:
        query_emb = self.embedder.encode(query)
        scores = []

        for cid, summary in community_summaries.items():
            summary_emb = self.embedder.encode(summary)
            score = cosine_similarity([query_emb], [summary_emb])[0][0]
            scores.append((cid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scores[:top_k]]
