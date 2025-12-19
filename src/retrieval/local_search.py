from sklearn.metrics.pairwise import cosine_similarity


class LocalGraphSearch:
    def __init__(self, embedder):
        self.embedder = embedder

    def search(
        self,
        query: str,
        entity_embeddings: dict,
        chunk_embeddings: dict,
        tau_e: float = 0.3,
        tau_d: float = 0.3,
        top_k: int = 5,
    ) -> list[int]:
        query_emb = self.embedder.encode(query)
        results = []

        for entity, entity_emb in entity_embeddings.items():
            if cosine_similarity([query_emb], [entity_emb])[0][0] > tau_e:
                for cid, chunk_emb in chunk_embeddings.items():
                    if cosine_similarity([entity_emb], [chunk_emb])[0][0] > tau_d:
                        results.append(cid)

        return list(dict.fromkeys(results))[:top_k]
