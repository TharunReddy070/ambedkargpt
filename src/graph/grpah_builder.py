import networkx as nx


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def add_chunk(self, chunk_id: int, entities: list[str]):
        for entity in entities:
            self.graph.add_node(entity)

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                self.graph.add_edge(entities[i], entities[j], chunk=chunk_id)

    def get_graph(self) -> nx.Graph:
        return self.graph
