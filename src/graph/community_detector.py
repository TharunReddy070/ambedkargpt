import community.community_louvain as louvain
import networkx as nx


class CommunityDetector:
    def detect(self, graph: nx.Graph) -> dict[int, list[str]]:
        partition = louvain.best_partition(graph)
        communities = {}

        for node, cid in partition.items():
            communities.setdefault(cid, []).append(node)

        return communities
