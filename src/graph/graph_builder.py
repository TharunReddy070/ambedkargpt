"""
Knowledge Graph Builder with relationship extraction.

Builds graph structure with:
- Nodes: Entities extracted from text
- Edges: Relationships extracted via dependency parsing
"""
import networkx as nx
import spacy


class KnowledgeGraphBuilder:
    """
    Build knowledge graph with entity relationships.
    
    Uses dependency parsing to extract meaningful relationships
    between entities, not just co-occurrence.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.nlp = spacy.load("en_core_web_sm")
        self.chunks_text = {}  # Store chunk text for relationship extraction
    
    def _extract_relationships(self, text: str, entities: list[str]) -> list[tuple]:
        """
        Extract relationships between entities using dependency parsing.
        
        Looks for syntactic relationships:
        - Subject-Verb-Object patterns
        - Prepositional relationships
        - Conjunctions
        
        Args:
            text: Chunk text
            entities: List of entities in chunk
            
        Returns:
            List of (entity1, entity2, relation_type) tuples
        """
        doc = self.nlp(text)
        relationships = []
        
        # Map entities to their spans in doc
        entity_spans = {}
        for ent in doc.ents:
            if ent.text in entities:
                entity_spans[ent.text] = ent
        
        # Extract relationships via dependency parsing
        for token in doc:
            # Subject-Object relationships
            if token.dep_ in ('nsubj', 'nsubjpass'):
                subject = token.text
                # Find object
                for child in token.head.children:
                    if child.dep_ in ('dobj', 'pobj', 'attr'):
                        obj = child.text
                        # Check if both are entities
                        if subject in entities and obj in entities:
                            relation = token.head.lemma_  # Verb as relation
                            relationships.append((subject, obj, relation))
            
            # Prepositional relationships
            if token.dep_ == 'prep':
                # Entity -> prep -> Entity
                head_ent = None
                obj_ent = None
                
                # Check if head is entity
                for ent_text in entities:
                    if ent_text in token.head.text:
                        head_ent = ent_text
                        break
                
                # Check if object of prep is entity
                for child in token.children:
                    if child.dep_ == 'pobj':
                        for ent_text in entities:
                            if ent_text in child.text:
                                obj_ent = ent_text
                                break
                
                if head_ent and obj_ent:
                    relationships.append((head_ent, obj_ent, token.text))
        
        # Fallback: Co-occurrence for entities without explicit relationships
        for i, e1 in enumerate(entities):
            has_relation = any(e1 in (r[0], r[1]) for r in relationships)
            if not has_relation:
                for e2 in entities[i+1:]:
                    relationships.append((e1, e2, 'co-occurs'))
        
        return relationships

    def add_chunk(self, chunk_id: int, entities: list[str], text: str = ""):
        """
        Add chunk entities and extract relationships.
        
        Args:
            chunk_id: Chunk identifier
            entities: List of entities in chunk
            text: Full chunk text for relationship extraction
        """
        # Add entity nodes
        for entity in entities:
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, mentions=1)
            else:
                self.graph.nodes[entity]['mentions'] += 1
        
        # Extract and add relationships
        if text and len(entities) > 1:
            relationships = self._extract_relationships(text, entities)
            
            for e1, e2, rel_type in relationships:
                if self.graph.has_edge(e1, e2):
                    # Strengthen existing edge
                    self.graph[e1][e2]['weight'] = self.graph[e1][e2].get('weight', 1) + 1
                    self.graph[e1][e2]['relations'].add(rel_type)
                else:
                    # Create new edge
                    self.graph.add_edge(e1, e2, 
                                       weight=1, 
                                       chunk=chunk_id,
                                       relations={rel_type})

    def get_graph(self) -> nx.Graph:
        """Return constructed knowledge graph."""
        return self.graph
