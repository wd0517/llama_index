"""TiDB property graph store index."""
import json
from typing import Tuple, Optional, List, Dict, Any
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    TEXT,
    ForeignKey,
    sql,
)
from sqlalchemy.orm import (
    Session,
    declarative_base,
    relationship,
)

from tidb_vector.sqlalchemy import VectorType
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    LabelledNode,
    EntityNode,
    ChunkNode,
    Relation,
    Triplet,
    VectorStoreQuery,
)
from llama_index.graph_stores.tidb.utils import (
    check_db_availability,
    remove_empty_values,
    get_or_create,
)


rel_depth_query = """
WITH RECURSIVE PATH AS
  (SELECT 1 AS depth,
          r.source_id,
          r.target_id,
          r.label,
          r.properties
   FROM {relation_table} r
   WHERE r.source_id IN :ids
   UNION ALL SELECT p.depth + 1,
                    r.source_id,
                    r.target_id,
                    r.label,
                    r.properties
   FROM PATH p
   JOIN {relation_table} r ON p.target_id = r.source_id
   WHERE p.depth < :depth )
SELECT e1.id AS e1_id,
       e1.name AS e1_name,
       e1.label AS e1_label,
       e1.properties AS e1_properties,
       p.label AS rel_label,
       p.properties AS rel_properties,
       e2.id AS e2_id,
       e2.name AS e2_name,
       e2.label AS e2_label,
       e2.properties AS e2_properties
FROM PATH p
JOIN {node_table} e1 ON p.source_id = e1.id
JOIN {node_table} e2 ON p.target_id = e2.id
ORDER BY p.depth
LIMIT :limit;
"""


class TiDBPropertyGraphStore(PropertyGraphStore):
    supports_structured_queries: bool = False
    supports_vector_queries: bool = True

    def __init__(
        self,
        db_connection_string: str,
    ) -> None:
        # TiDB Serverless clusters have a limitation: if there are no active connections for 5 minutes,
        # they will shut down, which closes all connections, so we need to recycle the connections
        self._engine = create_engine(db_connection_string, pool_recycle=300)
        check_db_availability(self._engine, check_vector=True)

        self._node_model, self._relation_model = self.init_schema()

    def init_schema(self) -> Tuple:
        """Initialize schema."""
        Base = declarative_base()

        class BaseMixin:
            created_at = Column(DateTime, nullable=False, server_default=sql.func.now())
            updated_at = Column(
                DateTime,
                nullable=False,
                server_default=sql.func.now(),
                onupdate=sql.func.now(),
            )

        class NodeModel(BaseMixin, Base):
            __tablename__ = "pkg_nodes"
            id = Column(String(512), primary_key=True)
            text = Column(TEXT, nullable=True)
            name = Column(String(512), nullable=True)
            label = Column(String(512), nullable=False, default="node")
            properties = Column(JSON, default={})
            embedding = Column(VectorType(1536), comment="hnsw(distance=cosine)")

        class RelationModel(BaseMixin, Base):
            __tablename__ = "pkg_relations"
            id = Column(Integer, primary_key=True)
            label = Column(String(512), nullable=False)
            source_id = Column(String(512), ForeignKey("pkg_nodes.id"))
            target_id = Column(String(512), ForeignKey("pkg_nodes.id"))
            properties = Column(JSON, default={})

            source = relationship("NodeModel", foreign_keys=[source_id])
            target = relationship("NodeModel", foreign_keys=[target_id])

        Base.metadata.create_all(self._engine)
        return NodeModel, RelationModel

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        print("-" * 50 + "get" + "-" * 50)
        print(properties)
        print(ids)
        print("-" * 50)
        with Session(self._engine) as session:
            query = session.query(self._node_model)
            if properties:
                for key, value in properties.items():
                    query = query.filter(
                        self._node_model.properties[key].astext == value
                    )
            if ids:
                query = query.filter(self._node_model.id.in_(ids))

            nodes = []
            for n in query.all():
                if n.label == "text_chunk":
                    nodes.append(
                        ChunkNode(
                            id=n.id,
                            text=n.text,
                            label=n.label,
                            properties=remove_empty_values(n.properties),
                        )
                    )
                else:
                    nodes.append(
                        EntityNode(
                            name=n.name,
                            label=n.label,
                            properties=remove_empty_values(n.properties),
                        )
                    )
            return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets."""
        # if nothing is passed, return empty list
        if not ids and not properties and not entity_names and not relation_names:
            return []

        with Session(self._engine) as session:
            query = session.query(self._relation_model)
            if ids:
                query = query.filter(
                    self._relation_model.source_id.in_(ids)
                    | self._relation_model.target_id.in_(ids)
                )
            if properties:
                for key, value in properties.items():
                    query = query.filter(
                        self._relation_model.properties[key].astext
                        == value | self._relation_model.source.properties[key].astext
                        == value | self._relation_model.target.properties[key].astext
                        == value
                    )
            if entity_names:
                query = query.filter(
                    self._relation_model.source.name.in_(entity_names)
                    | self._relation_model.target.name.in_(entity_names)
                )
            if relation_names:
                query = query.filter(self._relation_model.label.in_(relation_names))

            triplets = []
            for r in query.all():
                source = EntityNode(
                    name=r.source.name,
                    label=r.source.label,
                    properties=remove_empty_values(r.source.properties),
                )
                target = EntityNode(
                    name=r.target.name,
                    label=r.target.label,
                    properties=remove_empty_values(r.target.properties),
                )
                relation = Relation(
                    label=r.label,
                    source_id=source.id,
                    target_id=target.id,
                    properties=remove_empty_values(r.properties),
                )
                triplets.append([source, relation, target])
            print("-" * 50 + "get_triplets" + "-" * 50)
            print(entity_names, relation_names, properties, ids)
            print(triplets)
            print("-" * 50)
            return triplets

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        triplets = []
        ids = [node.id for node in graph_nodes]

        if not ids:
            return []

        with Session(self._engine) as session:
            result = session.execute(
                sql.text(
                    rel_depth_query.format(
                        relation_table="pkg_relations",
                        node_table="pkg_nodes",
                    )
                ),
                {
                    "ids": ids,
                    "depth": depth,
                    "limit": limit,
                },
            )

            keys = result.keys()
            raw_rels = [dict(zip(keys, row)) for row in result.fetchall()]

            ignore_rels = ignore_rels or []
            for row in raw_rels:
                if row["rel_label"] in ignore_rels:
                    continue

                source = EntityNode(
                    id=row["e1_id"],
                    name=row["e1_name"],
                    label=row["e1_label"],
                    properties=json.loads(row["e1_properties"]),
                )
                target = EntityNode(
                    id=row["e2_id"],
                    name=row["e2_name"],
                    label=row["e2_label"],
                    properties=json.loads(row["e2_properties"]),
                )
                relation = Relation(
                    label=row["rel_label"],
                    source_id=source.id,
                    target_id=target.id,
                    properties=json.loads(row["rel_properties"]),
                )
                triplets.append([source, relation, target])
        print("-" * 50 + "get_rel_map" + "-" * 50)
        print(graph_nodes, depth, limit, ignore_rels)
        print(triplets)
        print("-" * 50)
        return triplets

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        """Upsert nodes."""
        entity_list: List[EntityNode] = []
        chunk_list: List[ChunkNode] = []
        other_list: List[LabelledNode] = []

        for item in nodes:
            if isinstance(item, EntityNode):
                entity_list.append(item)
            elif isinstance(item, ChunkNode):
                chunk_list.append(item)
            else:
                other_list.append(item)
        print("-" * 50 + "upsert_nodes" + "-" * 50)
        print(entity_list)
        print(chunk_list)
        print(other_list)
        print("-" * 50)

        with Session(self._engine) as session:
            # TODO: use upsert instead of get_or_create
            for entity in entity_list:
                print(entity)
                entity_instance, _ = get_or_create(
                    session, self._node_model, id=entity.id
                )
                entity_instance.name = entity.name
                entity_instance.label = entity.label
                entity_instance.properties = entity.properties
                entity_instance.embedding = entity.embedding
                session.add(entity_instance)

            for chunk in chunk_list:
                print(chunk)
                chunk_instance, _ = get_or_create(
                    session, self._node_model, id=chunk.id
                )
                chunk_instance.text = chunk.text
                chunk_instance.label = chunk.label
                chunk_instance.properties = chunk.properties
                chunk_instance.embedding = chunk.embedding
                session.add(chunk_instance)
            session.commit()

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations."""
        print("-" * 50 + "upsert_relations" + "-" * 50)
        print(relations)
        print("-" * 50)
        with Session(self._engine) as session:
            for r in relations:
                get_or_create(
                    session,
                    self._node_model,
                    id=r.source_id,
                )
                get_or_create(
                    session,
                    self._node_model,
                    id=r.target_id,
                )
                relation_instance, _ = get_or_create(
                    session,
                    self._relation_model,
                    label=r.label,
                    source_id=r.source_id,
                    target_id=r.target_id,
                )
                relation_instance.properties = r.properties
                session.add(relation_instance)
                session.commit()

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        print("-" * 50 + "delete" + "-" * 50)
        print(entity_names, relation_names, properties, ids)
        print("-" * 50)
        # TODO: implement

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Query the graph store with statement and parameters."""
        raise NotImplementedError(
            "Structured queries are not supported by TiDB property graph store."
        )

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""
        with Session(self._engine) as session:
            result = (
                session.query(
                    self._node_model,
                    self._node_model.embedding.cosine_distance(
                        query.query_embedding
                    ).label("embedding_distance"),
                )
                .filter(self._node_model.name.is_not(None))
                .order_by(sql.asc("embedding_distance"))
                .limit(query.similarity_top_k)
                .all()
            )

            nodes = []
            scores = []
            for node, score in result:
                nodes.append(
                    EntityNode(
                        name=node.name,
                        label=node.label,
                        properties=remove_empty_values(node.properties),
                    )
                )
                scores.append(score)
            print("-" * 50 + "vector_query" + "-" * 50)
            print(query)
            print(nodes)
            print("-" * 50)
            return nodes, scores
