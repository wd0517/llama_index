"""TiDB graph store index."""
from typing import Tuple, Any, List, Optional, Dict
from collections import defaultdict
from sqlalchemy import (
    Engine,
    create_engine,
    exc,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
    sql,
    delete,
)
from sqlalchemy.orm import (
    Session,
    declarative_base,
    relationship,
    joinedload,
)

from llama_index.core.graph_stores.types import GraphStore


rel_depth_query = sql.text(
    """
WITH RECURSIVE PATH AS
  (SELECT 1 AS depth,
          r.subject_id,
          r.object_id,
          r.description
   FROM relationships r
   WHERE r.subject_id IN
       (SELECT id
        FROM entities
        WHERE name IN :subjs )
   UNION ALL SELECT p.depth + 1,
                    r.subject_id,
                    r.object_id,
                    r.description
   FROM PATH p
   JOIN relationships r ON p.object_id = r.subject_id
   WHERE p.depth < :depth )
SELECT p.depth,
       e1.name AS subject,
       p.description,
       e2.name AS object
FROM PATH p
JOIN entities e1 ON p.subject_id = e1.id
JOIN entities e2 ON p.object_id = e2.id
ORDER BY p.depth
LIMIT :limit;
"""
)


class TiDBGraphStore(GraphStore):
    def __init__(
        self,
        db_connection_string: str,
        entity_table_name: str = "entities",
        relationship_table_name: str = "relationships",
    ) -> None:
        # TiDB Serverless clusters have a limitation: if there are no active connections for 5 minutes,
        # they will shut down, which closes all connections, so we need to recycle the connections
        self._engine = create_engine(db_connection_string, pool_recycle=300)
        check_db_availability(self._engine)

        self._entity_table_name = entity_table_name
        self._relationship_table_name = relationship_table_name
        self._entity_model, self._rel_model = self.init_schema(
            entity_table_name, relationship_table_name
        )

    def init_schema(
        self, entity_table_name: str, relationship_table_name: str
    ) -> Tuple:
        Base = declarative_base()

        class EntityModel(Base):
            __tablename__ = entity_table_name

            id = Column(Integer, primary_key=True)
            name = Column(String(512), nullable=False)
            created_at = Column(DateTime, nullable=False, server_default=sql.func.now())
            updated_at = Column(
                DateTime,
                nullable=False,
                server_default=sql.func.now(),
                onupdate=sql.func.now(),
            )

        class RelationshipModel(Base):
            __tablename__ = relationship_table_name

            id = Column(Integer, primary_key=True)
            description = Column(Text, nullable=False)
            subject_id = Column(Integer, ForeignKey(f"{entity_table_name}.id"))
            object_id = Column(Integer, ForeignKey(f"{entity_table_name}.id"))
            created_at = Column(DateTime, nullable=False, server_default=sql.func.now())
            updated_at = Column(
                DateTime,
                nullable=False,
                server_default=sql.func.now(),
                onupdate=sql.func.now(),
            )

            subject = relationship("EntityModel", foreign_keys=[subject_id])
            object = relationship("EntityModel", foreign_keys=[object_id])

        Base.metadata.create_all(self._engine)

        return EntityModel, RelationshipModel

    @property
    def get_client(self) -> Any:
        return self._engine

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        with Session(self._engine) as session:
            subj_instance, _ = get_or_create(session, self._entity_model, name=subj)
            obj_instance, _ = get_or_create(session, self._entity_model, name=obj)
            get_or_create(
                session,
                self._rel_model,
                description=rel,
                subject=subj_instance,
                object=obj_instance,
            )

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        with Session(self._engine) as session:
            rels = (
                session.query(self._rel_model)
                .options(
                    joinedload(self._rel_model.subject),
                    joinedload(self._rel_model.object),
                )
                .filter(self._rel_model.subject.has(name=subj))
                .all()
            )
            return [[rel.description, rel.object.name] for rel in rels]

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""
        rel_map: Dict[str, List[List[str]]] = defaultdict(list)
        with Session(self._engine) as session:
            # `raw_rels`` is a list of tuples (depth, subject, description, object), ordered by depth
            # Example:
            # +-------+------------------+------------------+------------------+
            # | depth | subject          | description      | object           |
            # +-------+------------------+------------------+------------------+
            # |     1 | Software         | Mention in       | Footnotes        |
            # |     1 | Viaweb           | Started by       | Paul graham      |
            # |     2 | Paul graham      | Invited to       | Lisp conference  |
            # |     2 | Paul graham      | Coded            | Bel              |
            # +-------+------------------+------------------+------------------+
            raw_rels = session.execute(
                rel_depth_query,
                {
                    "subjs": subjs,
                    "depth": depth,
                    "limit": limit,
                },
            ).fetchall()
            # `obj_reverse_map` is a dict of sets, where the key is a tuple (object, depth)
            # and the value is a set of subjects that have the object at the previous depth
            obj_reverse_map = defaultdict(set)
            for depth, subj, rel, obj in raw_rels:
                if depth == 1:
                    rel_map[subj].append([subj, rel, obj])
                    obj_reverse_map[(obj, depth)].update([subj])
                else:
                    for _subj in obj_reverse_map[(subj, depth - 1)]:
                        rel_map[_subj].append([subj, rel, obj])
                        obj_reverse_map[(obj, depth)].update([_subj])
            return dict(rel_map)

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""
        with Session(self._engine) as session:
            stmt = delete(self._rel_model).where(
                self._rel_model.subject.has(name=subj),
                self._rel_model.description == rel,
                self._rel_model.object.has(name=obj),
            )
            result = session.execute(stmt)
            session.commit()
            # no rows affected, do not need to delete entities
            print(result.rowcount)
            if result.rowcount == 0:
                return

            def delete_entity(entity_name: str):
                stmt = delete(self._entity_model).where(
                    self._entity_model.name == entity_name
                )
                session.execute(stmt)
                session.commit()

            def entity_was_referenced(entity_name: str):
                return (
                    session.query(self._rel_model)
                    .filter(
                        self._rel_model.subject.has(name=entity_name)
                        | self._rel_model.object.has(name=entity_name)
                    )
                    .one_or_none()
                )

            if not entity_was_referenced(subj):
                delete_entity(subj)
            if not entity_was_referenced(obj):
                delete_entity(obj)

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """Query the graph store with statement and parameters."""
        with Session(self._engine) as session:
            return session.execute(query, param_map).fetchall()


def check_db_availability(engine: Engine) -> None:
    try:
        with engine.connect() as conn:
            # conn.execute(sql.text("""SELECT Vec_Dims("[1]");"""))
            conn.execute(sql.text("""SELECT 1;"""))
    except exc.DatabaseError as e:
        db_error_code = e.orig.args[0]
        if db_error_code == 1045:
            raise ValueError(
                "Could not connect to the TiDB server. "
                "Please check if the connection string is correct."
            ) from e
        # elif db_error_code == 1305:
        #     raise ValueError(
        #         "Please confirm if your TiDB supports vector search. "
        #         "You can check this by running the query `SELECT Vec_Dims('[1]')` in TiDB."
        #     ) from e
        else:
            raise ValueError(
                "An error occurred while checking the database availability."
            ) from e


def get_or_create(session, model, defaults=None, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False
    else:
        if defaults:
            kwargs.update(defaults)
        instance = model(**kwargs)
        session.add(instance)
        session.commit()
        return instance, True
