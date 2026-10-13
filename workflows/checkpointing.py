"""Checkpointer factories: PostgreSQL is required for durable deployments."""

import os


def postgres_checkpointer(database_url=None):
    """Create and initialize a durable PostgreSQL saver.

    The caller owns the returned saver/connection lifetime. Set
    ``LANGGRAPH_STRICT_MSGPACK=true`` in deployment to restrict deserialization.
    """
    from psycopg.rows import dict_row
    from psycopg import connect
    from langgraph.checkpoint.postgres import PostgresSaver

    url = database_url or os.getenv("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL is required for durable workflow checkpoints")
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    # SQLAlchemy URLs use a driver suffix that psycopg itself does not accept.
    dsn = url.replace("postgresql+psycopg://", "postgresql://", 1)
    connection = connect(dsn, autocommit=True, row_factory=dict_row)
    saver = PostgresSaver(connection)
    saver.setup()
    return saver
