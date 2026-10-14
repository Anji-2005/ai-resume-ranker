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


def _psycopg_dsn(url):
    # SQLAlchemy URLs use a driver suffix that psycopg itself does not accept.
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


_POOLED_SAVERS = {}


def pooled_postgres_checkpointer(database_url=None, max_size=5):
    """Return a process-wide durable saver backed by a bounded connection pool.

    ``setup()`` runs once per database URL. State lives only in PostgreSQL, so a
    restarted process resumes any interrupted thread from its last checkpoint.
    """
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver

    url = database_url or os.getenv("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL is required for durable workflow checkpoints")
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    saver = _POOLED_SAVERS.get(url)
    if saver is None:
        pool = ConnectionPool(
            _psycopg_dsn(url), min_size=1, max_size=max_size, open=True, timeout=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        saver = PostgresSaver(pool)
        saver.setup()
        _POOLED_SAVERS[url] = saver
    return saver


def close_pooled_checkpointers():
    """Close pooled connections (tests and graceful shutdown)."""
    while _POOLED_SAVERS:
        _, saver = _POOLED_SAVERS.popitem()
        saver.conn.close()
