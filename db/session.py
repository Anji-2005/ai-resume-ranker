"""Database configuration with no import-time connection side effects."""

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


def database_url() -> str:
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL must be configured for the document API")
    return value


@lru_cache(maxsize=1)
def get_session_factory():
    engine = create_engine(
        database_url(), pool_pre_ping=True,
        pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
        echo=os.getenv("DATABASE_ECHO", "false").casefold() == "true",
    )
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def reset_session_factory():
    """Test helper for changing DATABASE_URL in a process."""
    get_session_factory.cache_clear()
