"""SQLite configuration and UTC-aware persistence, with no import-time connection.

SERVICEPILOT_DB_PATH selects the database file. Relative paths are resolved from
the Agent project root, never from the process working directory. Callers must
explicitly create the schema; this module does not run migrations or create tables.
"""

from datetime import datetime, timezone
import os
from pathlib import Path

from sqlalchemy import DateTime, Engine, URL, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator[datetime]):
    """Accept aware datetimes only; store naive UTC and restore aware UTC.

    SQLite has no timezone-preserving datetime type. Raw SQL must also respect
    this UTC storage contract; timezone normalization applies to typed ORM/Core IO.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError("Datetime must include an explicit timezone")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


def get_database_path(database_path: str | Path | None = None) -> Path:
    configured = database_path if database_path is not None else os.getenv(
        "SERVICEPILOT_DB_PATH", "servicepilot.sqlite3"
    )
    if not str(configured).strip():
        raise ValueError("Database path must not be empty")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        raise ValueError("Database file must use .db, .sqlite or .sqlite3")
    return path


def create_sqlite_engine(database_path: str | Path | None = None) -> Engine:
    path = get_database_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        URL.create("sqlite+pysqlite", database=str(path)),
        connect_args={"check_same_thread": False, "timeout": 5.0},
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        # PRAGMA foreign_keys must run outside a transaction on every connection.
        previous_autocommit = dbapi_connection.autocommit
        try:
            dbapi_connection.autocommit = True
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()
        finally:
            dbapi_connection.autocommit = previous_autocommit

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Callers own session lifetime and commit/rollback via Session.begin()."""
    return sessionmaker(bind=engine, expire_on_commit=True)
