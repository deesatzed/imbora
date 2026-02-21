"""PostgreSQL database engine for The Associate.

Manages connections via psycopg3 and handles schema initialization.
All queries flow through this engine; the Repository class builds on top.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

import psycopg
from psycopg.rows import dict_row

from src.core.config import DatabaseConfig
from src.core.exceptions import ConnectionError, DatabaseError, SchemaInitError

logger = logging.getLogger("associate.db")

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_DATASCI_PATH = Path(__file__).parent / "schema_datasci.sql"


class DatabaseEngine:
    """PostgreSQL engine wrapping psycopg3.

    Usage:
        engine = DatabaseEngine(config)
        rows = engine.fetch_all("SELECT * FROM tasks WHERE status = %s", ["PENDING"])

        with engine.transaction() as cur:
            cur.execute("INSERT INTO tasks ...")
            cur.execute("UPDATE projects ...")
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._conn: Optional[psycopg.Connection] = None

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._connect()
        assert self._conn is not None
        return self._conn

    def _connect(self) -> None:
        try:
            self._conn = psycopg.connect(
                self.config.connection_string,
                row_factory=dict_row,
                autocommit=True,
            )
            logger.info("Connected to PostgreSQL at %s:%s/%s",
                        self.config.host, self.config.port, self.config.dbname)
        except psycopg.OperationalError as e:
            raise ConnectionError(f"Failed to connect to database: {e}") from e

    def initialize_schema(self) -> None:
        """Run schema.sql and schema_datasci.sql to create all tables and indexes."""
        if not SCHEMA_PATH.exists():
            raise SchemaInitError(f"Schema file not found: {SCHEMA_PATH}")

        sql = SCHEMA_PATH.read_text()
        try:
            self.conn.execute(sql)
            logger.info("Database schema initialized successfully")
        except psycopg.Error as e:
            raise SchemaInitError(f"Failed to initialize schema: {e}") from e

        if SCHEMA_DATASCI_PATH.exists():
            ds_sql = SCHEMA_DATASCI_PATH.read_text()
            try:
                self.conn.execute(ds_sql)
                logger.info("Data science schema extension initialized successfully")
            except psycopg.Error as e:
                raise SchemaInitError(f"Failed to initialize datasci schema: {e}") from e

    def execute(self, query: str, params: Optional[Sequence[Any]] = None) -> None:
        """Execute a query without returning results."""
        try:
            self.conn.execute(query, params)
        except psycopg.Error as e:
            raise DatabaseError(f"Query failed: {e}") from e

    def fetch_one(self, query: str, params: Optional[Sequence[Any]] = None) -> Optional[dict[str, Any]]:
        """Execute a query and return the first row as a dict, or None."""
        try:
            cur = self.conn.execute(query, params)
            return cur.fetchone()
        except psycopg.Error as e:
            raise DatabaseError(f"Query failed: {e}") from e

    def fetch_all(self, query: str, params: Optional[Sequence[Any]] = None) -> list[dict[str, Any]]:
        """Execute a query and return all rows as dicts."""
        try:
            cur = self.conn.execute(query, params)
            return cur.fetchall()
        except psycopg.Error as e:
            raise DatabaseError(f"Query failed: {e}") from e

    @contextmanager
    def transaction(self):
        """Context manager for explicit transactions.

        Usage:
            with engine.transaction() as cur:
                cur.execute("INSERT ...")
                cur.execute("UPDATE ...")
        """
        # Temporarily disable autocommit for the transaction
        old_autocommit = self.conn.autocommit
        self.conn.autocommit = False
        try:
            cur = self.conn.cursor(row_factory=dict_row)
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.autocommit = old_autocommit

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Database connection closed")

    def __del__(self):
        self.close()
