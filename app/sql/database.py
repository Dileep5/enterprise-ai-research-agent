"""Read-only SQL execution against the retail PostgreSQL database."""

from __future__ import annotations

import datetime as dt
import decimal
import logging
from dataclasses import dataclass, field

import psycopg
from psycopg import sql as pgsql

from app.errors import DatabaseUnavailableError, SQLExecutionError
from app.sql.guard import ALLOWED_TABLES, validate_sql

logger = logging.getLogger(__name__)


@dataclass
class SQLResult:
    query: str
    columns: list[str]
    rows: list[list] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
        }

    def to_markdown(self, max_rows: int = 30) -> str:
        if not self.columns:
            return "(no columns)"
        lines = ["| " + " | ".join(self.columns) + " |", "|" + "---|" * len(self.columns)]
        for row in self.rows[:max_rows]:
            lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
        if self.row_count > max_rows or self.truncated:
            lines.append(f"... ({self.row_count} rows shown{', truncated' if self.truncated else ''})")
        return "\n".join(lines)


def _jsonable(value):
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


class RetailDatabase:
    """Executes validated SELECT statements inside READ ONLY transactions."""

    def __init__(self, dsn: str | None, *, row_limit: int = 200, statement_timeout_ms: int = 5000,
                 connect_timeout_s: int = 5):
        self.dsn = dsn
        self.row_limit = row_limit
        self.statement_timeout_ms = statement_timeout_ms
        self.connect_timeout_s = connect_timeout_s
        self._schema_cache: str | None = None

    def _connect(self) -> psycopg.Connection:
        if not self.dsn:
            raise DatabaseUnavailableError("DATABASE_URL is not configured.")
        try:
            return psycopg.connect(self.dsn, connect_timeout=self.connect_timeout_s, autocommit=False)
        except psycopg.OperationalError as exc:
            logger.warning("Database connection failed: %s", exc)
            raise DatabaseUnavailableError("The retail database is unavailable.") from exc

    def execute(self, sql: str) -> SQLResult:
        # Cap one row above the limit so truncation can be detected and reported.
        safe_sql = validate_sql(sql, row_limit=self.row_limit + 1)
        with self._connect() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SET TRANSACTION READ ONLY")
                    cur.execute(
                        pgsql.SQL("SET LOCAL statement_timeout = {}").format(
                            pgsql.Literal(int(self.statement_timeout_ms))
                        )
                    )
                    # No parameters are passed, so psycopg does not interpret '%'.
                    cur.execute(safe_sql)
                    columns = [d.name for d in cur.description] if cur.description else []
                    fetched = cur.fetchmany(self.row_limit + 1) if columns else []
            except psycopg.errors.QueryCanceled as exc:
                raise SQLExecutionError(
                    f"Query exceeded the {self.statement_timeout_ms} ms statement timeout."
                ) from exc
            except psycopg.errors.InsufficientPrivilege as exc:
                raise SQLExecutionError("Permission denied: the agent role is read-only.") from exc
            except psycopg.OperationalError as exc:
                raise DatabaseUnavailableError("Lost connection to the retail database.") from exc
            except psycopg.Error as exc:
                message = (exc.diag.message_primary if getattr(exc, "diag", None) else None) or str(exc)
                raise SQLExecutionError(f"Database rejected the query: {message}") from exc
            finally:
                conn.rollback()  # never commit anything from the agent

        truncated = len(fetched) > self.row_limit
        rows = [[_jsonable(v) for v in row] for row in fetched[: self.row_limit]]
        return SQLResult(safe_sql, columns, rows, len(rows), truncated)

    def list_tables(self) -> list[str]:
        return sorted(ALLOWED_TABLES)

    def describe_table(self, table: str) -> list[dict]:
        table = table.strip().lower()
        if table not in ALLOWED_TABLES:
            raise SQLExecutionError(f"Unknown table '{table}'. Available: {', '.join(sorted(ALLOWED_TABLES))}")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table,),
            )
            return [{"column": c, "type": t, "nullable": n == "YES"} for c, t, n in cur.fetchall()]

    def schema_description(self) -> str:
        """Compact schema text for the text-to-SQL prompt (cached)."""
        if self._schema_cache is None:
            parts = []
            for table in self.list_tables():
                cols = ", ".join(f"{c['column']} {c['type']}" for c in self.describe_table(table))
                parts.append(f"{table}({cols})")
            self._schema_cache = "\n".join(parts)
        return self._schema_cache

    def ping(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False
