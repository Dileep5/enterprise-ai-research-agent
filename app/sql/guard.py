"""Static validation of model-generated SQL before it reaches PostgreSQL.

This is the first of three layers of defence. The others are a database role with
SELECT-only grants and a READ ONLY transaction with a statement timeout (see
``app.sql.database``). The guard parses SQL into an AST with sqlglot and enforces:

* exactly one statement, which must be a SELECT / set operation (WITH allowed);
* no data-modifying or DDL nodes anywhere in the tree (incl. SELECT ... INTO);
* no row locks (FOR UPDATE / FOR SHARE);
* only allow-listed tables in the public schema;
* no dangerous functions (pg_sleep, pg_read_file, dblink, set_config, ...);
* a LIMIT no greater than the configured row limit.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.errors import UnsafeSQLError

ALLOWED_TABLES = frozenset(
    {"stores", "products", "customers", "orders", "order_items", "inventory"}
)

_FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "TruncateTable",
        "Command", "Copy", "Grant", "Revoke", "Set", "Transaction", "Commit", "Rollback",
        "Use", "Pragma", "LoadData",
    )
    if hasattr(exp, name)
)

_FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "query_to_xml", "table_to_xml")
_FORBIDDEN_FUNCTIONS = frozenset(
    {"set_config", "current_setting", "version", "inet_server_addr", "txid_current", "nextval", "setval"}
)
_SET_OPERATIONS = (exp.Union, exp.Intersect, exp.Except)


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def validate_sql(sql: str, *, row_limit: int, allowed_tables: frozenset[str] = ALLOWED_TABLES) -> str:
    """Return a normalised, LIMIT-capped SQL string or raise ``UnsafeSQLError``."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty SQL query.")
    if len(sql) > 10_000:
        raise UnsafeSQLError("SQL query is too long.")

    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except ParseError as exc:
        raise UnsafeSQLError(f"SQL could not be parsed: {exc}") from exc

    if len(statements) != 1:
        raise UnsafeSQLError("Exactly one SQL statement is allowed.")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, *_SET_OPERATIONS)):
        raise UnsafeSQLError(f"Only SELECT queries are allowed (got {tree.key.upper()}).")

    for node in tree.find_all(exp.Expression):
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeSQLError(f"Forbidden SQL operation: {node.key.upper()}.")
        if isinstance(node, exp.Select):
            if node.args.get("into") is not None:
                raise UnsafeSQLError("SELECT ... INTO is not allowed.")
            if node.args.get("locks"):
                raise UnsafeSQLError("Row locking clauses are not allowed.")
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_FUNCTION_PREFIXES):
                raise UnsafeSQLError(f"Function '{name}' is not allowed.")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        schema = (table.db or "").lower()
        if not name:  # e.g. table-valued function such as generate_series
            raise UnsafeSQLError("Table functions are not allowed.")
        if schema and schema != "public":
            raise UnsafeSQLError(f"Access to schema '{schema}' is not allowed.")
        if name in cte_names and not schema:
            continue
        if name not in allowed_tables:
            raise UnsafeSQLError(f"Table '{name}' is not in the allow-list.")

    tree = _enforce_limit(tree, row_limit)
    return tree.sql(dialect="postgres")


def _enforce_limit(tree: exp.Expression, row_limit: int) -> exp.Expression:
    limit = tree.args.get("limit")
    if limit is None:
        return tree.limit(row_limit)
    value = limit.expression if isinstance(limit, exp.Limit) else None
    if isinstance(value, exp.Literal) and value.is_int:
        if int(value.this) > row_limit:
            return tree.limit(row_limit)
        return tree
    # Non-literal limits (expressions, parameters) are replaced outright.
    return tree.limit(row_limit)
