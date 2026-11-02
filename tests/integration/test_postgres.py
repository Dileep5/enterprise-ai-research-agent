"""Integration tests against a real PostgreSQL server.

Run with:  TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/retail_test pytest -m postgres
The URL must be an admin connection to a disposable database; it is re-seeded.
"""

import json
import os

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from app.errors import DatabaseUnavailableError, SQLExecutionError, UnsafeSQLError
from app.sql.database import RetailDatabase
from scripts.seed_db import seed
from tests.conftest import FakeLLM

pytestmark = pytest.mark.postgres

READONLY_USER = "agent_readonly_test"
READONLY_PASSWORD = "test-readonly-pw"


@pytest.fixture(scope="module")
def readonly_dsn():
    admin = os.environ["TEST_DATABASE_URL"]
    counts = seed(admin, READONLY_USER, READONLY_PASSWORD)
    assert counts["orders"] == 2500
    params = conninfo_to_dict(admin)
    params.update(user=READONLY_USER, password=READONLY_PASSWORD)
    return make_conninfo(**params)


@pytest.fixture
def db(readonly_dsn):
    return RetailDatabase(readonly_dsn, row_limit=50, statement_timeout_ms=300)


def test_aggregate_query(db):
    result = db.execute("SELECT count(*) AS n FROM orders")
    assert result.columns == ["n"] and result.rows == [[2500]]


def test_seed_is_deterministic(readonly_dsn):
    db = RetailDatabase(readonly_dsn)
    total = db.execute(
        "SELECT round(sum(quantity * unit_price * (1 - discount_pct / 100)), 2) FROM order_items"
    ).rows[0][0]
    seed(os.environ["TEST_DATABASE_URL"], READONLY_USER, READONLY_PASSWORD)
    again = db.execute(
        "SELECT round(sum(quantity * unit_price * (1 - discount_pct / 100)), 2) FROM order_items"
    ).rows[0][0]
    assert total == again


def test_row_limit_and_truncation(db):
    result = db.execute("SELECT * FROM order_items")
    assert result.row_count == 50 and result.truncated is True
    assert result.query.endswith("LIMIT 51")  # one extra row detects truncation
    small = db.execute("SELECT * FROM stores")
    assert small.row_count == 8 and small.truncated is False


def test_decimals_and_dates_are_json_serialisable(db):
    result = db.execute("SELECT order_date, unit_price FROM orders JOIN order_items USING (order_id) LIMIT 3")
    json.dumps(result.to_dict())


def test_statement_timeout_is_enforced(db):
    with pytest.raises(SQLExecutionError, match="statement timeout"):
        db.execute("SELECT count(*) FROM order_items a, order_items b, order_items c")


def test_bad_column_gives_readable_error(db):
    with pytest.raises(SQLExecutionError, match='column "revenue" does not exist'):
        db.execute("SELECT revenue FROM orders")


def test_guard_runs_before_database(db):
    with pytest.raises(UnsafeSQLError):
        db.execute("DELETE FROM orders")


def test_readonly_role_cannot_write_even_without_guard(readonly_dsn):
    denied = (psycopg.errors.ReadOnlySqlTransaction, psycopg.errors.InsufficientPrivilege)
    with psycopg.connect(readonly_dsn) as conn, pytest.raises(denied):
        conn.execute("DELETE FROM orders")


def test_readonly_role_cannot_create_tables(readonly_dsn):
    with psycopg.connect(readonly_dsn, autocommit=True) as conn:
        conn.execute("SET default_transaction_read_only = off")  # even if the session flag is lifted
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("CREATE TABLE evil (id int)")


def test_schema_introspection(db):
    assert {c["column"] for c in db.describe_table("orders")} >= {"order_id", "status", "channel"}
    assert "order_items(" in db.schema_description()
    with pytest.raises(SQLExecutionError):
        db.describe_table("pg_authid")


def test_unreachable_database():
    db = RetailDatabase("postgresql://nobody@127.0.0.1:1/none", statement_timeout_ms=200)
    with pytest.raises(DatabaseUnavailableError):
        db.execute("SELECT 1 FROM orders")
    assert db.ping() is False


def test_agent_answers_sql_question_against_real_database(make_services, readonly_dsn):
    sql = ("SELECT s.store_name, count(*) AS orders FROM orders o JOIN stores s USING (store_id) "
           "GROUP BY s.store_name ORDER BY orders DESC LIMIT 1")

    def handler(kind, system, user):
        if kind == "sql":
            assert "stores(" in system  # real schema reached the prompt
            return sql
        return "The busiest store is shown in [SQL]."

    services = make_services(llm=FakeLLM(handler), database=RetailDatabase(readonly_dsn))
    state = services.agent.run("Which store has the most orders?", mode="sql")
    assert state["sql_error"] is None
    assert state["sql_result"]["columns"] == ["store_name", "orders"]
    assert state["sql_result"]["row_count"] == 1
