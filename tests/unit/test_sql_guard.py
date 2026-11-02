import pytest

from app.errors import UnsafeSQLError
from app.sql.guard import validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "select count(*) from customers where loyalty_tier = 'Gold'",
        "WITH t AS (SELECT store_id, count(*) n FROM orders GROUP BY 1) SELECT * FROM t ORDER BY n DESC",
        "SELECT store_id FROM stores UNION SELECT store_id FROM inventory",
        "SELECT o.order_id FROM orders o JOIN order_items oi ON oi.order_id = o.order_id",
        "SELECT date_trunc('month', order_date)::date AS m, count(*) FROM orders GROUP BY 1",
        "SELECT * FROM public.products WHERE product_name LIKE '%Tent%'",
        "SELECT * FROM (SELECT * FROM orders) sub",
    ],
)
def test_allows_read_only_selects(sql):
    out = validate_sql(sql, row_limit=100)
    assert out.upper().startswith(("SELECT", "WITH"))
    assert "LIMIT" in out.upper()


@pytest.mark.parametrize(
    "sql, fragment",
    [
        ("INSERT INTO orders VALUES (1)", "Only SELECT"),
        ("UPDATE orders SET status = 'x'", "Only SELECT"),
        ("DELETE FROM orders", "Only SELECT"),
        ("DROP TABLE orders", "Only SELECT"),
        ("TRUNCATE orders", "Only SELECT"),
        ("SELECT 1; DROP TABLE orders", "Exactly one"),
        ("SELECT * FROM orders; SELECT * FROM stores", "Exactly one"),
        ("SELECT * INTO new_table FROM orders", "INTO"),
        ("SELECT * FROM orders FOR UPDATE", "locking"),
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
        ("SELECT set_config('role', 'postgres', false)", "set_config"),
        ("SELECT * FROM pg_catalog.pg_authid", "schema"),
        ("SELECT * FROM information_schema.tables", "schema"),
        ("SELECT * FROM pg_user", "allow-list"),
        ("SELECT * FROM secrets", "allow-list"),
        ("SELECT * FROM generate_series(1, 1000000000)", "Table functions"),
        ("COPY orders TO '/tmp/out.csv'", "Only SELECT"),
        ("WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d", "Forbidden"),
        ("", "Empty"),
        ("SELEC * FRM orders", ""),
    ],
)
def test_rejects_unsafe_sql(sql, fragment):
    with pytest.raises(UnsafeSQLError) as info:
        validate_sql(sql, row_limit=100)
    assert fragment in info.value.message


def test_limit_is_added_when_missing():
    assert validate_sql("SELECT * FROM orders", row_limit=50).endswith("LIMIT 50")


def test_large_limit_is_capped():
    assert validate_sql("SELECT * FROM orders LIMIT 100000", row_limit=50).endswith("LIMIT 50")


def test_small_limit_is_preserved():
    assert validate_sql("SELECT * FROM orders LIMIT 5", row_limit=50).endswith("LIMIT 5")


def test_cte_names_are_not_treated_as_tables():
    sql = "WITH monthly AS (SELECT * FROM orders) SELECT * FROM monthly"
    assert "monthly" in validate_sql(sql, row_limit=10)
