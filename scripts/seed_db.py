"""Create the retail schema, load deterministic demo data and a read-only role.

Usage:
    python -m scripts.seed_db                     # uses DATABASE_ADMIN_URL
    python -m scripts.seed_db --admin-url postgresql://postgres@localhost:5432/retail

The data is generated with a fixed random seed, so every run produces identical
rows. That keeps the SQL questions in eval/benchmark.jsonl reproducible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
from pathlib import Path

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

SCHEMA_SQL = ROOT / "data" / "retail" / "schema.sql"
SEED = 42

STORES = [
    (1, "Maple Ridge Toronto Queen St", "Toronto", "ON", "Central", "2015-04-01"),
    (2, "Maple Ridge Ottawa Rideau", "Ottawa", "ON", "Central", "2016-09-15"),
    (3, "Maple Ridge Montreal Plateau", "Montreal", "QC", "East", "2017-05-20"),
    (4, "Maple Ridge Vancouver Kitsilano", "Vancouver", "BC", "West", "2018-03-10"),
    (5, "Maple Ridge Calgary Kensington", "Calgary", "AB", "West", "2019-06-01"),
    (6, "Maple Ridge Waterloo Uptown", "Waterloo", "ON", "Central", "2021-08-28"),
    (7, "Maple Ridge Kelowna", "Kelowna", "BC", "West", "2025-03-14"),
    (8, "Maple Ridge Halifax Waterfront", "Halifax", "NS", "East", "2025-10-02"),
]

CATEGORIES = {
    "Tents": (["Trailhead 2P Tent", "Summit 4P Tent", "Ultralight Solo Tent", "Basecamp 6P Tent"], 180, 650),
    "Sleeping Bags": (["Aurora -7C Bag", "Glacier -18C Bag", "Summer Breeze Bag", "Kids Explorer Bag"], 60, 320),
    "Backpacks": (["Ridgeline 45L Pack", "Daytrip 22L Pack", "Expedition 70L Pack", "Commuter 18L Pack"], 45, 340),
    "Footwear": (["Granite Hiking Boot", "Trail Runner X", "Camp Sandal", "Winter Pac Boot"], 55, 260),
    "Apparel": (["Storm Shell Jacket", "Merino Base Layer", "Fleece Midlayer", "Insulated Parka"], 35, 420),
    "Electronics": (["Solar Power Bank", "GPS Trail Watch", "Headlamp 400", "Satellite Messenger"], 30, 480),
    "Cooking": (["Compact Stove", "Titanium Pot Set", "Water Filter Pro", "Insulated Bottle"], 20, 160),
}
BRANDS = ["Maple Ridge", "Northbound", "Cedar & Co", "PeakForm", "Lakeshore"]
FIRST = ["Olivia", "Liam", "Emma", "Noah", "Ava", "Ethan", "Sophia", "Lucas", "Mia", "Aarav",
         "Chloe", "Mohammed", "Zoe", "Wei", "Priya", "Jacob", "Fatima", "Leo", "Hannah", "Arjun"]
LAST = ["Tremblay", "Smith", "Nguyen", "Patel", "Martin", "Roy", "Wilson", "Chen", "Singh", "Brown",
        "Gagnon", "Lee", "Taylor", "Kumar", "Campbell", "Ahmed", "Anderson", "Leblanc", "Wong", "Clark"]


def generate(seed: int = SEED) -> dict[str, list[tuple]]:
    rng = random.Random(seed)

    products = []
    pid = 1
    for category, (names, low, high) in CATEGORIES.items():
        for name in names:
            price = round(rng.uniform(low, high), 2)
            cost = round(price * rng.uniform(0.45, 0.65), 2)
            products.append((pid, f"MR-{category[:3].upper()}-{pid:04d}", name, category,
                             rng.choice(BRANDS), price, cost))
            pid += 1

    customers = []
    tiers = ["Bronze"] * 6 + ["Silver"] * 3 + ["Gold"]
    for cid in range(1, 401):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        store = rng.choice(STORES)
        signup = dt.date(2019, 1, 1) + dt.timedelta(days=rng.randint(0, 6 * 365))
        customers.append((cid, first, last, f"{first}.{last}.{cid}@example.com".lower(),
                          store[2], store[3], rng.choice(tiers), signup))

    orders, items = [], []
    item_id = 1
    start = dt.date(2024, 1, 1)
    for oid in range(1, 2501):
        order_date = start + dt.timedelta(days=rng.randint(0, 670))  # through 2025-11-01
        open_stores = [s for s in STORES if dt.date.fromisoformat(s[5]) <= order_date]
        store = rng.choice(open_stores)
        status = rng.choices(["completed", "returned", "cancelled"], weights=[88, 8, 4])[0]
        channel = rng.choices(["online", "in_store"], weights=[31, 69])[0]
        orders.append((oid, rng.randint(1, 400), store[0], order_date, channel, status))
        for product in rng.sample(products, rng.randint(1, 4)):
            discount = rng.choice([0, 0, 0, 5, 10, 15, 20])
            items.append((item_id, oid, product[0], rng.randint(1, 3), product[5], discount))
            item_id += 1

    inventory = []
    for store in STORES:
        for product in products:
            reorder = rng.randint(5, 15)
            on_hand = rng.randint(0, 60)
            inventory.append((store[0], product[0], on_hand, reorder, dt.date(2025, 11, 1)))

    stores = [(s[0], s[1], s[2], s[3], s[4], dt.date.fromisoformat(s[5])) for s in STORES]
    return {"stores": stores, "products": products, "customers": customers,
            "orders": orders, "order_items": items, "inventory": inventory}


def seed(admin_url: str, readonly_user: str, readonly_password: str) -> dict[str, int]:
    data = generate()
    with psycopg.connect(admin_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL.read_text())
            for table, rows in data.items():
                with cur.copy(sql.SQL("COPY {} FROM STDIN").format(sql.Identifier(table))) as copy:
                    for row in rows:
                        copy.write_row(row)
            _ensure_readonly_role(cur, readonly_user, readonly_password, conn.info.dbname)
        conn.commit()
    return {table: len(rows) for table, rows in data.items()}


def _ensure_readonly_role(cur, user: str, password: str, dbname: str) -> None:
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,))
    action = "ALTER" if cur.fetchone() else "CREATE"
    cur.execute(
        sql.SQL(action + " ROLE {} WITH LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE").format(
            sql.Identifier(user), sql.Literal(password)
        )
    )
    ident = sql.Identifier(user)
    cur.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(ident))
    cur.execute(sql.SQL("ALTER ROLE {} SET statement_timeout = '10s'").format(ident))
    cur.execute(sql.SQL("REVOKE ALL ON SCHEMA public FROM {}").format(ident))
    # PostgreSQL < 15 grants CREATE on schema public to PUBLIC; remove it explicitly.
    cur.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
    cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(dbname), ident))
    cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(ident))
    cur.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(ident))


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--admin-url", default=settings.database_admin_url)
    args = parser.parse_args()
    if not args.admin_url:
        parser.error("Provide --admin-url or set DATABASE_ADMIN_URL.")
    password = settings.readonly_db_password.get_secret_value() if settings.readonly_db_password else ""
    if len(password) < 8 or password.startswith("change-me"):
        parser.error("Set READONLY_DB_PASSWORD (8+ characters, not the .env.example placeholder).")
    counts = seed(args.admin_url, settings.readonly_db_user, password)
    for table, n in counts.items():
        print(f"{table:12s} {n:>6d} rows")
    print(f"Read-only role '{settings.readonly_db_user}' is ready.")


if __name__ == "__main__":
    main()
