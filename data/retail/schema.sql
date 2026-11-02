-- Retail demo schema for Maple Ridge Outfitters (fictional company).
DROP TABLE IF EXISTS inventory, order_items, orders, customers, products, stores CASCADE;

CREATE TABLE stores (
    store_id      INTEGER PRIMARY KEY,
    store_name    TEXT NOT NULL,
    city          TEXT NOT NULL,
    province      TEXT NOT NULL,
    region        TEXT NOT NULL,
    opened_date   DATE NOT NULL
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    sku           TEXT NOT NULL UNIQUE,
    product_name  TEXT NOT NULL,
    category      TEXT NOT NULL,
    brand         TEXT NOT NULL,
    unit_price    NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
    unit_cost     NUMERIC(10, 2) NOT NULL CHECK (unit_cost >= 0)
);

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    city          TEXT NOT NULL,
    province      TEXT NOT NULL,
    loyalty_tier  TEXT NOT NULL CHECK (loyalty_tier IN ('Bronze', 'Silver', 'Gold')),
    signup_date   DATE NOT NULL
);

CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers (customer_id),
    store_id      INTEGER NOT NULL REFERENCES stores (store_id),
    order_date    DATE NOT NULL,
    channel       TEXT NOT NULL CHECK (channel IN ('online', 'in_store')),
    status        TEXT NOT NULL CHECK (status IN ('completed', 'returned', 'cancelled'))
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders (order_id),
    product_id    INTEGER NOT NULL REFERENCES products (product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    unit_price    NUMERIC(10, 2) NOT NULL,
    discount_pct  NUMERIC(5, 2) NOT NULL DEFAULT 0 CHECK (discount_pct BETWEEN 0 AND 100)
);

CREATE TABLE inventory (
    store_id          INTEGER NOT NULL REFERENCES stores (store_id),
    product_id        INTEGER NOT NULL REFERENCES products (product_id),
    quantity_on_hand  INTEGER NOT NULL CHECK (quantity_on_hand >= 0),
    reorder_point     INTEGER NOT NULL CHECK (reorder_point >= 0),
    last_counted_at   DATE NOT NULL,
    PRIMARY KEY (store_id, product_id)
);

CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_date ON orders (order_date);
CREATE INDEX idx_order_items_order ON order_items (order_id);
CREATE INDEX idx_order_items_product ON order_items (product_id);

COMMENT ON COLUMN order_items.unit_price IS 'Price per unit at time of sale, before discount';
COMMENT ON COLUMN order_items.discount_pct IS 'Line discount in percent; line revenue = quantity * unit_price * (1 - discount_pct/100)';
