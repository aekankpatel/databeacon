import duckdb
import pytest

from databeacon.sql_guard import SQLValidationError, execute_readonly, validate_sql


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select region, sum(revenue) from ecommerce_orders group by 1",
    "WITH q AS (SELECT * FROM ecommerce_orders) SELECT COUNT(*) FROM q",
    "SELECT 'a;b' AS s",
    "-- comment\nSELECT 1;",
])
def test_allows_read_only(sql):
    assert validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM ecommerce_orders",
    "DROP TABLE ecommerce_orders",
    "INSERT INTO ecommerce_orders VALUES (1)",
    "UPDATE ecommerce_orders SET revenue = 0",
    "SELECT 1; DROP TABLE ecommerce_orders",
    "ATTACH 'x.db' AS x",
    "COPY ecommerce_orders TO 'out.csv'",
    "SELECT * FROM read_csv_auto('/etc/passwd')",
    "SELECT * FROM read_parquet('s3://bucket/x.parquet')",
    "SELECT getenv('HOME')",
    "PRAGMA database_list",
    "INSTALL httpfs",
    "",
])
def test_rejects_unsafe(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_execute_limits_rows(registry, dataset):
    res = execute_readonly(registry.db_path(dataset.id), "SELECT * FROM ecommerce_orders", max_rows=10)
    assert res.row_count == 10 and res.truncated


def test_connection_is_read_only(registry, dataset):
    from databeacon.sql_guard import connect_readonly

    con = connect_readonly(registry.db_path(dataset.id))
    with pytest.raises(duckdb.Error):
        con.execute("DELETE FROM ecommerce_orders")
    con.close()


def test_bad_column_reports_error(registry, dataset):
    with pytest.raises(SQLValidationError, match="revenu"):
        execute_readonly(registry.db_path(dataset.id), "SELECT SUM(revenu) FROM ecommerce_orders")
