"""
Integration tests for SQLQueryTool.

These tests require a real (temporary) SQLite database but no model servers.
The `temp_db` fixture in conftest.py seeds the demo schema (Nextera) automatically.
"""

import pytest

from src.engine.tools.sql_query import SQLQueryTool


@pytest.fixture
def sql_tool(request):
    """SQL tool backed by the temp database seeded with the Nextera demo schema."""
    db_path = request.getfixturevalue("temp_db")
    return SQLQueryTool(db_path=db_path)


# ── Nextera scenario query constants ─────────────────────────────────────────

if True:  # kept as a block for diff-friendliness with prior scenario branching
    # Nextera scenario
    _T_MAIN = "products"
    _T_SECONDARY = "customers"
    _T_THIRD = "sales"
    _T_FOURTH = "products"
    _T_FIFTH = "customers"

    _Q_SELECT_ALL = "SELECT * FROM products"
    _Q_SELECT_ALL_COUNT = 3
    _Q_SELECT_ALL_COL1 = "id"
    _Q_SELECT_ALL_COL2 = "price_monthly"

    _Q_WHERE = "SELECT name, price_monthly FROM products WHERE category = 'platform' AND price_monthly > 500"
    _Q_WHERE_COUNT = 2  # Professional + Enterprise

    _Q_AGG = "SELECT SUM(revenue) as total FROM sales WHERE year = 2024"
    _Q_AGG_TOTAL = 311500  # 55100 + 68300 + 84900 + 103200

    _Q_MULTI_ROW = "SELECT quarter, revenue FROM sales WHERE year = 2024 ORDER BY quarter"
    _Q_MULTI_ROW_COUNT = 4
    _Q_MULTI_ROW_KEY = "quarter"
    _Q_MULTI_ROW_VALS = ["Q1", "Q2", "Q3", "Q4"]

    _Q_COUNT = "SELECT COUNT(*) as n FROM customers"
    _Q_COUNT_N = 3

    _Q_JOIN = (
        "SELECT c.name, c.tier, p.price_monthly "
        "FROM customers c "
        "JOIN products p ON c.tier = p.category "
        "ORDER BY p.price_monthly DESC"
    )

    _Q_FILTER = "SELECT name, mrr FROM customers WHERE tier = 'enterprise'"
    _Q_FILTER_COUNT = 2  # Acme Corp + BrightHealth GmbH

    _Q_RESULT_SHAPE = "SELECT id, name FROM products"

    # Row limiting
    _Q_LIMIT_TABLE = "SELECT * FROM sales"
    _Q_LIMIT_EXPECTED = 4

    # Security (table names for DML)
    _T_DML = "products"

    # Fuzzing
    _Q_STACKED = "SELECT 1; DROP TABLE products"
    _Q_STACKED_CHECK = "SELECT COUNT(*) as n FROM products"
    _Q_STACKED_CHECK_N = 3
    _Q_UNION = "SELECT name FROM products UNION SELECT name FROM customers"
    _Q_UNION_MIN = 3
    _Q_SUB = "SELECT * FROM (SELECT name, price_monthly FROM products)"
    _Q_COMMENT = "SELECT * FROM products -- WHERE restricted = 1"
    _Q_QUOTE = "SELECT * FROM customers WHERE name = 'O''Brien'"
    _Q_SEMI = "SELECT * FROM customers WHERE name = 'test; DROP TABLE customers'"
    _Q_UNICODE = "SELECT * FROM products WHERE name = '日本語テスト'"
    _Q_LONG_TABLE = "products"
    _Q_LONG_COL = "name"
    _Q_NESTED = "(SELECT * FROM (SELECT name FROM products WHERE id IN (1, 2)))"
    _Q_PAREN = "(SELECT COUNT(*) as n FROM products)"


# ---------------------------------------------------------------------------
# Valid SELECT queries
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestValidSelects:
    async def test_select_all(self, sql_tool):
        result = await sql_tool.execute(_Q_SELECT_ALL)
        assert result.success
        assert result.data["count"] == _Q_SELECT_ALL_COUNT
        assert _Q_SELECT_ALL_COL1 in result.data["columns"]
        assert _Q_SELECT_ALL_COL2 in result.data["columns"]

    async def test_select_with_where(self, sql_tool):
        result = await sql_tool.execute(_Q_WHERE)
        assert result.success
        assert result.data["count"] == _Q_WHERE_COUNT

    async def test_select_aggregation(self, sql_tool):
        result = await sql_tool.execute(_Q_AGG)
        assert result.success
        assert result.data["count"] == 1
        total = result.data["rows"][0]["total"]
        assert total == _Q_AGG_TOTAL

    async def test_select_multi_row(self, sql_tool):
        result = await sql_tool.execute(_Q_MULTI_ROW)
        assert result.success
        assert result.data["count"] == _Q_MULTI_ROW_COUNT
        vals = [row[_Q_MULTI_ROW_KEY] for row in result.data["rows"]]
        assert vals == _Q_MULTI_ROW_VALS

    async def test_select_count(self, sql_tool):
        result = await sql_tool.execute(_Q_COUNT)
        assert result.success
        assert result.data["rows"][0]["n"] == _Q_COUNT_N

    async def test_select_joins(self, sql_tool):
        result = await sql_tool.execute(_Q_JOIN)
        # Test that the query executes cleanly.
        assert result.success

    async def test_select_filtered(self, sql_tool):
        result = await sql_tool.execute(_Q_FILTER)
        assert result.success
        assert result.data["count"] == _Q_FILTER_COUNT

    async def test_result_has_columns_and_rows(self, sql_tool):
        result = await sql_tool.execute(_Q_RESULT_SHAPE)
        assert result.success
        assert "columns" in result.data
        assert "rows" in result.data
        assert "count" in result.data


# ---------------------------------------------------------------------------
# Security: non-SELECT statements blocked
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSecurity:
    async def test_insert_rejected(self, sql_tool):
        result = await sql_tool.execute(
            f"INSERT INTO {_T_DML} VALUES (99, 'X', 'test', 0, 0)"
        )
        assert not result.success
        assert "SELECT" in result.error or "select" in result.error.lower()

    async def test_update_rejected(self, sql_tool):
        result = await sql_tool.execute(f"UPDATE {_T_DML} SET {_Q_SELECT_ALL_COL1} = 'x'")
        assert not result.success

    async def test_delete_rejected(self, sql_tool):
        result = await sql_tool.execute(f"DELETE FROM {_T_DML}")
        assert not result.success

    async def test_drop_rejected(self, sql_tool):
        result = await sql_tool.execute(f"DROP TABLE {_T_DML}")
        assert not result.success

    async def test_wrapped_select_with_leading_paren(self, sql_tool):
        """(SELECT ...) should be allowed — the tool strips leading parens."""
        result = await sql_tool.execute(_Q_PAREN)
        assert result.success


# ---------------------------------------------------------------------------
# Row limiting
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRowLimiting:
    async def test_default_limit_applied(self, sql_tool):
        """LIMIT should be appended automatically if not present."""
        result = await sql_tool.execute(_Q_LIMIT_TABLE)
        assert result.success
        assert result.data["count"] <= 50  # default limit

    async def test_explicit_limit_respected(self, sql_tool):
        result = await sql_tool.execute(_Q_LIMIT_TABLE, limit=2)
        assert result.success
        assert result.data["count"] <= 2

    async def test_limit_not_doubled_when_present(self, sql_tool):
        """Queries that already include LIMIT should not get a second LIMIT."""
        result = await sql_tool.execute(f"{_Q_LIMIT_TABLE} LIMIT 1")
        assert result.success
        assert result.data["count"] == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestErrorHandling:
    async def test_nonexistent_table(self, sql_tool):
        result = await sql_tool.execute("SELECT * FROM no_such_table")
        assert not result.success
        assert result.error is not None

    async def test_invalid_sql_syntax(self, sql_tool):
        result = await sql_tool.execute("SELECT FROM WHERE")
        assert not result.success

    async def test_nonexistent_column(self, sql_tool):
        result = await sql_tool.execute(f"SELECT nonexistent_column FROM {_T_MAIN}")
        assert not result.success


# ---------------------------------------------------------------------------
# SQL injection / fuzz testing
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSQLFuzzing:
    """Verify the tool handles adversarial and edge-case SQL inputs safely."""

    async def test_classic_injection_rejected(self, sql_tool):
        """'; DROP TABLE ...; -- must be blocked (not a SELECT)."""
        result = await sql_tool.execute(f"'; DROP TABLE {_T_DML}; --")
        assert not result.success

    async def test_stacked_query_rejected(self, sql_tool):
        """SQLite execute() only runs one statement; stacked queries should fail."""
        result = await sql_tool.execute(_Q_STACKED)
        assert not result.success
        # Verify table still exists
        check = await sql_tool.execute(_Q_STACKED_CHECK)
        assert check.success
        assert check.data["rows"][0]["n"] == _Q_STACKED_CHECK_N

    async def test_attach_database_rejected(self, sql_tool):
        result = await sql_tool.execute("ATTACH DATABASE ':memory:' AS injected")
        assert not result.success

    async def test_create_table_rejected(self, sql_tool):
        result = await sql_tool.execute("CREATE TABLE hacked (id INTEGER)")
        assert not result.success

    async def test_union_select_executes(self, sql_tool):
        """UNION SELECT is valid SQL — should execute (it's still a SELECT)."""
        result = await sql_tool.execute(_Q_UNION)
        assert result.success
        assert result.data["count"] >= _Q_UNION_MIN

    async def test_subquery_executes(self, sql_tool):
        """Subqueries are valid SELECT statements."""
        result = await sql_tool.execute(_Q_SUB)
        assert result.success

    async def test_comment_injection(self, sql_tool):
        """SQL comments should not bypass safety — still a valid SELECT."""
        result = await sql_tool.execute(_Q_COMMENT)
        assert result.success

    async def test_blocked_keyword_inside_block_comment_does_not_false_positive(self, sql_tool):
        """`SELECT /* DROP */ …` — the DROP is a comment, the SELECT is safe.

        Before comment-stripping, BLOCKED_KEYWORDS would see DROP and reject
        a perfectly safe query. After stripping, this succeeds.
        """
        result = await sql_tool.execute(
            f"SELECT /* DROP TABLE {_T_DML} */ * FROM {_T_DML} LIMIT 1"
        )
        assert result.success

    async def test_blocked_keyword_inside_line_comment_does_not_false_positive(self, sql_tool):
        """`SELECT … -- DROP \\n …` — the DROP is in a comment, query is safe."""
        result = await sql_tool.execute(
            f"SELECT * FROM {_T_DML} -- DROP TABLE {_T_DML}\nLIMIT 1"
        )
        assert result.success

    async def test_leading_block_comment_disguising_dml_rejected(self, sql_tool):
        """`/* SELECT 1 */ UPDATE …` — leading comment must not let DML through."""
        result = await sql_tool.execute(
            f"/* SELECT 1 */ UPDATE {_T_DML} SET name = 'pwned' WHERE id = 1"
        )
        assert not result.success

    async def test_string_with_single_quotes(self, sql_tool):
        """Names with apostrophes should not break the query."""
        result = await sql_tool.execute(_Q_QUOTE)
        assert result.success
        assert result.data["count"] == 0  # no match, but no crash

    async def test_string_with_semicolon_in_literal(self, sql_tool):
        """Semicolons inside string literals should not trigger stacking."""
        result = await sql_tool.execute(_Q_SEMI)
        assert result.success
        assert result.data["count"] == 0

    async def test_unicode_in_query(self, sql_tool):
        """Unicode characters should not crash the SQL engine."""
        result = await sql_tool.execute(_Q_UNICODE)
        assert result.success
        assert result.data["count"] == 0

    async def test_empty_string_rejected(self, sql_tool):
        """Empty input should fail (not a SELECT)."""
        result = await sql_tool.execute("")
        assert not result.success

    async def test_whitespace_only_rejected(self, sql_tool):
        """Whitespace-only input should fail."""
        result = await sql_tool.execute("   \n\t  ")
        assert not result.success

    async def test_null_byte_handled(self, sql_tool):
        """Null bytes should not cause crashes."""
        result = await sql_tool.execute(f"SELECT\x00* FROM {_T_MAIN}")
        # Should either execute or fail gracefully — never crash
        assert isinstance(result.success, bool)

    async def test_very_long_query(self, sql_tool):
        """Extremely long queries should not crash the tool."""
        values = ", ".join(f"'{i}'" for i in range(500))
        result = await sql_tool.execute(
            f"SELECT * FROM {_Q_LONG_TABLE} WHERE {_Q_LONG_COL} IN ({values})"
        )
        assert result.success
        assert result.data["count"] == 0  # no matches, but no crash

    async def test_nested_parens_subquery(self, sql_tool):
        """Deeply nested parentheses should be handled correctly."""
        result = await sql_tool.execute(_Q_NESTED)
        assert result.success
