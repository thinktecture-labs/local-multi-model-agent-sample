"""
SQLQueryTool — Structured data retrieval via SQL.

For aggregations, time-series data, and anything that lives in rows and
columns rather than documents, SQL is orders of magnitude better than
vector search. This tool gives the agent access to a local SQLite database.

Security: only SELECT statements are permitted. No DDL, DML, or stored procs.
"""

import re

import aiosqlite

from ..inference.config import SQL_MAX_ROWS, SCENARIO_CONFIG, DB_PATH
from .base_tool import BaseTool
from .tool_result import ToolResult


ALLOWED_TABLES = SCENARIO_CONFIG.sql_allowed_tables

# Keywords that should never appear in a SELECT query.
# Covers: multi-statement injection, schema introspection, extension loading, pragmas.
BLOCKED_KEYWORDS = re.compile(
    r"""
    ;                       # multi-statement injection
    | \bATTACH\b            # attach external database
    | \bDETACH\b            # detach database
    | \bPRAGMA\b            # SQLite pragmas (can leak info or change settings)
    | \bload_extension\b    # load arbitrary shared libraries
    | \bsqlite_master\b     # schema introspection
    | \bsqlite_sequence\b   # autoincrement introspection
    | \bsqlite_temp\b       # temp schema introspection
    | \bCREATE\b            # DDL
    | \bDROP\b              # DDL
    | \bALTER\b             # DDL
    | \bINSERT\b            # DML
    | \bUPDATE\b            # DML
    | \bDELETE\b            # DML
    | \bREPLACE\b           # DML
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Extract table names from FROM and JOIN clauses.
_TABLE_REF_PATTERN = re.compile(
    r'\b(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', re.IGNORECASE
)


class SQLQueryTool(BaseTool):
    """
    Execute read-only SQL queries against a local SQLite database.

    Best for: sales figures, counts, aggregations, date-range filters,
    or anything where the data is already structured in tabular form.

    Security:
    - Only SELECT statements are permitted.
    - Blocked keywords: ;, ATTACH, DETACH, PRAGMA, load_extension, DDL, DML.
    - Table allowlist: only {products, customers, sales, competitors}.
    - Result rows capped at SQL_MAX_ROWS.
    """

    name = "sql_query"
    description = SCENARIO_CONFIG.sql_tool_description

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": SCENARIO_CONFIG.sql_parameter_description,
                },
                "limit": {
                    "type":        "integer",
                    "description": f"Maximum rows to return (default: 50, max: {SQL_MAX_ROWS})",
                    "default":     50,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, limit: int = 50, params: tuple | list | None = None) -> ToolResult:
        query = query.strip()

        # Strip a single layer of wrapping parentheses that some LLMs add,
        # e.g. "(SELECT ...)" → "SELECT ...".  Only strip when the parens
        # are a matched outer pair — never touch inner subquery parens.
        bare = query.rstrip(";").rstrip()
        if bare.startswith("(") and bare.endswith(")"):
            inner = bare[1:-1]
            # Only strip if the parens are balanced (no unmatched opens inside)
            depth = 0
            balanced = True
            for ch in inner:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        balanced = False
                        break
                    depth -= 1
            if balanced and depth == 0:
                query = inner.strip()

        # --- Strip SQL comments before safety checks ---
        # SQLite tolerates leading/trailing comments and we need to see the
        # same thing it sees. Order matters: /* … */ first (may span lines),
        # then -- to EOL. This is also what BLOCKED_KEYWORDS runs against,
        # so a comment can neither hide a dangerous keyword nor force a
        # false positive on a benign one.
        stripped = re.sub(r"/\*[\s\S]*?\*/", " ", query)
        stripped = re.sub(r"--[^\n]*", " ", stripped)

        # --- Safety: only SELECT is allowed ---
        normalized = stripped.lstrip().lstrip("(").lower()
        if not normalized.startswith("select"):
            return ToolResult(
                success=False,
                data=None,
                error="Only SELECT queries are allowed. Received: " + query[:60],
            )

        # --- Safety: block dangerous keywords ---
        # Also strip string literals so semicolons / keywords inside quotes
        # are not flagged as injection attempts.
        query_cleaned = re.sub(r"'[^']*'", "", stripped)
        blocked = BLOCKED_KEYWORDS.search(query_cleaned)
        if blocked:
            return ToolResult(
                success=False,
                data=None,
                error=f"Query contains blocked keyword: {blocked.group().strip()}",
            )

        # --- Safety: table allowlist ---
        tables_referenced = {
            m.group(1).lower() for m in _TABLE_REF_PATTERN.finditer(query)
        }
        disallowed = tables_referenced - ALLOWED_TABLES
        if disallowed:
            return ToolResult(
                success=False,
                data=None,
                error=f"Query references disallowed table(s): {', '.join(sorted(disallowed))}",
            )

        limit = max(1, min(limit, SQL_MAX_ROWS))

        # Enforce LIMIT: replace existing LIMIT if it exceeds max, or append.
        # Also handle parameterized "LIMIT ?" — leave as-is for SQLite binding.
        limit_match = re.search(r'\bLIMIT\s+(\d+|\?)', query, re.IGNORECASE)
        if limit_match:
            val = limit_match.group(1)
            if val != "?":
                existing_limit = int(val)
                if existing_limit > SQL_MAX_ROWS:
                    query = query[:limit_match.start()] + f"LIMIT {SQL_MAX_ROWS}" + query[limit_match.end():]
        else:
            query = f"{query.rstrip(';')} LIMIT {limit}"

        # Normalize params: LLMs may pass a single value, a list, or a tuple
        if params is not None:
            if not isinstance(params, (list, tuple)):
                params = (params,)
            else:
                params = tuple(params)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []

            results = [dict(zip(columns, row)) for row in rows]
            return ToolResult(
                success=True,
                data={"columns": columns, "rows": results, "count": len(results)},
            )

        except aiosqlite.OperationalError as exc:
            return ToolResult(success=False, data=None, error=f"SQL error: {exc}")
        except Exception as exc:
            return ToolResult(success=False, data=None, error=str(exc))
