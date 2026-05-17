"""
CalculatorTool — Safe mathematical expression evaluator.

LLMs are notoriously unreliable at arithmetic. Rather than hoping gemma3
gets "47 * 1.19" right, we delegate all math to this deterministic tool.

Uses simpleeval for safe sandboxed evaluation — no access to builtins,
imports, or attribute lookups. Only whitelisted math functions are exposed.
"""

import math
import re

from simpleeval import SimpleEval, FeatureNotAvailable, NameNotDefined

from .base_tool import BaseTool
from .tool_result import ToolResult


def _make_evaluator() -> SimpleEval:
    """Create a locked-down expression evaluator with math functions only."""
    s = SimpleEval()
    s.functions = {
        "abs":   abs,
        "round": round,
        "min":   min,
        "max":   max,
        "pow":   pow,
        "int":   int,
        "float": float,
        "sqrt":  math.sqrt,
        "ceil":  math.ceil,
        "floor": math.floor,
        "log":   math.log,
        "log10": math.log10,
    }
    s.names = {
        "pi": math.pi,
        "e":  math.e,
    }
    return s


class CalculatorTool(BaseTool):
    """
    Evaluate mathematical expressions safely.

    Handles arithmetic, percentages, and common math functions.
    Uses simpleeval — a purpose-built safe evaluator that blocks attribute
    access, imports, and arbitrary code execution by design.
    """

    name = "calculator"
    description = (
        "Evaluate a mathematical expression with specific numbers. "
        "Use ONLY when the user provides concrete numbers to compute. "
        "Do NOT use for data lookups, aggregations, or product questions."
    )

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type":        "string",
                    "description": (
                        "A mathematical expression to evaluate. "
                        "Use Python syntax: ** for exponents, // for integer division. "
                        "^ also works as an alias for **."
                    ),
                },
            },
            "required": ["expression"],
        }

    @staticmethod
    def _normalize_expression(expr: str) -> str:
        """Convert common LLM expression patterns to valid Python math.

        Handles:
          - "23 * 52400 = 1297200"  → "23 * 52400"  (strip model-appended answer)
          - "23% of 84900"  → "0.23 * 84900"
          - "23% * 84900"   → "0.23 * 84900"
          - "23% = 84900"   → "0.23 * 84900"  (LLM expression quirk)
          - "23% 84900"     → "0.23 * 84900"  (missing operator)
          - "15%"           → "0.15"  (bare percentage)
          - "^"             → "**"
        """
        s = expr.strip()
        # Handle "X% = X * Y" — model restates percentage as integer before multiplying
        # e.g. "15% = 15 * 45000" → "0.15 * 45000" (strip the redundant "15 *")
        s = re.sub(
            r'(-?\d+(?:\.\d+)?)%\s*=\s*\1\s*\*\s*',
            lambda m: f"{float(m.group(1)) / 100} * ",
            s,
        )
        # Replace "X% of Y" or "X% = Y" or "X% * Y" with "X/100 * Y"
        # MUST run before trailing "= result" stripper (15% = 45000 needs the = kept)
        s = re.sub(
            r'(-?\d+(?:\.\d+)?)%\s*(?:of|=|\*)\s*',
            lambda m: f"{float(m.group(1)) / 100} * ",
            s,
        )
        # Replace "X% Y" (missing operator) with "X/100 * Y"
        s = re.sub(
            r'(-?\d+(?:\.\d+)?)%\s+(\d)',
            lambda m: f"{float(m.group(1)) / 100} * {m.group(2)}",
            s,
        )
        # Replace remaining bare "X%" with "X/100"
        s = re.sub(r'(-?\d+(?:\.\d+)?)%', lambda m: str(float(m.group(1)) / 100), s)
        # Strip trailing "= <result>" that the model sometimes appends
        # Runs AFTER % handling so "15% = 45000" is already converted
        s = re.sub(r'\s*=\s*[\d,.]+\s*$', '', s)
        # Allow ^ as an alias for exponentiation
        s = s.replace("^", "**")
        return s

    async def execute(self, expression: str) -> ToolResult:
        try:
            sanitized = self._normalize_expression(expression)

            evaluator = _make_evaluator()
            result = evaluator.eval(sanitized)

            # Ensure result is a plain Python number
            if isinstance(result, (int, float)):
                return ToolResult(
                    success=True,
                    data={
                        "expression": expression,
                        "result":     round(result, 10) if isinstance(result, float) else result,
                    },
                )
            return ToolResult(
                success=False,
                data=None,
                error=f"Expression did not return a number: {result!r}",
            )

        except ZeroDivisionError:
            return ToolResult(success=False, data=None, error="Division by zero")
        except (FeatureNotAvailable, NameNotDefined) as exc:
            return ToolResult(
                success=False,
                data=None,
                error=f"Blocked unsafe expression '{expression}': {exc}",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                data=None,
                error=f"Could not evaluate '{expression}': {exc}",
            )
