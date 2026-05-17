"""
Unit tests for CalculatorTool.

No external dependencies — all tests run without model servers or a database.
"""

import math

import pytest

from src.engine.tools.calculator import CalculatorTool


@pytest.fixture
def calc():
    return CalculatorTool()


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBasicArithmetic:
    async def test_addition(self, calc):
        result = await calc.execute("2 + 3")
        assert result.success
        assert result.data["result"] == 5

    async def test_subtraction(self, calc):
        result = await calc.execute("10 - 4")
        assert result.success
        assert result.data["result"] == 6

    async def test_multiplication(self, calc):
        result = await calc.execute("7 * 8")
        assert result.success
        assert result.data["result"] == 56

    async def test_float_division(self, calc):
        result = await calc.execute("10 / 4")
        assert result.success
        assert abs(result.data["result"] - 2.5) < 1e-9

    async def test_integer_division(self, calc):
        result = await calc.execute("10 // 3")
        assert result.success
        assert result.data["result"] == 3

    async def test_exponentiation(self, calc):
        result = await calc.execute("2 ** 10")
        assert result.success
        assert result.data["result"] == 1024

    async def test_caret_alias_for_exponent(self, calc):
        """^ should be treated as ** (article requirement)."""
        result = await calc.execute("2^10")
        assert result.success
        assert result.data["result"] == 1024

    async def test_modulo(self, calc):
        result = await calc.execute("17 % 5")
        assert result.success
        assert result.data["result"] == 2

    async def test_negative_numbers(self, calc):
        result = await calc.execute("-5 * 3")
        assert result.success
        assert result.data["result"] == -15

    async def test_parentheses_order_of_operations(self, calc):
        result = await calc.execute("(2 + 3) * 4")
        assert result.success
        assert result.data["result"] == 20


# ---------------------------------------------------------------------------
# Business calculations (from the demo showcase)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBusinessCalculations:
    async def test_percentage(self, calc):
        """23% of 84900 = 19527."""
        result = await calc.execute("0.23 * 84900")
        assert result.success
        assert abs(result.data["result"] - 19527.0) < 0.01

    async def test_arr_calculation(self, calc):
        """50 customers × €999/month × 12 months = €599,400 ARR."""
        result = await calc.execute("50 * 999 * 12")
        assert result.success
        assert result.data["result"] == 599400

    async def test_roi(self, calc):
        """ROI = (120000 - 50000) / 50000 * 100 = 140%."""
        result = await calc.execute("(120000 - 50000) / 50000 * 100")
        assert result.success
        assert abs(result.data["result"] - 140.0) < 0.01

    async def test_monthly_from_annual(self, calc):
        result = await calc.execute("36000 / 12")
        assert result.success
        assert result.data["result"] == 3000.0


# ---------------------------------------------------------------------------
# Math functions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMathFunctions:
    async def test_sqrt(self, calc):
        result = await calc.execute("sqrt(144)")
        assert result.success
        assert result.data["result"] == 12.0

    async def test_ceil(self, calc):
        result = await calc.execute("ceil(4.2)")
        assert result.success
        assert result.data["result"] == 5

    async def test_floor(self, calc):
        result = await calc.execute("floor(4.9)")
        assert result.success
        assert result.data["result"] == 4

    async def test_abs(self, calc):
        result = await calc.execute("abs(-42)")
        assert result.success
        assert result.data["result"] == 42

    async def test_round(self, calc):
        result = await calc.execute("round(3.14159, 2)")
        assert result.success
        assert result.data["result"] == 3.14

    async def test_pi(self, calc):
        result = await calc.execute("pi * 2")
        assert result.success
        assert abs(result.data["result"] - 2 * math.pi) < 1e-9

    async def test_log(self, calc):
        result = await calc.execute("log(math.e)")
        # log() without math prefix — should fail (math not in namespace)
        # But log is imported from math, not math.e
        # Actually log is available but math.e is not (math module is not in namespace)
        # Let's just call log(2.718281828)
        result = await calc.execute("round(log(2.718281828), 4)")
        assert result.success
        assert abs(result.data["result"] - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestErrorHandling:
    async def test_division_by_zero(self, calc):
        result = await calc.execute("1 / 0")
        assert not result.success
        assert "zero" in result.error.lower()

    async def test_invalid_expression(self, calc):
        result = await calc.execute("not a math expression !!!")
        assert not result.success
        assert result.error is not None

    async def test_import_attempt_blocked(self, calc):
        """Sandboxed eval should reject attempts to import modules."""
        result = await calc.execute("__import__('os').system('echo pwned')")
        assert not result.success

    async def test_string_result_rejected(self, calc):
        """Expressions that produce non-numeric results should fail."""
        result = await calc.execute("'hello'")
        assert not result.success

    async def test_expression_stored_in_result(self, calc):
        """The original expression should be echoed back in the data."""
        result = await calc.execute("1 + 1")
        assert result.success
        assert result.data["expression"] == "1 + 1"
