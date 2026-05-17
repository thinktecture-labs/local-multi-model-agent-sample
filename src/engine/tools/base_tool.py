"""BaseTool — abstract base class that every agent tool must implement."""

from abc import ABC, abstractmethod

from .tool_result import ToolResult


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Every tool must declare:
      name        — unique string identifier (used in function schemas)
      description — plain-English description of what it does
                    (the tool-calling model reads this to pick the right tool)

    And implement:
      execute()         — the actual async logic
      _get_parameters() — JSON Schema describing the tool's inputs
    """

    name: str
    description: str

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters."""

    def get_schema(self) -> dict:
        """
        Return an OpenAI-compatible function schema for this tool.

        Passed directly to the tool-calling model's `tools` parameter.
        Clear, specific descriptions here directly improve tool selection accuracy
        — this is one of the highest-leverage things to fine-tune.
        """
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self._get_parameters(),
            },
        }

    @abstractmethod
    def _get_parameters(self) -> dict:
        """Return a JSON Schema object describing this tool's input parameters."""
