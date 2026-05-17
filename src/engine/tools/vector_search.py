"""
VectorSearchTool — Semantic document retrieval using embeddinggemma.

This is the heart of the RAG pipeline. When the agent needs to answer
a question from the knowledge base, it calls this tool to find the most
relevant context documents before synthesizing a response.

Searches two collections when available:
  - knowledge_base: curated, seeded docs for the active scenario
  - uploads: user-uploaded documents (PDF/TXT/MD via /upload-document)

Results are merged by cosine similarity score and the top-k returned.
"""

from ..inference.config import VECTOR_SEARCH_MAX_K, UPLOAD_MERGE_MIN_SCORE
from .base_tool import BaseTool
from .tool_result import ToolResult


class VectorSearchTool(BaseTool):
    """
    Search the local knowledge base and uploaded documents using semantic similarity.

    Powered by embeddinggemma — which understands domain-specific
    terminology much better than keyword search ever could.

    When an upload_store is provided, both the curated knowledge base
    and user-uploaded documents are searched, with results merged by
    cosine similarity score.
    """

    name = "vector_search"
    description = (
        "Search documentation and knowledge articles. Use when the user asks "
        "about how things work, what features exist, compliance, deployment, "
        "or support policies. Do NOT use for database lookups or numeric data."
    )

    def __init__(self, vector_store, upload_store=None) -> None:
        self.vector_store = vector_store
        self.upload_store = upload_store

    def _get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": (
                        "The search query. Be specific and use domain terminology "
                        "for better results. Example: 'enterprise plan features and limits'"
                    ),
                },
                "top_k": {
                    "type":        "integer",
                    "description": f"Number of documents to retrieve (default: 5, max: {VECTOR_SEARCH_MAX_K})",
                    "default":     5,
                },
            },
            "required": ["query"],
        }

    async def execute(
        self, query: str, top_k: int = 5, include_all_uploads: bool = False,
    ) -> ToolResult:
        """
        Search the curated KB and (optionally) uploaded documents.

        Args:
            query: Natural-language question.
            top_k: Number of chunks to return (capped at VECTOR_SEARCH_MAX_K).
            include_all_uploads: When True, upload chunks bypass the
                UPLOAD_MERGE_MIN_SCORE filter and are merged unconditionally.
                Used by HITL cloud escalation, where the user has explicitly
                opted in and the cloud needs the user's local context as an
                anchor — pollution risk is no longer a concern at that point.
                Default False preserves the existing protective behaviour
                for normal RAG queries (where an off-topic upload like a
                conference agenda must not bleed into curated-KB answers).
        """
        try:
            top_k = max(1, min(top_k, VECTOR_SEARCH_MAX_K))

            # Search curated knowledge base
            kb_results = await self.vector_store.search(query, top_k=top_k)

            # Search uploaded documents (if any) and merge.
            # Default: only chunks scoring above UPLOAD_MERGE_MIN_SCORE are included
            # (prevents weakly-matching uploads from polluting domain-specific queries).
            # When include_all_uploads=True (escalation path), the threshold is bypassed.
            if self.upload_store is not None:
                upload_count = await self.upload_store.count()
                if upload_count > 0:
                    upload_results = await self.upload_store.search(query, top_k=top_k)
                    if include_all_uploads:
                        relevant_uploads = upload_results
                    else:
                        relevant_uploads = [
                            d for d in upload_results
                            if d.score is not None and d.score >= UPLOAD_MERGE_MIN_SCORE
                        ]
                    if relevant_uploads:
                        combined = kb_results + relevant_uploads
                        combined.sort(key=lambda d: d.score if d.score is not None else 0, reverse=True)
                        return ToolResult(success=True, data=combined[:top_k])

            return ToolResult(success=True, data=kb_results)
        except Exception as exc:
            return ToolResult(success=False, data=None, error=str(exc))
