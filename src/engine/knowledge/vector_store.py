"""
VectorStore — Persistent semantic document store powered by ChromaDB.

Documents are embedded with embeddinggemma and stored in a local ChromaDB
collection. Both indexing (batch) and querying use the same Gemma embedding
model, ensuring semantic consistency across the retrieval pipeline.

No data ever leaves the machine.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document stored in (or retrieved from) the vector store."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: Optional[float] = None   # Cosine similarity (0–1), set on retrieval


class VectorStore:
    """
    ChromaDB-backed vector store using embeddinggemma for embeddings.

    Architecture note: We manage embeddings ourselves (rather than using
    ChromaDB's built-in embedding functions) so that the exact same
    embeddinggemma model is used for both indexing and querying. This is
    critical: mismatched embedding models destroy retrieval quality.
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_dir: str | None = None,
    ):
        if persist_dir is None:
            from src.engine.inference.config import SCENARIO_CONFIG
            persist_dir = SCENARIO_CONFIG.chroma_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        # cosine distance → higher similarity = better match
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._gemma_client = None  # injected via set_client()

    def set_client(self, gemma_client) -> None:
        """Inject the SmallLanguageModelClient used for generating embeddings."""
        self._gemma_client = gemma_client

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def add_documents(self, documents: list[Document]) -> None:
        """
        Embed and index a list of documents.

        Tries batch embedding first for efficiency. Falls back to individual
        embedding when the batch exceeds the server's physical batch size
        (llama-server default: 512 tokens — long document chunks can exceed this).
        """
        if not self._gemma_client:
            raise RuntimeError("Call set_client(gemma_client) before adding documents.")

        texts      = [doc.content  for doc in documents]
        ids        = [doc.id       for doc in documents]
        # ChromaDB ≥ 0.5 rejects empty metadata dicts; fall back to a placeholder
        # so documents with no metadata can still be indexed.
        metadatas  = [doc.metadata if doc.metadata else {"_": True} for doc in documents]

        # Batch-embed with embeddinggemma — fall back to individual calls on 500
        try:
            embeddings = await self._gemma_client.embed_batch(texts)
        except Exception as exc:
            # Batch too large for server's physical batch size — embed individually
            logger.warning(
                "Batch embedding failed (%s), falling back to individual embedding for %d texts",
                exc, len(texts),
            )
            embeddings = []
            for text in texts:
                emb = await self._gemma_client.embed(text)
                embeddings.append(emb)

        await asyncio.to_thread(
            self._collection.add,
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def add_document(self, document: Document) -> None:
        """Convenience wrapper for indexing a single document."""
        await self.add_documents([document])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[Document]:
        """
        Semantic search: find the top-k most relevant documents.

        The query is embedded with embeddinggemma (same model as indexing),
        and ChromaDB returns the closest vectors by cosine similarity.

        Optional ``where`` filter narrows results by metadata (e.g.
        ``{"document_id": "snowflake-fy2025"}`` to search within a
        specific uploaded document).
        """
        if not self._gemma_client:
            raise RuntimeError("Call set_client(gemma_client) before searching.")

        n_docs = await self.count()
        if n_docs == 0:
            return []

        query_embedding = await self._gemma_client.embed(query)
        n_results = min(top_k, n_docs)

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        results = await asyncio.to_thread(
            self._collection.query,
            **kwargs,
        )

        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB cosine distance ∈ [0, 2]: 0=identical, 2=opposite.
                # Convert to similarity ∈ [0, 1]: similarity = 1 - distance/2.
                # (1 - distance alone gives [-1, 1] — wrong range for distances > 1.)
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                similarity = 1.0 - distance / 2.0

                documents.append(Document(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                    score=round(similarity, 4),
                ))

        return documents

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def count(self) -> int:
        """Return the total number of indexed documents."""
        return await asyncio.to_thread(lambda: self._collection.count())

    async def clear(self) -> None:
        """Delete all documents and reset the collection."""
        name = self._collection.name
        await asyncio.to_thread(self._client.delete_collection, name)
        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    async def get_ids_by_document_id(self, document_id: str) -> list[str]:
        """Return all chunk IDs that belong to the given document_id."""
        result = await asyncio.to_thread(
            self._collection.get, where={"document_id": document_id}
        )
        return result.get("ids", [])

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to the given document_id. Returns count deleted."""
        ids = await self.get_ids_by_document_id(document_id)
        if ids:
            await asyncio.to_thread(self._collection.delete, ids=ids)
        return len(ids)

    async def document_exists(self, doc_id: str) -> bool:
        """Check whether a document with the given ID is already indexed."""
        result = await asyncio.to_thread(
            self._collection.get, ids=[doc_id]
        )
        return len(result["ids"]) > 0
