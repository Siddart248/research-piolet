"""Knowledge Base Document Ingestion, Chunking, and NumPy Vector Retrieval.

Implements:
- load_documents: Reads markdown files from data/knowledge_base/.
- chunk_documents: Paragraph-aware sliding window text chunking.
- LocalVectorStore: NumPy-based vector store calculating cosine similarity.
- retrieve: Queries vector store, enforces similarity thresholding, and logs traces.
- build_index: Complete pipeline to load, chunk, embed, and serialize index.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence
import numpy as np
from google import genai

from app.config import get_settings, PROJECT_ROOT
from app.embeddings import (
    cosine_similarity,
    embed_text,
    embed_texts,
)
from app.schemas import Chunk, Document, RetrievalResult, RetrievedChunk

logger = logging.getLogger("retrieval")

DEFAULT_KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"


def load_documents(kb_dir: Path | str | None = None) -> list[Document]:
    """Load all Markdown and text documents from the knowledge base directory.

    Args:
        kb_dir: Directory containing knowledge base files. Defaults to data/knowledge_base/.

    Returns:
        list[Document]: Loaded documents with content and metadata.
    """
    directory = Path(kb_dir) if kb_dir else DEFAULT_KB_DIR
    if not directory.exists():
        logger.warning(f"Knowledge base directory '{directory}' does not exist.")
        return []

    documents: list[Document] = []
    supported_extensions = {".md", ".markdown", ".txt"}

    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                content = file_path.read_text(encoding="utf-8").strip()
                if content:
                    documents.append(
                        Document(
                            id=file_path.stem,
                            content=content,
                            source=file_path.name,
                            metadata={"file_path": str(file_path), "size_chars": len(content)},
                        )
                    )
            except Exception as err:
                logger.error(f"Failed to read '{file_path.name}': {err}")

    logger.info(f"Loaded {len(documents)} documents from '{directory.name}'")
    return documents


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split documents into smaller, semantically coherent text chunks.

    Uses a paragraph-aware sliding window strategy:
    1. Splits text by logical paragraph breaks.
    2. Groups consecutive paragraphs into chunks up to chunk_size characters.
    3. Handles oversized paragraphs with sliding character windows and chunk_overlap.

    Args:
        documents: List of input Document instances.
        chunk_size: Target maximum character length per chunk.
        chunk_overlap: Overlap length in characters between adjacent chunks.

    Returns:
        list[Chunk]: Generated chunks with unique chunk_id and source references.
    """
    chunks: list[Chunk] = []

    for doc in documents:
        # Split by paragraph breaks
        raw_paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
        current_chunk_paragraphs: list[str] = []
        current_length = 0
        chunk_counter = 0

        def emit_chunk(text: str, index: int) -> Chunk:
            return Chunk(
                chunk_id=f"{doc.id}_chunk_{index:03d}",
                source=doc.source,
                text=text.strip(),
                metadata={
                    "char_length": len(text.strip()),
                    "chunk_index": index,
                },
            )

        for paragraph in raw_paragraphs:
            # If paragraph itself is larger than chunk_size, split by sliding window
            if len(paragraph) > chunk_size:
                # Flush accumulated smaller paragraphs first
                if current_chunk_paragraphs:
                    combined_text = "\n\n".join(current_chunk_paragraphs)
                    chunks.append(emit_chunk(combined_text, chunk_counter))
                    chunk_counter += 1
                    current_chunk_paragraphs = []
                    current_length = 0

                start = 0
                while start < len(paragraph):
                    end = min(start + chunk_size, len(paragraph))
                    sub_text = paragraph[start:end]
                    chunks.append(emit_chunk(sub_text, chunk_counter))
                    chunk_counter += 1
                    if end >= len(paragraph):
                        break
                    start += chunk_size - chunk_overlap
                continue

            # Standard paragraph accumulation
            added_length = len(paragraph) + (2 if current_chunk_paragraphs else 0)
            if current_length + added_length <= chunk_size:
                current_chunk_paragraphs.append(paragraph)
                current_length += added_length
            else:
                # Emit current accumulated chunk
                combined_text = "\n\n".join(current_chunk_paragraphs)
                chunks.append(emit_chunk(combined_text, chunk_counter))
                chunk_counter += 1

                # Begin new chunk with overlap if possible
                overlap_text = current_chunk_paragraphs[-1] if current_chunk_paragraphs else ""
                if len(overlap_text) <= chunk_overlap:
                    current_chunk_paragraphs = [overlap_text, paragraph]
                    current_length = len(overlap_text) + 2 + len(paragraph)
                else:
                    current_chunk_paragraphs = [paragraph]
                    current_length = len(paragraph)

        # Flush remaining paragraphs
        if current_chunk_paragraphs:
            combined_text = "\n\n".join(current_chunk_paragraphs)
            chunks.append(emit_chunk(combined_text, chunk_counter))

    logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


class LocalVectorStore:
    """NumPy-based vector store for dense retrieval."""

    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]):
        """Initialize store with 2D vectors array (shape N, D) and matching chunks."""
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.chunks = chunks

        if len(self.chunks) != len(self.vectors):
            raise ValueError(
                f"Mismatch between chunks count ({len(self.chunks)}) and vectors count ({len(self.vectors)})"
            )

    def save(self, index_dir: Path | str | None = None) -> Path:
        """Persist vectors (.npy) and chunk metadata (.json) to disk."""
        target_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        vectors_path = target_dir / "vectors.npy"
        chunks_path = target_dir / "chunks.json"

        np.save(vectors_path, self.vectors)
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        chunks_path.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

        logger.info(f"Saved vector index ({len(self.chunks)} chunks) to '{target_dir}'")
        return target_dir

    @classmethod
    def load(cls, index_dir: Path | str | None = None) -> "LocalVectorStore":
        """Load vector store from disk."""
        target_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        vectors_path = target_dir / "vectors.npy"
        chunks_path = target_dir / "chunks.json"

        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Vector index not found in '{target_dir}'. Run index build first."
            )

        vectors = np.load(vectors_path)
        chunks_raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in chunks_raw]

        return cls(vectors=vectors, chunks=chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
        client: genai.Client | None = None,
        use_mock: bool | None = None,
    ) -> RetrievalResult:
        """Retrieve most relevant chunks for query, enforcing similarity thresholding.

        Args:
            query: Natural language research query.
            top_k: Maximum number of relevant chunks to retrieve.
            threshold: Minimum cosine similarity required. Defaults to Settings.retrieval_threshold.
            client: Optional GenAI client instance.
            use_mock: Optional flag to force deterministic embedder.

        Returns:
            RetrievalResult: Contains retrieved chunks, best score, and insufficient_context flag.
        """
        settings = get_settings(require_api_key=False)
        effective_threshold = threshold if threshold is not None else settings.retrieval_threshold

        if len(self.chunks) == 0:
            logger.info(
                f"[RETRIEVAL]\nQuery: {query}\nTop result: None\nSimilarity: 0.0000\nSource: None"
            )
            return RetrievalResult(
                query=query,
                chunks=[],
                best_similarity=0.0,
                threshold=effective_threshold,
                insufficient_context=True,
            )

        # Generate query embedding
        query_vec = np.array(
            embed_text(query, client=client, use_mock=use_mock),
            dtype=np.float32,
        )

        # Compute cosine similarity across all stored vectors
        similarities = cosine_similarity(self.vectors, query_vec)

        # Rank indices in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[:top_k]

        best_score = float(similarities[sorted_indices[0]]) if len(sorted_indices) > 0 else 0.0
        best_chunk = self.chunks[sorted_indices[0]] if len(sorted_indices) > 0 else None

        insufficient_context = best_score < effective_threshold

        # Log trace in specified format
        logger.info(
            f"[RETRIEVAL]\n"
            f"Query: {query}\n"
            f"Top result: {best_chunk.chunk_id if best_chunk else 'None'}\n"
            f"Similarity: {best_score:.4f}\n"
            f"Source: {best_chunk.source if best_chunk else 'None'}\n"
            f"Threshold: {effective_threshold:.4f} (Insufficient: {insufficient_context})"
        )

        if insufficient_context:
            # Reject context rather than returning irrelevant material
            return RetrievalResult(
                query=query,
                chunks=[],
                best_similarity=best_score,
                threshold=effective_threshold,
                insufficient_context=True,
            )

        # Filter top-k results that meet or exceed the threshold
        qualifying_chunks: list[RetrievedChunk] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= effective_threshold:
                chunk = self.chunks[idx]
                qualifying_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        source=chunk.source,
                        text=chunk.text,
                        similarity=round(score, 4),
                    )
                )

        return RetrievalResult(
            query=query,
            chunks=qualifying_chunks,
            best_similarity=round(best_score, 4),
            threshold=effective_threshold,
            insufficient_context=False,
        )


def build_index(
    kb_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
    client: genai.Client | None = None,
    use_mock: bool | None = None,
) -> LocalVectorStore:
    """Build and persist the local NumPy vector store from knowledge base files.

    Args:
        kb_dir: Path to directory containing knowledge base markdown files.
        index_dir: Path to directory where vector files will be stored.
        client: Optional Gemini client.
        use_mock: Force deterministic embedding if True.

    Returns:
        LocalVectorStore: Initialized vector store.
    """
    logger.info("Starting index build...")
    documents = load_documents(kb_dir)
    if not documents:
        raise ValueError("No documents found in knowledge base directory to index.")

    chunks = chunk_documents(documents)
    texts = [chunk.text for chunk in chunks]

    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    vectors_list = embed_texts(texts, client=client, use_mock=use_mock)
    vectors = np.array(vectors_list, dtype=np.float32)

    store = LocalVectorStore(vectors=vectors, chunks=chunks)
    store.save(index_dir)
    return store


if __name__ == "__main__":
    from app.logging_config import setup_logging
    setup_logging("INFO")

    parser = argparse.ArgumentParser(description="Build or test ResearchPilot vector store index.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector index from knowledge base")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock embeddings for offline testing")
    parser.add_argument("--query", type=str, default=None, help="Optional test query to run after loading")
    args = parser.parse_args()

    if args.rebuild or not (DEFAULT_INDEX_DIR / "vectors.npy").exists():
        build_index(use_mock=True if args.mock else None)
    else:
        logger.info("Index already exists. Use --rebuild to regenerate.")

    if args.query:
        store = LocalVectorStore.load()
        res = store.retrieve(args.query, use_mock=True if args.mock else None)
        print(f"\nResult for {args.query!r}:")
        print(f"Insufficient Context: {res.insufficient_context}")
        print(f"Best Similarity: {res.best_similarity}")
        for c in res.chunks:
            print(f"- [{c.source}] ({c.similarity}): {c.text[:80]}...")
