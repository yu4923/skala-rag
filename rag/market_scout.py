"""Market-document RAG search interface.

This module intentionally owns only retrieval from PDFs in ``data/market``.
Company-specific web searches and investment scoring belong to the Market
Scout agent/orchestrator, not to this RAG layer.

Public API:

    initialize_market_rag()
    rag_search(query: str) -> list[dict]

The on-disk FAISS index is a disposable cache under ``market_index``.  It is
rebuilt automatically when a source PDF or an indexing setting changes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Sequence
import unicodedata


# 원본 PDF는 루트 data/market, 생성 인덱스는 rag/market_index에 둔다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/market"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag/market_index"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+", re.UNICODE)
_ENGINE: "MarketRAG | None" = None
_ENGINE_LOCK = threading.Lock()


class MarketRAGError(RuntimeError):
    """Raised when the local market-document search system cannot operate."""


@dataclass(frozen=True)
class MarketChunk:
    chunk_id: str
    source_title: str
    source_type: str
    source: str
    excerpt: str
    page_zero_based: int
    published_date: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MarketChunk":
        return cls(**value)


@dataclass(frozen=True)
class MarketRAGSettings:
    data_dir: Path = DEFAULT_DATA_DIR
    index_dir: Path = DEFAULT_INDEX_DIR
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size_tokens: int = 440
    chunk_overlap_tokens: int = 70
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    candidate_k: int = 10
    result_k: int = 3
    min_dense_score: float = 0.85
    published_date: str | None = None

    @classmethod
    def from_env(cls) -> "MarketRAGSettings":
        published_date = os.getenv("MARKET_RAG_PUBLISHED_DATE") or None
        if published_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", published_date):
            raise MarketRAGError(
                "MARKET_RAG_PUBLISHED_DATE must use the YYYY-MM-DD format."
            )

        return cls(
            data_dir=Path(os.getenv("MARKET_RAG_DATA_DIR", str(DEFAULT_DATA_DIR))),
            index_dir=Path(os.getenv("MARKET_RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR))),
            embedding_model=os.getenv(
                "MARKET_RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
            chunk_size_tokens=int(os.getenv("MARKET_RAG_CHUNK_SIZE", "440")),
            chunk_overlap_tokens=int(os.getenv("MARKET_RAG_CHUNK_OVERLAP", "70")),
            dense_weight=float(os.getenv("MARKET_RAG_DENSE_WEIGHT", "0.6")),
            sparse_weight=float(os.getenv("MARKET_RAG_SPARSE_WEIGHT", "0.4")),
            candidate_k=int(os.getenv("MARKET_RAG_CANDIDATE_K", "10")),
            result_k=int(os.getenv("MARKET_RAG_RESULT_K", "3")),
            min_dense_score=float(os.getenv("MARKET_RAG_MIN_DENSE_SCORE", "0.85")),
            published_date=published_date,
        )

    def validate(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise MarketRAGError("chunk_size_tokens must be positive.")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise MarketRAGError(
                "chunk_overlap_tokens must be non-negative and smaller than chunk_size_tokens."
            )
        if self.candidate_k <= 0 or self.result_k <= 0:
            raise MarketRAGError("candidate_k and result_k must be positive.")
        if self.result_k > self.candidate_k:
            raise MarketRAGError("result_k cannot be larger than candidate_k.")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise MarketRAGError("retrieval weights cannot be negative.")
        if self.dense_weight + self.sparse_weight == 0:
            raise MarketRAGError("at least one retrieval weight must be positive.")


class E5Embedder:
    """SentenceTransformer adapter that applies E5 query/passage prefixes."""

    def __init__(self, model_name: str) -> None:
        # macOS의 FAISS/PyTorch OpenMP 충돌을 피하면서 사용자 설정은 우선한다.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MarketRAGError(
                "sentence-transformers is required. Install requirements.txt."
            ) from exc

        self.model_name = model_name
        try:
            try:
                self.model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise MarketRAGError(
                f"failed to load embedding model {model_name!r}: {exc}"
            ) from exc
        self.tokenizer = self.model.tokenizer

    def embed_documents(self, texts: Sequence[str]):
        return self.model.encode(
            [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    def embed_query(self, query: str):
        return self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")


class BM25Index:
    """Small, dependency-free BM25 index for Korean/English market terms."""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokens = [_lexical_tokens(text) for text in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        count = len(self.tokens)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.tokens:
            return []
        query_tokens = _lexical_tokens(query)
        if not query_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            document_length = self.lengths[index]
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b
                    * document_length
                    / (self.average_length or 1.0)
                )
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            if score > 0:
                scores.append((index, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:k]


class MarketRAG:
    def __init__(
        self,
        settings: MarketRAGSettings,
        *,
        embedder: E5Embedder | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.embedder = embedder
        self.chunks: list[MarketChunk] = []
        self.faiss_index: Any = None
        self.bm25: BM25Index | None = None

    @property
    def _faiss_path(self) -> Path:
        return self.settings.index_dir / "market.faiss"

    @property
    def _chunks_path(self) -> Path:
        return self.settings.index_dir / "market_chunks.json"

    @property
    def _manifest_path(self) -> Path:
        return self.settings.index_dir / "market_manifest.json"

    def initialize(self, *, force_rebuild: bool = False) -> "MarketRAG":
        pdf_paths = _find_pdf_paths(self.settings.data_dir)
        manifest = _build_manifest(pdf_paths, self.settings)
        self.settings.index_dir.mkdir(parents=True, exist_ok=True)

        if not force_rebuild and self._cache_matches(manifest):
            self._load_cache()
        else:
            self._build_index(pdf_paths)
            self._save_cache(manifest)

        self.bm25 = BM25Index([chunk.excerpt for chunk in self.chunks])
        return self

    def search(self, query: str) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        if self.faiss_index is None or self.bm25 is None or not self.chunks:
            raise MarketRAGError("market RAG is not initialized.")

        query_vector = self._get_embedder().embed_query(query)
        dense_scores, dense_indices = self.faiss_index.search(
            query_vector, min(self.settings.candidate_k, len(self.chunks))
        )
        dense = [
            (int(index), float(score))
            for index, score in zip(dense_indices[0], dense_scores[0])
            if int(index) >= 0
        ]
        sparse = self.bm25.search(query, self.settings.candidate_k)

        if not dense and not sparse:
            return []

        dense_is_relevant = bool(
            dense and dense[0][1] >= self.settings.min_dense_score
        )
        query_terms = set(_lexical_tokens(query))
        sparse_is_relevant = bool(
            query_terms
            and any(
                len(query_terms.intersection(self.bm25.tokens[index]))
                / len(query_terms)
                >= 0.5
                for index, _score in sparse
            )
        )
        if not dense_is_relevant and not sparse_is_relevant:
            return []

        ranked_indices = _reciprocal_rank_fusion(
            dense,
            sparse,
            dense_weight=self.settings.dense_weight,
            sparse_weight=self.settings.sparse_weight,
        )

        results: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for index in ranked_indices:
            chunk = self.chunks[index]
            duplicate_key = (
                chunk.source,
                chunk.page_zero_based,
                _normalise_for_deduplication(chunk.excerpt),
            )
            if duplicate_key in seen:
                continue
            seen.add(duplicate_key)
            results.append(_to_agent_result(chunk))
            if len(results) >= self.settings.result_k:
                break
        return results

    def _get_embedder(self) -> E5Embedder:
        if self.embedder is None:
            self.embedder = E5Embedder(self.settings.embedding_model)
        return self.embedder

    def _build_index(self, pdf_paths: Sequence[Path]) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise MarketRAGError(
                "faiss-cpu is required. Install requirements.txt."
            ) from exc

        embedder = self._get_embedder()
        self.chunks = _load_and_chunk_pdfs(pdf_paths, self.settings, embedder.tokenizer)
        if not self.chunks:
            raise MarketRAGError("no searchable text was extracted from data PDFs.")

        vectors = embedder.embed_documents([chunk.excerpt for chunk in self.chunks])
        if len(vectors.shape) != 2 or vectors.shape[0] != len(self.chunks):
            raise MarketRAGError("embedding output shape does not match the chunk count.")
        self.faiss_index = faiss.IndexFlatIP(int(vectors.shape[1]))
        self.faiss_index.add(vectors)

    def _cache_matches(self, expected_manifest: dict[str, Any]) -> bool:
        required = (self._faiss_path, self._chunks_path, self._manifest_path)
        if not all(path.is_file() for path in required):
            return False
        try:
            actual = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return actual == expected_manifest

    def _save_cache(self, manifest: dict[str, Any]) -> None:
        try:
            import faiss

            faiss.write_index(self.faiss_index, str(self._faiss_path))
            self._chunks_path.write_text(
                json.dumps(
                    [asdict(chunk) for chunk in self.chunks],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            raise MarketRAGError(f"failed to save the market RAG cache: {exc}") from exc

    def _load_cache(self) -> None:
        try:
            import faiss

            self.faiss_index = faiss.read_index(str(self._faiss_path))
            raw_chunks = json.loads(self._chunks_path.read_text(encoding="utf-8"))
            self.chunks = [MarketChunk.from_dict(value) for value in raw_chunks]
        except Exception as exc:
            raise MarketRAGError(f"failed to load the market RAG cache: {exc}") from exc
        if self.faiss_index.ntotal != len(self.chunks):
            raise MarketRAGError("cached FAISS index and chunk metadata are inconsistent.")


def initialize_market_rag(
    *,
    force_rebuild: bool = False,
    settings: MarketRAGSettings | None = None,
) -> MarketRAG:
    """Initialize and cache the market RAG engine once per process."""

    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None or force_rebuild or settings is not None:
            resolved_settings = settings or MarketRAGSettings.from_env()
            try:
                _ENGINE = MarketRAG(resolved_settings).initialize(
                    force_rebuild=force_rebuild
                )
            except MarketRAGError:
                raise
            except Exception as exc:
                raise MarketRAGError(f"failed to initialize market RAG: {exc}") from exc
    return _ENGINE


def rag_search(query: str) -> list[dict]:
    """Return up to three relevant PDF chunks in the Agent contract format.

    An empty/blank query or a search with no sufficiently relevant result
    returns ``[]``.  Operational failures raise ``MarketRAGError`` so the
    caller can append them to the Agent's error history.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query.strip():
        return []
    engine = _ENGINE or initialize_market_rag()
    try:
        return engine.search(query)
    except MarketRAGError:
        raise
    except Exception as exc:
        raise MarketRAGError(f"market RAG search failed: {exc}") from exc


def _find_pdf_paths(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise MarketRAGError(f"market data directory does not exist: {data_dir}")
    pdf_paths = sorted(
        (path for path in data_dir.glob("*.pdf") if path.is_file()),
        key=lambda path: path.name,
    )
    if not pdf_paths:
        raise MarketRAGError(f"no PDF files found in market data directory: {data_dir}")
    return pdf_paths


def _load_and_chunk_pdfs(
    pdf_paths: Sequence[Path],
    settings: MarketRAGSettings,
    tokenizer: Any,
) -> list[MarketChunk]:
    try:
        import pymupdf
    except ImportError as exc:
        raise MarketRAGError(
            "PyMuPDF is required. Install requirements.txt."
        ) from exc

    chunks: list[MarketChunk] = []
    for pdf_path in pdf_paths:
        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise MarketRAGError(f"failed to open PDF {pdf_path}: {exc}") from exc

        relative_source = _relative_source_path(pdf_path)
        try:
            for page_number in range(document.page_count):
                page_text = _clean_extracted_text(
                    document.load_page(page_number).get_text("text")
                )
                if not page_text or not page_text.strip():
                    continue
                page_chunks = _split_with_token_offsets(
                    page_text,
                    tokenizer,
                    chunk_size=settings.chunk_size_tokens,
                    overlap=settings.chunk_overlap_tokens,
                )
                for chunk_number, excerpt in enumerate(page_chunks):
                    chunks.append(
                        MarketChunk(
                            chunk_id=(
                                f"{sha256(relative_source.encode('utf-8')).hexdigest()[:10]}-"
                                f"{page_number + 1}-{chunk_number + 1}"
                            ),
                            source_title=unicodedata.normalize("NFC", pdf_path.stem),
                            source_type="pdf",
                            source=relative_source,
                            excerpt=excerpt,
                            page_zero_based=page_number,
                            published_date=settings.published_date,
                        )
                    )
        finally:
            document.close()
    return chunks


def _split_with_token_offsets(
    text: str,
    tokenizer: Any,
    *,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split one page by model tokens while returning exact source substrings."""

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoded["offset_mapping"]
    except Exception as exc:
        raise MarketRAGError(f"tokenizer does not provide offset mappings: {exc}") from exc
    if not offsets:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for token_start in range(0, len(offsets), step):
        token_end = min(token_start + chunk_size, len(offsets))
        start_character = int(offsets[token_start][0])
        end_character = int(offsets[token_end - 1][1])
        excerpt = text[start_character:end_character].strip()
        if excerpt:
            chunks.append(excerpt)
        if token_end >= len(offsets):
            break
    return chunks


def _build_manifest(
    pdf_paths: Sequence[Path], settings: MarketRAGSettings
) -> dict[str, Any]:
    return {
        "version": 2,
        "embedding_model": settings.embedding_model,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "published_date": settings.published_date,
        "documents": [
            {
                "source": _relative_source_path(path),
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in pdf_paths
        ],
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_source_path(path: Path) -> str:
    try:
        value = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        value = path.resolve().as_posix()
    return unicodedata.normalize("NFC", value)


def _clean_extracted_text(text: str) -> str:
    """Remove conversion-only spacing artifacts without rewriting content."""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _lexical_tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _reciprocal_rank_fusion(
    dense: Sequence[tuple[int, float]],
    sparse: Sequence[tuple[int, float]],
    *,
    dense_weight: float,
    sparse_weight: float,
    rank_constant: int = 60,
) -> list[int]:
    fused: dict[int, float] = {}
    for weight, ranking in ((dense_weight, dense), (sparse_weight, sparse)):
        for rank, (index, _score) in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + weight / (rank_constant + rank)
    return [
        index
        for index, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    ]


def _normalise_for_deduplication(text: str) -> str:
    return " ".join(text.split()).casefold()


def _to_agent_result(chunk: MarketChunk) -> dict[str, Any]:
    return {
        "source_title": chunk.source_title,
        "source_type": "pdf",
        "source": chunk.source,
        "excerpt": chunk.excerpt,
        "page": chunk.page_zero_based + 1,
        "published_date": chunk.published_date,
    }


__all__ = [
    "MarketRAG",
    "MarketRAGError",
    "MarketRAGSettings",
    "initialize_market_rag",
    "rag_search",
]
