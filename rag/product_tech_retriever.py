"""담당자 ProductTechAgent의 search 프로토콜에 기존 기술 검색기를 연결한다."""
import hashlib
import json

from .tech_ingest import COMPANIES, INDEX
from .tech_search import create_rag_search
from .tech_embeddings import LocalE5Embeddings


def resolve_company(company_name: str) -> str:
    """정식 기업명·지원 별칭·기업 코드를 하나의 인덱스 코드로 변환한다."""
    if not isinstance(company_name, str) or not company_name.strip():
        raise ValueError("company_name에 기업명이 필요합니다.")
    normalized = company_name.strip().replace(" ", "").casefold()
    aliases = {"인코어드테크놀로지스": "EC", "크로커스에너지": "CR",
               "leecell": "LC", "haezoom": "HZ", "enlighten": "EN",
               "60hertz": "SH", "synergy": "SY", "encored": "EC",
               "vpplab": "VP", "vgen": "VG", "energyx": "EX", "crocusenergy": "CR"}
    for code, name in COMPANIES.items():
        aliases[code.casefold()] = code
        aliases[name.casefold()] = code
    if normalized not in aliases:
        raise ValueError(f"지원하지 않는 기업명: {company_name}")
    return aliases[normalized]


def to_agent_document(result: dict, company_name: str, company_id: str) -> dict:
    """공통 검색 결과를 담당자의 RetrievedDocument 필드로 변환한다."""
    # 원문·문서·페이지가 같으면 질의가 달라도 같은 청크 ID를 사용한다.
    identity = json.dumps([result["source"], result["page"], result["excerpt"]],
                          ensure_ascii=False, separators=(",", ":"))
    return {
        "document_id": result["source"],
        "chunk_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "content": result["excerpt"],
        "source_title": result["source_title"],
        "page": result["page"],
        "source_url": None,
        "metadata": {
            "source_type": "pdf",
            "source": result["source"],
            "company_id": company_id,
            "company_name": company_name,
            "canonical_company_name": COMPANIES[company_id],
            "published_date": result["published_date"],
        },
    }


class ProductTechRAGRetriever:
    """기업별 검색기를 보관하고 복수 질의를 처리하는 Agent 주입 객체다."""

    def __init__(self, *, top_k=3, min_similarity=None, index_dir=INDEX):
        """벡터 DB 위치와 질의별 반환 개수를 설정한다."""
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.index_dir = index_dir
        self._searchers = {}

    def search(self, *, company_name: str, queries: list[str]) -> list[dict]:
        """기업 필터를 적용하고 질의별 상위 결과를 중복 없이 합친다."""
        code = resolve_company(company_name)
        if not isinstance(queries, list) or any(not isinstance(q, str) for q in queries):
            raise TypeError("queries는 문자열 목록이어야 합니다.")
        queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        if not queries:
            return []
        if code not in self._searchers:
            self._searchers[code] = create_rag_search(
                code, top_k=self.top_k, min_similarity=self.min_similarity,
                index_dir=self.index_dir,
            )
        results, seen = [], set()
        vectors = LocalE5Embeddings().embed_queries(queries)
        for query, vector in zip(queries, vectors, strict=True):
            for result in self._searchers[code](query, query_vector=vector):
                document = to_agent_document(result, company_name, code)
                key = (document["document_id"], document["chunk_id"], document["page"])
                if key not in seen:
                    seen.add(key)
                    results.append(document)
        return results
