"""저장된 기업별 기술 DB를 Agent의 공통 검색 형식으로 연결한다."""
import hashlib
import json
import math
from datetime import date
from pathlib import Path

from langchain_community.vectorstores import FAISS

from .tech_ingest import COMPANIES, INDEX, PDF
from .tech_embeddings import LocalE5Embeddings


def document_to_result(doc) -> dict:
    """LangChain Document를 원문과 PDF 페이지를 보존한 딕셔너리로 바꾼다."""
    metadata = doc.metadata
    for field in ("source_title", "source"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"필수 메타데이터 누락: {field}")
    page = metadata.get("page_number")
    if page is None:
        raw_page, base = metadata.get("page"), metadata.get("page_index_base")
        if type(raw_page) is not int or base not in (0, 1):
            raise ValueError("page와 명시적인 page_index_base(0 또는 1)가 필요합니다.")
        page = raw_page + (1 if base == 0 else 0)
    if type(page) is not int or page < 1:
        raise ValueError("검색 문서에 올바른 PDF 페이지 번호가 없습니다.")
    published = metadata.get("published_date")
    if published is not None:
        if not isinstance(published, str) or date.fromisoformat(published).isoformat() != published:
            raise ValueError("published_date는 YYYY-MM-DD 형식이어야 합니다.")
    return {
        "source_title": metadata["source_title"],
        "source_type": "pdf",
        "source": metadata["source"],
        "excerpt": doc.page_content,
        "page": page,
        "published_date": published,
    }


def create_rag_search(company_id: str, top_k: int = 3, min_similarity=None,
                      index_dir=INDEX):
    """기업을 한 번 선택하고 query만 받는 검색 함수를 반환한다."""
    code = company_id.upper()
    if code not in COMPANIES or type(top_k) is not int or top_k < 1:
        raise ValueError("올바른 기업 코드와 양의 top_k가 필요합니다.")
    if min_similarity is not None and (not math.isfinite(min_similarity)
                                       or not -1 <= min_similarity <= 1):
        raise ValueError("min_similarity는 -1부터 1 사이여야 합니다.")
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("메타데이터 보완 인덱스가 필요합니다. DB를 다시 구축하세요.")
    if (manifest["model"] != "intfloat/multilingual-e5-small"
            or not manifest["normalized"] or manifest["distance"] != "squared_l2"):
        raise ValueError("임베딩 모델 또는 인덱스 거리 설정이 다릅니다.")
    if hashlib.sha256(PDF.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("PDF가 인덱스 생성 이후 변경됐습니다. DB를 다시 구축하세요.")
    embeddings = LocalE5Embeddings()
    # 직접 생성한 로컬 인덱스만 읽는다. 외부에서 받은 pickle 파일로 교체하지 않는다.
    store = FAISS.load_local(str(index_dir / code), embeddings,
                             allow_dangerous_deserialization=True)
    if (store.index.ntotal != manifest["companies"][code]
            or store.index.d != manifest["dimension"]):
        raise ValueError("저장된 벡터 개수와 manifest가 일치하지 않습니다.")
    for doc_id in store.index_to_docstore_id.values():
        doc = store.docstore.search(doc_id)
        document_to_result(doc)
        if doc.metadata.get("company_id") != code:
            raise ValueError("인덱스에 다른 기업의 문서가 섞여 있습니다.")

    def rag_search(query: str) -> list[dict]:
        """관련도순 원문 청크를 반환하며 검색 실패는 예외로 전달한다."""
        if not isinstance(query, str):
            raise TypeError("query는 문자열이어야 합니다.")
        if not query.strip():
            return []
        # 기업당 약 10~15개라 전체 후보에서 중복 제거 후 상위 k개를 선택한다.
        if store.index.ntotal == 0:
            return []
        matches = store.similarity_search_with_score(query, k=store.index.ntotal)
        # 작은 패널티만 적용해 본문을 우선하되 관련도 차이가 큰 순서는 뒤집지 않는다.
        matches.sort(key=lambda pair: float(pair[1]) + (
            0.04 if pair[0].metadata.get("content_kind") in
            ("references", "heading", "toc", "market") else 0
        ))
        results, seen = [], set()
        for doc, distance in matches:
            if doc.metadata.get("company_id") != code:
                raise ValueError("인덱스에 다른 기업의 문서가 섞여 있습니다.")
            # 정규화된 벡터의 제곱 L2 거리를 코사인 유사도로 변환한다.
            if min_similarity is not None and 1 - float(distance) / 2 < min_similarity:
                continue
            result = document_to_result(doc)
            key = (result["source"], result["page"], result["excerpt"])
            if key in seen:
                continue
            seen.add(key)
            results.append(result)
            if len(results) == top_k:
                break
        return results

    return rag_search
