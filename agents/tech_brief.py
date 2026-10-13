"""bk의 기술 검색/원문 검증을 자료 수집 계약으로 연결한다. 점수를 산정하지 않는다."""
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit
from pydantic import Field, field_validator
from .collection import collect, web_search
from .contracts import Contract
from prompts.tech_brief_prompt import TECH_BRIEF_PROMPT


class RetrievedDocument(Contract):
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    content: str = ''
    source_title: str = ''
    page: int | None = Field(default=None, ge=1, strict=True)
    source_url: str | None = None
    retrieval_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator('source_url')
    @classmethod
    def valid_url(cls, value):
        if value is not None:
            parsed = urlsplit(value)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname:
                raise ValueError('source_url requires HTTP(S)')
        return value


def adapt_retrieval_result(raw):
    if not isinstance(raw, (list, tuple)):
        raise ValueError('검색 결과는 문서 목록이어야 합니다.')
    return [RetrievedDocument.model_validate(d.model_dump() if isinstance(d, RetrievedDocument) else d) for d in raw]


def source_type(doc):
    explicit = doc.metadata.get('source_type')
    if explicit is not None:
        if explicit not in ('pdf', 'web', 'input'):
            raise ValueError('Invalid source_type')
        return explicit
    if doc.page is not None or doc.source_title.lower().endswith('.pdf') or (doc.source_url and urlsplit(doc.source_url).path.lower().endswith('.pdf')):
        return 'pdf'
    return 'web' if doc.source_url else 'input'


def prepare_documents(documents, company_name):
    """기업 필터, 페이지/URL, 원문, 청크 중복·충돌을 검증한다."""
    registry, missing = {}, []
    for doc in documents:
        if doc.metadata.get('company_name', company_name) != company_name:
            missing.append('다른 기업의 기술 문서 제외')
            continue
        if not doc.content.strip() or not doc.source_title.strip():
            missing.append('기술 문서 원문 또는 출처명 없음')
            continue
        kind = source_type(doc)
        if kind == 'pdf' and doc.page is None:
            missing.append('기술 PDF 원본 페이지 없음')
            continue
        if kind == 'web' and not doc.source_url:
            missing.append('기술 웹 자료 URL 없음')
            continue
        identity = (doc.document_id, doc.chunk_id, doc.page)
        if identity in registry:
            if registry[identity].model_dump(exclude={'retrieval_score'}) != doc.model_dump(exclude={'retrieval_score'}):
                raise ValueError('동일 기술 청크에 서로 다른 원문 또는 출처')
            continue
        registry[identity] = doc
    return list(registry.values()), list(dict.fromkeys(missing))


@lru_cache(maxsize=1)
def default_retriever():
    from rag.product_tech_retriever import ProductTechRAGRetriever
    return ProductTechRAGRetriever(top_k=3)


def make_searchers(company_name, *, retriever=None, adapter=adapt_retrieval_result, web_backend=None):
    """PDF 실패 시에도 웹 조사 가능. Supervisor의 실제 추가 질의를 두 검색기에 전달한다."""
    seen_documents = []
    def tech_documents(query):
        selected = retriever if retriever is not None else default_retriever()
        raw = selected.search(company_name=company_name, queries=[query])
        incoming = adapt_retrieval_result(adapter(raw))
        documents, missing = prepare_documents(seen_documents + incoming, company_name)
        seen_documents[:] = documents
        rows = [{'_missing_item': message} for message in missing]
        for doc in documents:
            kind = source_type(doc)
            # 문서/청크 ID는 내부 식별에만 쓰며 출처·원문·페이지는 원본을 보존한다.
            source = doc.source_url if kind == 'web' else doc.metadata.get('source', doc.document_id)
            rows.append({'source_type': kind, 'source': source, 'source_title': doc.source_title,
                         'excerpt': doc.content, 'page': doc.page,
                         'provenance': doc.metadata.get('provenance', 'unknown'),
                         'document_id': doc.document_id, 'chunk_id': doc.chunk_id,
                         'raw_document': doc.model_dump(mode='json')})
        return rows
    return [tech_documents, web_backend if web_backend is not None else web_search]


def run(task, *, model=None, searchers=None, retriever=None, adapter=adapt_retrieval_result, web_backend=None):
    task = dict(task)
    # 지원 제품명이 있으면 기업 질의에 함께 넣는다. 임의 제품명은 만들지 않는다.
    products = task.get('product_names', [])
    if products:
        if any(not isinstance(p, str) or not p.strip() for p in products):
            raise ValueError('제품명은 비어 있지 않은 문자열이어야 합니다.')
        task['queries'] = [f"{' '.join(products)} {q}" for q in task['queries']]
    selected = searchers if searchers is not None else make_searchers(task['company_name'], retriever=retriever, adapter=adapter, web_backend=web_backend)
    return collect(task, prompt=TECH_BRIEF_PROMPT, model_env='TECH_MODEL', model=model, searchers=selected)


def as_node(**kwargs):
    from .collection import research_node
    return research_node('tech_brief', run, **kwargs)
