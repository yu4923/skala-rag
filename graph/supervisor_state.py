"""세 수집 Agent 공통 State.

설계 원칙:
- 조사 payload는 perspectives에 직접 보관하고 관점별 최신 결과로 교체한다.
- 제어/재시도 정보와 현재 최종 verdict만 유지한다. 로그/계획 이력은 누적하지 않는다.
- 관측성은 logger/콜백으로 분리하고 trace_id로 연결한다. 파일 저장은 하지 않는다.
- 점수/투자 추천을 생성하지 않는다. 메모리 State는 프로세스 종료 시 사라진다.
"""
from typing import Annotated, Literal
from uuid import uuid4
from datetime import datetime
from zoneinfo import ZoneInfo
from typing_extensions import TypedDict


class PerspectiveResult(TypedDict):
    # agents.contracts.Collection.model_dump()와 동일한 공통 산출물 계약.
    company_name: str
    agent: str
    findings: list[dict]
    evidence: list[dict]  # 실제 URL/파일·페이지·검색 원문
    missing_items: list[str]
    queries_used: list[str]
    search_errors: list[str]
    raw_documents: dict[str, dict]  # 원본 검색 데이터와 질의. 보고서 근거 검토용


NodeStatus = Literal['pending', 'running', 'success', 'partial', 'error', 'stopped']


def merge_by_node(current: dict | None, update: dict | None) -> dict:
    # 각 Worker는 자기 관점 키만 갱신한다. 동일 관점의 중복 fan-out은 금지한다.
    merged = dict(current or {})
    for key, value in (update or {}).items():
        if value is None and key in merged:
            merged.pop(key)
        elif value is not None:
            merged[key] = value
    return merged


def merge_errors(current: dict | None, update: dict | None) -> dict:
    # None은 오류 해제를 뜻한다. 오류 메시지 본문/stack trace는 저장하지 않는다.
    return {**(current or {}), **(update or {})}


class State(TypedDict):
    company_name: str
    research_request: str
    product_names: list[str]
    research_tasks: dict[str, dict]  # 현재 실행에 필요한 배정 질의/지시만
    perspectives: Annotated[dict[str, PerspectiveResult], merge_by_node]
    trace_id: str
    step_count: int
    max_steps: int
    max_retries: int
    node_status: Annotated[dict[str, NodeStatus], merge_by_node]
    node_attempts: Annotated[dict[str, int], merge_by_node]
    node_errors: Annotated[dict[str, str | None], merge_errors]
    last_error: str | None
    action: Literal["collect", "write", "review", "finish", "blocked"]
    status: Literal["running", "finished", "blocked"]
    evidence_sufficient: bool
    report_date: str
    report: dict | None
    report_revision: int
    report_review: dict | None
    review_revision: int | None
    verdict: dict | None  # 현재 최종 판단. 전체 결정/계획 이력을 저장하지 않는다.


ResearchState = State


def initial_state(company_name: str, *, research_request: str,
                  product_names: list[str] | None = None, max_steps: int = 12,
                  max_retries: int = 2, trace_id: str | None = None) -> State:
    if not company_name.strip() or not research_request.strip():
        raise ValueError('company_name과 research_request가 필요합니다.')
    if max_steps < 1 or max_retries < 0:
        raise ValueError('max_steps >= 1, max_retries >= 0이어야 합니다.')
    return State(company_name=company_name.strip(), research_request=research_request,
                 product_names=list(product_names or []), research_tasks={}, perspectives={},
                 trace_id=trace_id or str(uuid4()), step_count=0, max_steps=max_steps,
                 max_retries=max_retries, node_status={}, node_attempts={}, node_errors={}, last_error=None, action="collect",
                 status="running", evidence_sufficient=False, verdict=None,
                 report_date=datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
                 report=None, report_revision=0, report_review=None, review_revision=None)
