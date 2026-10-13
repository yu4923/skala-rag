"""세 자료 수집 Agent가 공통으로 사용하는 조사 State.

설계 원칙:
- State는 제어/재시도 및 조정에 필요한 제한된 관점 요약만 보관한다.
- 전체 원문·발췌·계획 이력·결정 로그는 외부 파일에 두고 trace_id로 상관시킨다.
- findings 최대 6개(각 300자), sources 최대 12개, missing_items 최대 6개(각 300자).
- scores/evaluations/투자 추천은 조사 산출물에 포함하지 않는다.
"""
from typing import Annotated, Literal
from uuid import uuid4
from typing_extensions import TypedDict


class PerspectiveResult(TypedDict):
    company_name: str
    view: str
    findings: list[str]
    sources: list[str]  # 실제 URL 또는 PDF 파일명/원본 페이지
    missing_items: list[str]
    result_ref: str    # 전체 Collection 외부 저장 참조


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
                 max_retries=max_retries, node_status={}, node_attempts={}, node_errors={}, last_error=None)
