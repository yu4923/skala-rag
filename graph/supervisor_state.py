"""Supervisor Graph의 작업 페이로드, 제어 메타데이터, 결과 계약."""
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


NodeStatus = Literal["pending", "running", "success", "partial", "error", "stopped"]




class ResultModel(BaseModel):
   model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)




class PerspectiveResult(ResultModel):
   view: str = Field(min_length=1)
   findings: str = Field(min_length=1)
   sources: list[str] = Field(default_factory=list)  # 아래 evidence의 evidence_id




class StateEvidence(ResultModel):
   evidence_id: str
   excerpt: str                       # 검색 원문; LLM이 재작성한 문장이 아님
   source_title: str
   source_url: str
   checked_at: str                    # 검색 확인일, 발행일이 아님




class FounderResearchResult(PerspectiveResult):
   company_name: str
   evidence: list[StateEvidence] = Field(default_factory=list)
   risks: list[str] = Field(default_factory=list)
   missing_items: list[str] = Field(default_factory=list)
   search_issues: list[str] = Field(default_factory=list)
   needs_more_information: bool = False




class EvalVerdict(ResultModel):
   """Supervisor/평가 노드만 기록하는 종합 판단. 창업자 노드는 작성하지 않는다."""
   decision: Literal["pending", "additional_research", "recommend", "hold", "reject"]
   reason: str
   missing_items: list[str] = Field(default_factory=list)




def merge_by_node(current: dict, update: dict) -> dict:
   """각 노드는 자기 키만 갱신한다. 재조사 결과는 최신 값으로 교체한다.


   서로 다른 노드의 병렬 갱신은 안전하다. 같은 노드를 중복 fan-out하지 않는다.
   """
   return {**current, **update}




class State(TypedDict):
   # 작업 페이로드. 모델/검색기, 메시지 이력, 검색 원장, 결정 로그는 넣지 않는다.
   company_name: str
   founder_names: list[str]
   selected_tech: list[str]
   target_domain: str
   research_request: str
   perspectives: Annotated[dict[str, PerspectiveResult | FounderResearchResult], merge_by_node]
   synthesis: str
   report: str
   eval_result: EvalVerdict | None


   # 제어 메타데이터. scalar 필드는 Supervisor만 갱신한다.
   trace_id: str
   step_count: int                     # Supervisor 방문 횟수, 분기 전에 증가
   max_steps: int
   max_retries: int                    # 최초 호출 이후 허용 재시도 횟수
   node_status: Annotated[dict[str, NodeStatus], merge_by_node]
   node_attempts: Annotated[dict[str, int], merge_by_node]
   node_errors: Annotated[dict[str, str | None], merge_by_node]
   last_error: str | None              # Supervisor가 node_errors를 보고 갱신




def initial_state(company_name: str, *, research_request: str,
                 founder_names: list[str] | None = None,
                 selected_tech: list[str] | None = None,
                 target_domain: str = "AI 스타트업", max_steps: int = 12,
                 max_retries: int = 2, trace_id: str | None = None) -> State:
   if not company_name.strip() or not research_request.strip():
       raise ValueError("company_name과 research_request가 필요합니다.")
   if max_steps < 1 or max_retries < 0:
       raise ValueError("max_steps >= 1, max_retries >= 0이어야 합니다.")
   return State(company_name=company_name.strip(), founder_names=list(founder_names or []),
                selected_tech=list(selected_tech or []), target_domain=target_domain,
                research_request=research_request, perspectives={}, synthesis="", report="",
                eval_result=None, trace_id=trace_id or str(uuid4()), step_count=0,
                max_steps=max_steps, max_retries=max_retries, node_status={},
                node_attempts={}, node_errors={}, last_error=None)


# 기존 분리본에서 사용한 이름의 호환 별칭
FounderEvidence = StateEvidence
