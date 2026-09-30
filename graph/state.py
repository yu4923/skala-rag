"""그래프에서 공유하는 State 및 내부 데이터 타입."""

from operator import add
from typing import Annotated, Literal
from typing_extensions import NotRequired, TypedDict

# ──────────────────────────────────────
# 1. 공통 타입
# ──────────────────────────────────────

AgentName = Literal[
    "founder_insight",
    "market_scout",
    "tech_brief",
]

Decision = Literal[
    "pending",
    "recommend",
    "hold",
    "reject",
]

RetryReason = Literal[
    "initial",
    "insufficient_evidence",
    "criteria_not_met",
]

StopReason = Literal[
    "in_progress",
    "criteria_met",
    "insufficient_evidence",
    "criteria_not_met",
]


# ──────────────────────────────────────
# 2. 기업 정보 및 재시도
# ──────────────────────────────────────

class CompanyContext(TypedDict):
    company_candidates: list[str]
    company_index: int
    company_name: str


class RetryState(TypedDict):
    max_retries: int
    retry_count: int
    retry_reason: RetryReason


# ──────────────────────────────────────
# 3. 전문 Agent의 근거 및 평가 결과
# ──────────────────────────────────────

class Evidence(TypedDict):
    evidence_id: str
    claim: str                         # 평가·요약에 사용한 주장
    excerpt: str                       # 주장을 뒷받침하는 원문 발췌
    source_type: Literal["pdf", "web", "input"]
    source: str                        # 문서 ID, URL 또는 입력자료 ID
    page: NotRequired[int]             # 페이지가 있는 자료에만 사용


class CriterionScore(TypedDict):
    criterion: str                     # 평가 항목명
    score: float                       # 배점으로 환산된 점수
    max_score: float                    # 해당 항목의 최대 배점
    reason: str                        # 점수 산정 이유
    evidence_ids: list[str]             # 사용한 Evidence의 ID


class AgentResult(TypedDict):
    company_name: str
    round_no: int                      # 최초 평가 1, 재평가 2
    agent: AgentName
    status: Literal["success", "partial", "error"]

    summary: str
    evidence: list[Evidence]
    missing_items: list[str]

    scores: list[CriterionScore]
    score: NotRequired[float]          # 담당 항목을 모두 채점한 경우의 합계


# ──────────────────────────────────────
# 4. 전문 평가 통합 및 근거 검토
# ──────────────────────────────────────

class EvidenceReview(TypedDict):
    company_name: str
    round_no: int

    integrated_summary: str
    evidence_sufficient: bool
    reason: str                        # 근거 충분·부족 판단 이유
    missing_items: list[str]            # 추가로 확인할 정보
    conflicts: list[str]                # 자료 간 모순 또는 불일치


# ──────────────────────────────────────
# 5. 종합 투자 판단
# ──────────────────────────────────────

class InvestmentResult(TypedDict):
    company_name: str
    round_no: int

    total_score: float
    criteria_met: bool
    decision_reason: str


class EvaluationStatus(TypedDict):
    decision: Decision
    stop_reason: StopReason


# ──────────────────────────────────────
# 6. 오류 및 분기 이력
# ──────────────────────────────────────

class ErrorRecord(TypedDict):
    company_name: str
    round_no: int
    node: str
    code: str                          # SEARCH_FAILED, API_ERROR 등
    message: str


class RouteRecord(TypedDict):
    company_name: str
    round_no: int
    gate: Literal["evidence", "investment"]
    outcome: bool                      # 충분 여부 또는 기준 충족 여부
    next_node: str
    reason: str


# ──────────────────────────────────────
# 7. 평가가 종료된 기업별 결과
# ──────────────────────────────────────

class CompanyEvaluation(TypedDict):
    company_name: str
    round_no: int
    evaluation_status: EvaluationStatus

    founder_result: NotRequired[AgentResult]
    market_result: NotRequired[AgentResult]
    tech_result: NotRequired[AgentResult]
    evidence_review: NotRequired[EvidenceReview]

    # 근거 부족으로 종합 투자 판단을 수행하지 않았다면 생략
    investment_result: NotRequired[InvestmentResult]


# ──────────────────────────────────────
# 8. 전체 Graph가 공유하는 State
# ──────────────────────────────────────

class InvestmentState(TypedDict):
    # 평가 시작 시 준비
    company_context: CompanyContext
    evaluation_request: str
    retry_state: RetryState
    evaluation_status: EvaluationStatus

    # 각 Node 실행 후 생성
    founder_result: NotRequired[AgentResult]
    market_result: NotRequired[AgentResult]
    tech_result: NotRequired[AgentResult]

    evidence_review: NotRequired[EvidenceReview]
    investment_result: NotRequired[InvestmentResult]

    # 누적 기록
    errors: Annotated[list[ErrorRecord], add]
    route_history: Annotated[list[RouteRecord], add]
    company_results: Annotated[list[CompanyEvaluation], add]

    # 최종 보고서
    report: NotRequired[str]


