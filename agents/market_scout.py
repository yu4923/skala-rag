"""시장성 검증: 시장/정책 RAG + 웹 + 성장률 계산 -> 공통 AgentResult.

모델 및 검색기는 외부에서 주입한다. import/직접 실행으로 외부 API를 호출하지 않는다.
시장 규모·성장 가능성과 실제 고객 수요를 각각 25점 만점으로 평가한다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from decimal import Decimal
from itertools import islice
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlsplit

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "agents"

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import Field, model_validator

from prompts import market_scout_prompt
from .evaluation_support import (
    CRITERIA, AgentResult, ContractModel, CriterionEvaluation, Evidence,
)

# 프로젝트 .env 또는 상위 디렉터리의 env 파일을 읽는다. 기존 환경변수가 우선한다.
def _load_env() -> None:
    project = Path(__file__).resolve().parents[1]
    for path in (project / ".env", project.parent / "env"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"\''))
            break


_load_env()

logger = logging.getLogger(__name__)
MARKET_CRITERIA = {
    "시장 규모·성장 가능성": float(os.environ["MARKET_GROWTH_WEIGHT"]),
    "실제 고객 수요": float(os.environ["CUSTOMER_DEMAND_WEIGHT"]),
}
CRITERIA["market"].update(MARKET_CRITERIA)
MODEL_NAME = os.environ["MARKET_MODEL"]
MODEL_TEMPERATURE = 0


class MarketSource(ContractModel):
    """검색기가 반환할 원문 청크. PDF page는 원본 기준 1부터 시작한다.

    source는 PDF 경로/문서 ID, 웹 URL 또는 입력 자료 ID이다.
    source_title은 실제 문서명이다. evidence_id는 실행기가 생성한다.
    """

    source_title: str = Field(min_length=1)
    source_type: Literal["pdf", "web", "input"]
    source: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    published_date: date | None = None

    @model_validator(mode="after")
    def validate_location(self):
        if self.source_type == "pdf" and self.page is None:
            raise ValueError("PDF 근거에는 원본 페이지가 필요합니다.")
        if self.source_type == "web":
            url = urlsplit(self.source)
            if url.scheme not in {"http", "https"} or not url.hostname or self.page is not None:
                raise ValueError("웹 근거에는 HTTP(S) URL과 page=None이 필요합니다.")
        return self


class MarketAgentInput(ContractModel):
    company_name: str = Field(min_length=1)
    evaluation_request: str = Field(min_length=1)
    company_description: str = ""
    target_market: str = Field(default="국내 B2C 에너지 시장", min_length=1)
    round_no: int = Field(default=1, ge=1, le=2)
    previous_missing_items: list[str] = Field(default_factory=list)
    input_evidence: list[MarketSource] = Field(default_factory=list)
    criteria: dict[str, float] = Field(default_factory=lambda: dict(MARKET_CRITERIA))

    @model_validator(mode="after")
    def validate_contract(self):
        if self.criteria != MARKET_CRITERIA:
            raise ValueError("시장성 평가 항목과 배점은 MARKET_CRITERIA 계약을 따릅니다.")
        if any(item.source_type != "input" for item in self.input_evidence):
            raise ValueError("input_evidence의 source_type은 input이어야 합니다.")
        return self


class MarketCitation(ContractModel):
    evidence_id: str = Field(min_length=1, description="이번 검색/입력 자료에서 받은 evidence_id")
    claim: str = Field(min_length=1)
    excerpt: str = Field(min_length=1, description="해당 원문에 실제 존재하는 연속된 발췌")
    source_type: Literal["pdf", "web", "input"]
    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class MarketScore(ContractModel):
    criterion: str = Field(min_length=1)
    score: float | None = Field(ge=0)
    max_score: float = Field(gt=0)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class MarketAssessment(ContractModel):
    """외부 MARKET_SCOUT_PROMPT가 지정한 LLM 응답 형식."""

    company_name: str = Field(min_length=1)
    round_no: int = Field(ge=1, le=2)
    agent: Literal["market_scout"]
    status: Literal["success", "partial", "error"]
    summary: str = Field(min_length=1)
    evidence: list[MarketCitation]
    missing_items: list[str] = Field(default_factory=list)
    scores: list[MarketScore] = Field(min_length=2, max_length=2)
    score: float | None = Field(ge=0, le=50)

    @model_validator(mode="after")
    def validate_scores(self):
        criteria = [item.criterion for item in self.scores]
        if len(set(criteria)) != 2 or set(criteria) != set(MARKET_CRITERIA):
            raise ValueError("시장 규모·성장 가능성/실제 고객 수요 항목이 각각 하나씩 필요합니다.")
        if any(item.max_score != MARKET_CRITERIA[item.criterion] for item in self.scores):
            raise ValueError("시장성 평가 항목의 배점이 설정과 다릅니다.")
        if any(item.score is not None and item.score > item.max_score for item in self.scores):
            raise ValueError("시장성 평가 점수가 배점을 초과합니다.")
        ids = [item.evidence_id for item in self.evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("응답의 evidence_id가 중복되었습니다.")
        for item in self.scores:
            if set(item.evidence_ids) - set(ids):
                raise ValueError("평가 항목이 응답 evidence에 없는 ID를 참조합니다.")
            if item.score is not None and not item.evidence_ids:
                raise ValueError("근거 없는 점수는 0이 아니라 null이어야 합니다.")
        if any(item.score is None for item in self.scores):
            if self.score is not None:
                raise ValueError("미평가 항목이 있으면 총점은 null입니다.")
        else:
            total = sum(Decimal(str(item.score)) for item in self.scores)
            if self.score is None or Decimal(str(self.score)) != total:
                raise ValueError("score는 항목별 점수의 합과 일치해야 합니다.")
        if self.status == "success" and (self.missing_items or self.score is None):
            raise ValueError("success에는 누락 항목이나 미평가 점수가 있을 수 없습니다.")
        return self


class MarketError(ContractModel):
    company: str
    round: int
    node: str
    error_type: str
    message: str


class MarketRun(ContractModel):
    """상태/오류 이력이 필요한 Graph용 결과. invoke()는 result만 반환한다."""

    result: AgentResult
    assessment: MarketAssessment | None = None
    errors: list[MarketError] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    calculations: list[dict[str, Any]] = Field(default_factory=list)


MarketSearch = Callable[[str], Sequence[MarketSource | Mapping[str, Any]]]


def calculate_growth(start_value: float, end_value: float, years: float) -> dict[str, float]:
    """동일 단위·지역·시장 범위의 양수 값에 대한 전체 성장률 및 CAGR(%)."""
    if not all(math.isfinite(value) and value > 0 for value in (start_value, end_value, years)):
        raise ValueError("시작값, 종료값, 기간은 유한한 양수여야 합니다.")
    try:
        growth = (end_value / start_value - 1) * 100
        cagr = math.expm1((math.log(end_value) - math.log(start_value)) / years) * 100
    except OverflowError as exc:
        raise ValueError("계산 범위를 초과했습니다.") from exc
    if not all(math.isfinite(value) for value in (growth, cagr)):
        raise ValueError("계산 범위를 초과했습니다.")
    return {"growth_percent": round(growth, 4), "cagr_percent": round(cagr, 4)}


class MarketScoutAgent:
    def __init__(self, model: Any, web_search: MarketSearch, rag_search: MarketSearch, *,
                 system_prompt: str | None = None, max_searches: int = 8,
                 max_results: int = 3, recursion_limit: int = 30):
        if max_searches < 2 or not 1 <= max_results <= 5 or recursion_limit < 1:
            raise ValueError("max_searches>=2, max_results=1~5, recursion_limit>=1이 필요합니다.")
        prompt = market_scout_prompt.MARKET_SCOUT_PROMPT if system_prompt is None else system_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("system_prompt는 비어 있지 않은 문자열이어야 합니다.")
        self.system_prompt = prompt
        self.model, self.web_search, self.rag_search = model, web_search, rag_search
        self.max_searches, self.max_results, self.recursion_limit = max_searches, max_results, recursion_limit

    def invoke(self, data: MarketAgentInput | Mapping[str, Any],
               config: RunnableConfig | None = None) -> AgentResult:
        return self.run(data, config).result

    def run(self, data: MarketAgentInput | Mapping[str, Any],
            config: RunnableConfig | None = None) -> MarketRun:
        # 모델 인스턴스도 다시 검증하여 호출자가 수정한 입력이 검증을 우회하지 않게 한다.
        request = MarketAgentInput.model_validate(data.model_dump() if isinstance(data, MarketAgentInput) else data)
        registry: dict[str, MarketSource] = {}
        queries: list[str] = []
        errors: list[MarketError] = []
        calculations: list[dict[str, Any]] = []
        lock = Lock()

        def record_error(node: str, exc: Exception, message: str):
            # 예외 본문에는 인증정보가 포함될 수 있으므로 타입만 기록한다.
            logger.warning("market.%s: %s", node, type(exc).__name__)
            errors.append(MarketError(company=request.company_name, round=request.round_no,
                                      node=f"market.{node}", error_type=type(exc).__name__, message=message))

        def register(item: MarketSource) -> dict[str, Any]:
            identity = json.dumps([request.company_name, item.source_type, item.source,
                                   item.page, item.excerpt], ensure_ascii=False)
            eid = "market-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            if eid in registry and registry[eid] != item:
                raise ValueError("동일 근거의 메타데이터가 충돌했습니다.")
            registry[eid] = item
            return {"evidence_id": eid, **item.model_dump(mode="json")}

        def search(query: str, backend: MarketSearch, kind: str) -> str:
            # 한 모델 응답의 병렬 tool call도 예산과 원장을 공유한다.
            with lock:
                if len(queries) >= self.max_searches:
                    return json.dumps({"error": "검색 한도 도달. 확보한 근거로 평가를 종료하세요."}, ensure_ascii=False)
                queries.append(f"{kind}: {query}")
                try:
                    items = [MarketSource.model_validate(value.model_dump() if isinstance(value, MarketSource) else value)
                             for value in islice(backend(query), self.max_results)]
                    if any(item.source_type != kind for item in items):
                        raise ValueError("검색기와 출처 유형이 다릅니다.")
                    # 청크 일부만 유효한 검색 응답을 원장에 남기지 않는다.
                    before = dict(registry)
                    try:
                        found = [register(item) for item in items]
                    except Exception:
                        registry.clear()
                        registry.update(before)
                        raise
                    return json.dumps(found, ensure_ascii=False)
                except Exception as exc:
                    message = "시장·정책 PDF 검색 실패" if kind == "pdf" else "기업·시장 웹 검색 실패"
                    record_error(f"{kind}_search", exc, message)
                    return json.dumps({"error": message}, ensure_ascii=False)

        @tool
        def search_documents(query: str) -> str:
            """시장·정책·규제·수요 PDF 검색. 원본 페이지와 실제 원문, 근거 ID를 반환한다."""
            return search(query, self.rag_search, "pdf")

        @tool
        def search_web(query: str) -> str:
            """기업별 실제 고객·계약·PoC와 최신 시장·정책의 공개 웹 근거를 검색한다."""
            return search(query, self.web_search, "web")

        @tool("calculate_growth")
        def growth_tool(start_value: float, end_value: float, start_year: int, end_year: int,
                        unit: str, geography: str, market_scope: str, evidence_ids: list[str]) -> str:
            """같은 단위·지역·시장 범위의 자료로 성장률/CAGR 계산. 실제 검색 근거 ID가 필수다."""
            with lock:
                try:
                    if not evidence_ids or set(evidence_ids) - registry.keys():
                        raise ValueError("성장률 계산의 검색 근거가 없습니다.")
                    if not all(value.strip() for value in (unit, geography, market_scope)):
                        raise ValueError("단위·지역·시장 범위가 필요합니다.")
                    value = calculate_growth(start_value, end_value, end_year - start_year)
                    record = dict(start_value=start_value, end_value=end_value, start_year=start_year,
                                  end_year=end_year, unit=unit, geography=geography, market_scope=market_scope,
                                  evidence_ids=list(dict.fromkeys(evidence_ids)), **value)
                    calculations.append(record)
                    return json.dumps(record, ensure_ascii=False)
                except Exception as exc:
                    record_error("calculate_growth", exc, "성장률 계산 입력과 원문 수치의 추가 확인 필요")
                    return json.dumps({"error": "근거·범위·단위·기간·수치를 확인하세요."}, ensure_ascii=False)

        inputs = [register(item) for item in request.input_evidence]
        scope = f"{request.company_name} {request.target_market}"
        missing_query = " ".join(request.previous_missing_items)[:500]
        initial = [json.loads(search_documents.invoke({"query": f"{scope} 시장 규모 성장률 정책 규제 {missing_query}"})),
                   json.loads(search_web.invoke({"query": f"{scope} PoC 유료 고객 장기계약 설치 영업 {missing_query}"}))]
        if not registry:
            record_error("retrieval", LookupError(), "시장성과 고객 수요를 평가할 검색·입력 근거가 없습니다.")
            return MarketRun(result=self._unavailable(request, [item.message for item in errors]),
                             errors=errors, retrieval_queries=queries)
        try:
            agent = create_agent(model=self.model, tools=[search_documents, search_web, growth_tool],
                                 system_prompt=self.system_prompt,
                                 response_format=ToolStrategy(MarketAssessment))
            payload = {**request.model_dump(mode="json"), "input_evidence": inputs,
                       "as_of": date.today().isoformat(), "initial_evidence": initial}
            child_config = {**(config or {}), "recursion_limit": self.recursion_limit}
            response = agent.invoke({"messages": [("user", json.dumps(payload, ensure_ascii=False))]}, config=child_config)
            raw = response["structured_response"]
            assessment = MarketAssessment.model_validate(raw.model_dump() if isinstance(raw, MarketAssessment) else raw)
            result = self._convert(request, assessment, registry, errors)
            return MarketRun(result=result, assessment=assessment, errors=errors,
                             retrieval_queries=queries, calculations=calculations)
        except Exception as exc:
            record_error("evaluation", exc, "모델 응답 또는 근거 검증 실패로 시장성 재평가가 필요합니다.")
            return MarketRun(result=self._unavailable(request, [item.message for item in errors]),
                             errors=errors, retrieval_queries=queries, calculations=calculations)

    @staticmethod
    def _convert(request: MarketAgentInput, assessment: MarketAssessment,
                 registry: dict[str, MarketSource], errors: list[MarketError]) -> AgentResult:
        if assessment.company_name != request.company_name or assessment.round_no != request.round_no:
            raise ValueError("응답의 기업명 또는 평가 회차가 다릅니다.")
        normalize = lambda value: " ".join(value.split())
        cited: dict[str, Evidence] = {}
        for item in assessment.evidence:
            original = registry[item.evidence_id]
            if (item.source_type, item.source, item.page) != (original.source_type, original.source, original.page):
                raise ValueError("인용 출처/유형/페이지가 검색 원문과 다릅니다.")
            if normalize(item.excerpt) not in normalize(original.excerpt):
                raise ValueError("인용한 원문이 해당 검색 결과에 없습니다.")
            is_url = urlsplit(original.source).scheme in {"http", "https"}
            title = original.source_title if is_url else f"{original.source_title} [{original.source}]"
            cited[item.evidence_id] = Evidence(evidence_id=item.evidence_id,
                claim=f"{item.claim}\n원문: {item.excerpt}", source_title=title,
                source_url=original.source if is_url else None, page=original.page)
        missing = [*assessment.missing_items, *(item.message for item in errors)]
        evaluations: list[CriterionEvaluation] = []
        used: dict[str, Evidence] = {}
        for criterion in MARKET_CRITERIA:
            item = next(value for value in assessment.scores if value.criterion == criterion)
            ids = list(dict.fromkeys(item.evidence_ids))
            if item.score is None:
                missing.append(f"{criterion}: 점수 산정에 필요한 근거 부족")
            if not ids:
                missing.append(f"{criterion}: 연결된 근거 없음")
            elif all(registry[eid].source_type == "input" for eid in ids):
                missing.append(f"{criterion}: 입력 자료를 독립적인 출처로 교차검증해야 합니다.")
            used.update((eid, cited[eid]) for eid in ids)
            reason = item.reason
            if not evaluations:
                reason += f"\n시장성 종합: {assessment.summary}"
            evaluations.append(CriterionEvaluation(criterion=criterion, score=item.score,
                                                   reason=reason, evidence_ids=ids))
        if assessment.status != "success" and not missing:
            missing.append("시장성 평가가 완료되지 않아 추가 확인이 필요합니다.")
        if assessment.status == "error":
            return MarketScoutAgent._unavailable(request, missing)
        needs_more = bool(missing)
        return AgentResult(agent_name="market", company_name=request.company_name, evaluations=evaluations,
                           evidence=list(used.values()), risks=[], missing_information=list(dict.fromkeys(missing)),
                           confidence="low" if needs_more else "medium", needs_more_information=needs_more)

    @staticmethod
    def _unavailable(request: MarketAgentInput, reasons: list[str]) -> AgentResult:
        return AgentResult(agent_name="market", company_name=request.company_name,
            evaluations=[CriterionEvaluation(criterion=criterion, score=None, reason="근거 부족 또는 실행 실패로 평가 불가")
                         for criterion in MARKET_CRITERIA], evidence=[],
            missing_information=list(dict.fromkeys(reasons)), confidence="low", needs_more_information=True)

    def as_node(self):
        """Graph state['market_input']을 받아 market_result와 errors 변경분을 반환한다."""
        def node(state, config=None):
            run = self.run(state["market_input"], config)
            return {"market_result": run.result, "errors": [item.model_dump() for item in run.errors]}
        return node


def create_market_scout(model: Any, web_search: MarketSearch, rag_search: MarketSearch,
                        **kwargs) -> MarketScoutAgent:
    return MarketScoutAgent(model, web_search, rag_search, **kwargs)


def run(state: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Graph State를 기존 시장성 Agent의 입력 및 Graph 결과 계약으로 변환한다."""
    context = state["company_context"]
    round_no = state["retry_state"]["retry_count"] + 1
    criteria = kwargs.get("criteria", list(MARKET_CRITERIA.items()))
    if dict(criteria) != MARKET_CRITERIA:
        raise ValueError("시장성 평가 항목과 배점이 Agent 계약과 다릅니다.")
    agent_input = {
        **state.get("market_input", {}),
        "company_name": context["company_name"],
        "evaluation_request": state["evaluation_request"],
        "round_no": round_no,
    }
    request = MarketAgentInput.model_validate(agent_input)
    model = kwargs.get("model")
    if model is None:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)
    empty_search = lambda query: []
    agent = create_market_scout(
        model, kwargs.get("web_search") or empty_search,
        kwargs.get("rag_search") or empty_search,
    )
    result = agent.run(request, config=kwargs.get("config"))
    assessment = result.assessment
    evidence = [
        {"evidence_id": item.evidence_id, "claim": item.claim,
         "excerpt": item.excerpt, "source_type": item.source_type,
         "source": item.source, **({"page": item.page} if item.page is not None else {})}
        for item in assessment.evidence
    ] if assessment is not None else []
    used_ids = {item.evidence_id for item in result.result.evidence}
    evidence = [item for item in evidence if item["evidence_id"] in used_ids]
    scores = [
        {"criterion": item.criterion, "score": item.score,
         "max_score": MARKET_CRITERIA[item.criterion], "reason": item.reason,
         "evidence_ids": item.evidence_ids}
        for item in result.result.evaluations if item.score is not None
    ]
    missing = result.result.missing_information
    status = "success" if assessment is not None and assessment.status == "success" and not missing else "partial"
    if assessment is None and not scores:
        status = "error"
    output = {
        "company_name": request.company_name, "round_no": request.round_no,
        "agent": "market_scout", "status": status,
        "summary": assessment.summary if assessment is not None else "시장성 평가에 필요한 근거가 부족합니다.",
        "evidence": evidence, "missing_items": missing, "scores": scores,
    }
    if len(scores) == len(MARKET_CRITERIA):
        output["score"] = sum(item["score"] for item in scores)
    return output


if __name__ == "__main__":
    print("시장성 Agent 모듈을 정상적으로 불러왔습니다.\n"
          "create_market_scout(model, web_search, rag_search).invoke(MarketAgentInput(...))로 평가하세요.\n"
          "직접 실행만으로 웹 검색, RAG 또는 LLM 호출은 시작되지 않습니다.")
