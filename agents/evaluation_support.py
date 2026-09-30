"""investment_evaluator.py와 report_generator.py의 입출력 모델 및 공통 검증 함수."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from math import isfinite
from typing import Any, Literal
import json


class AgentGenerationError(RuntimeError):
    """모델 호출 실패 또는 유효하지 않은 생성 결과."""


def generate(model, prompt, payload, output_instruction=""):
    """주입된 채팅 모델에 프롬프트와 입력을 전달한다."""
    messages = [
        {"role": "system", "content": prompt + "\n\n" + output_instruction},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
    ]
    try:
        response = model.invoke(messages)
        content = response if isinstance(response, str) else response.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("빈 응답 또는 지원하지 않는 응답 형식")
        return content.strip()
    except Exception as exc:
        raise AgentGenerationError("LLM 호출 또는 응답 읽기에 실패했습니다") from exc


def generate_json(model, prompt, payload, instruction):
    content = generate(model, prompt, payload, instruction)
    try:
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("JSON 객체가 필요합니다")
        return value
    except (ValueError, TypeError) as exc:
        raise AgentGenerationError("LLM 응답이 유효한 JSON 객체가 아닙니다") from exc


CRITERIA = {
    "founder": {"창업자·팀 역량": 25},
    "market": {"시장 규모·성장 가능성": 25, "실제 고객 수요": 25},
    "technology": {"제품·기술 정보의 구체성": 15, "제품 차별성·성능 정보의 명확성": 10},
}


class InputValidationError(ValueError):
    """Graph 호출자가 잡아서 State의 오류 필드에 기록할 수 있는 오류."""


def require(condition, message):
    if not condition:
        raise InputValidationError(message)


def mapping(value, label):
    if callable(getattr(value, "model_dump", None)):
        value = value.model_dump()
    require(isinstance(value, Mapping), f"{label}: 딕셔너리 또는 model_dump() 모델이 필요합니다")
    return deepcopy(dict(value))


def nonempty(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label}: 빈 문자열은 허용되지 않습니다")


def strings(value, label):
    require(isinstance(value, list), f"{label}: 문자열 목록이 필요합니다")
    for item in value:
        nonempty(item, label)


def unique(values):
    return list(dict.fromkeys(values))


def validate_score(score, maximum, label):
    if score is None:
        return
    require(type(score) in (int, float), f"{label}: 숫자 또는 None이 필요합니다")
    require(isfinite(score) and 0 <= score <= maximum, f"{label}: 0~{maximum} 범위여야 합니다")


def validate_specialist(value, role, company):
    result = mapping(value, role)
    require(result.get("agent_name") == role, f"{role}: agent_name 불일치")
    require(result.get("company_name") == company, f"{role}: company_name 불일치")
    require(result.get("confidence") in ("high", "medium", "low"), f"{role}: confidence 오류")
    require(type(result.get("needs_more_information")) is bool, f"{role}: needs_more_information 오류")
    for key in ("risks", "missing_information"):
        result.setdefault(key, [])
        strings(result[key], f"{role}.{key}")
    for key in ("evaluations", "evidence"):
        require(isinstance(result.get(key), list), f"{role}.{key}: 목록이 필요합니다")
        result[key] = [mapping(item, key) for item in result[key]]
    evidence_by_id = {}
    for item in result["evidence"]:
        for key in ("evidence_id", "claim", "source_title"):
            nonempty(item.get(key), f"{role}.evidence.{key}")
        item.setdefault("source_url", None)
        item.setdefault("page", None)
        if item["source_url"] is not None:
            nonempty(item["source_url"], "source_url")
        if item["page"] is not None:
            require(type(item["page"]) is int and item["page"] > 0, "page: 양의 정수가 필요합니다")
        eid = item["evidence_id"]
        require(eid not in evidence_by_id or evidence_by_id[eid] == item,
                f"{role}: 서로 다른 근거가 같은 ID를 사용합니다: {eid}")
        evidence_by_id[eid] = item
    seen = set()
    for item in result["evaluations"]:
        criterion = item.get("criterion")
        nonempty(criterion, "criterion")
        require(criterion in CRITERIA[role], f"{role}: 합의되지 않은 평가 항목: {criterion}")
        require(criterion not in seen, f"{role}: 중복 평가 항목: {criterion}")
        seen.add(criterion)
        nonempty(item.get("reason"), "reason")
        item.setdefault("score", None)
        validate_score(item["score"], CRITERIA[role][criterion], criterion)
        item.setdefault("evidence_ids", [])
        strings(item["evidence_ids"], "evidence_ids")
        for eid in item["evidence_ids"]:
            require(eid in evidence_by_id, f"{role}: 존재하지 않는 근거 ID: {eid}")
    return result


def collect_evidence(results):
    """동일 근거만 병합하고 참조 ID를 보존한다. 같은 출처의 다른 주장/페이지는 유지한다."""
    records, aliases, identities, ids = [], {}, {}, {}
    for result in results:
        for item in result["evidence"]:
            eid = item["evidence_id"]
            identity = (item["claim"], item["source_title"], item["source_url"], item["page"])
            require(eid not in ids or ids[eid] == identity, f"Agent 간 근거 ID 충돌: {eid}")
            ids[eid] = identity
            if identity not in identities:
                identities[identity] = eid
                records.append(deepcopy(item))
            aliases[eid] = identities[identity]
    return records, aliases


def calculate_total(results):
    """필수 항목 또는 점수가 없으면 부분 합계를 총점으로 표시하지 않는다."""
    scores = []
    for result in results:
        evaluations = {item["criterion"]: item for item in result["evaluations"]}
        for criterion, maximum in CRITERIA[result["agent_name"]].items():
            score = evaluations.get(criterion, {}).get("score")
            if score is None:
                return None
            validate_score(score, maximum, criterion)
            scores.append(Decimal(str(score)))
    return float(sum(scores))


class Serializable:
    def model_dump(self):
        return asdict(self)


@dataclass
class InvestmentAgentInput(Serializable):
    company_name: str
    founder_result: Any = None
    # agents/market_scout.py에서 받을 시장성 AgentResult (agent_name="market").
    # evaluations에 시장 규모·성장 가능성 / 실제 고객 수요 평가와 점수를 담는다.
    market_result: Any = None
    # agents/tech_brief.py에서 받을 제품·기술 AgentResult (agent_name="technology").
    # evaluations에 제품·기술 정보의 구체성 / 제품 차별성·성능 정보의 명확성을 담는다.
    tech_result: Any = None
    company_context: dict = field(default_factory=dict)
    evidence_review: dict = field(default_factory=dict)
    criteria_met: bool | None = None
    evaluation_settings: dict = field(default_factory=dict)

    # TODO(RAG/공통 모델 연동): agents/founder_insight.py, agents/market_scout.py,
    # agents/tech_brief.py의 출력 계약 확정 후 이 파일의 validate_specialist()와
    # InvestmentAgentInput을 수정한다. 공통 AgentResult 정의 파일의 경로는 아직 미정이다.
    # 두 RAG 결과 모두 evidence와 evaluations[].evidence_ids를 연결해야 한다.
    # PDF 파일명은 source_title, 페이지는 page로 받는 계약을 확인하고,
    # 검색 실패·자료 부족은 missing_information / needs_more_information으로 전달받는다.
    # 현재 dict 또는 model_dump() 모델을 지원하며, 검색 청크/검색 기록은 여기서 사용하지 않는다.

    def __post_init__(self):
        nonempty(self.company_name, "company_name")
        for key in ("company_context", "evidence_review", "evaluation_settings"):
            setattr(self, key, mapping(getattr(self, key), key))
        require(self.criteria_met is None or type(self.criteria_met) is bool, "criteria_met: bool 또는 None 필요")
        if "company_name" in self.company_context:
            require(self.company_context["company_name"] == self.company_name, "company_context 기업명 불일치")
        for name, role in (("founder_result", "founder"), ("market_result", "market"),
                           ("tech_result", "technology")):
            value = getattr(self, name)
            require(value is not None, f"필수 Agent 결과 누락: {name}")
            setattr(self, name, validate_specialist(value, role, self.company_name))
        collect_evidence(self.results)

    @property
    def results(self):
        return [self.founder_result, self.market_result, self.tech_result]


@dataclass
class InvestmentResult(Serializable):
    company_name: str
    total_score: float | None = None
    decision: Literal["first_review_pass", "additional_research", "hold", "pending"] = "pending"
    key_reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    needs_more_information: bool = False

    def __post_init__(self):
        nonempty(self.company_name, "company_name")
        validate_score(self.total_score, 100, "total_score")
        require(self.decision in ("first_review_pass", "additional_research", "hold", "pending"), "decision 오류")
        for key in ("key_reasons", "risks", "missing_information"):
            strings(getattr(self, key), key)
        require(type(self.needs_more_information) is bool, "needs_more_information 오류")
        require(not self.missing_information or self.needs_more_information,
                "부족 정보가 있으면 needs_more_information=True여야 합니다")
        require(not self.needs_more_information or self.decision in ("pending", "additional_research"),
                "정보 부족 상태에서 최종 투자 판단을 내릴 수 없습니다")


@dataclass
class ReportAgentInput(InvestmentAgentInput):
    # market_result / tech_result는 위 입력 모델에서 상속한다.
    # Graph는 해당 RAG 결과로 산출한 investment_result를 함께 전달해야 한다.
    investment_result: Any = None
    evaluation_status: str | None = None

    def __post_init__(self):
        super().__post_init__()
        require(self.investment_result is not None, "필수 투자 판단 결과 누락")
        self.investment_result = InvestmentResult(**mapping(self.investment_result, "investment_result"))
        require(self.investment_result.company_name == self.company_name, "투자 판단의 company_name 불일치")
        require(self.investment_result.total_score == calculate_total(self.results),
                "투자 판단 총점과 전문 Agent 점수 합계가 다릅니다")
        if self.evaluation_status is None:
            self.evaluation_status = self.investment_result.decision
        require(self.evaluation_status == self.investment_result.decision,
                "evaluation_status와 투자 판단이 다릅니다")


@dataclass
class ReportResult(Serializable):
    summary: str
    business_overview: str
    market_analysis: str
    product_technology_and_team: str
    investment_review: str
    references: list[dict] = field(default_factory=list)

    def __post_init__(self):
        for key in ("summary", "business_overview", "market_analysis", "product_technology_and_team", "investment_review"):
            nonempty(getattr(self, key), key)
        require(isinstance(self.references, list), "references: 목록이 필요합니다")


# 전문 Agent 공통 결과 계약
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Evidence(ContractModel):
    """claim은 출처에서 확인한 근거 문장. 검색 어댑터는 원문 발췌를 넣는다."""

    evidence_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_url: str | None = None
    page: int | None = Field(default=None, ge=1)


class CriterionEvaluation(ContractModel):
    criterion: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0, le=100)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class AgentResult(ContractModel):
    agent_name: Literal["founder", "market", "technology"]
    company_name: str = Field(min_length=1)
    evaluations: list[CriterionEvaluation]
    evidence: list[Evidence]
    risks: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    needs_more_information: bool

    @model_validator(mode="after")
    def validate_references(self):
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id는 결과 내에서 중복될 수 없습니다.")
        criteria = [item.criterion for item in self.evaluations]
        if len(criteria) != len(set(criteria)):
            raise ValueError("평가 항목은 결과 내에서 중복될 수 없습니다.")
        for evaluation in self.evaluations:
            if set(evaluation.evidence_ids) - set(ids):
                raise ValueError("evidence_ids는 결과에 포함된 근거만 참조해야 합니다.")
            if evaluation.score is not None and not evaluation.evidence_ids:
                raise ValueError("근거가 없는 항목의 score는 None이어야 합니다.")
        return self
