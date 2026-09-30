"""창업자 검증 Agent: 웹 검색만 사용, 창업자·팀 역량 25점 담당."""
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from itertools import islice
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlsplit

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import Field, field_validator

from .common import Search, create_research_node
from .models import AgentResult, ContractModel, CriterionEvaluation, Evidence

SYSTEM_PROMPT = """당신은 국내 B2C 에너지 스타트업의 창업자 검증 Agent다.
...
"""


def create_founder_insight(model: Any, web_search: Search, *, system_prompt: str = SYSTEM_PROMPT, **kwargs):
    """기존 그래프 호환용. 새 공개 계약은 create_founder_agent를 사용한다."""
    return create_research_node(role="founder", model=model, web_search=web_search,
                                prompt=system_prompt, **kwargs)


class FounderAgentInput(ContractModel):
    company_name: str = Field(min_length=1)
    founder_names: list[str] = Field(default_factory=list)
    evaluation_request: str = Field(min_length=1)

    @field_validator("founder_names")
    @classmethod
    def clean_names(cls, names: list[str]) -> list[str]:
        if any(not name.strip() for name in names):
            raise ValueError("창업자 이름은 공백일 수 없습니다. 모르면 빈 목록을 사용하세요.")
        return list(dict.fromkeys(name.strip() for name in names))


FounderSearch = Callable[[str], Sequence[Evidence]]
logger = logging.getLogger(__name__)


class FounderAssessment(ContractModel):
    """LLM 전용 출력. 회사명·역할·출처 내용은 실행기가 직접 채운다."""

    evaluations: list[CriterionEvaluation] = Field(min_length=1, max_length=1)
    risks: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    needs_more_information: bool


FOUNDER_PROMPT = """당신은 국내 B2C 에너지 스타트업의 창업자 검증 Agent다.
...
"""


def adapt_founder_search(backend: Search) -> FounderSearch:
    """기존 TavilyWebSearch 등 common.Evidence 검색기를 공개 계약으로 변환한다."""
    def search(query: str) -> list[Evidence]:
        found = backend(query)
        if any(item.kind != "web" for item in found):
            raise ValueError("창업자 검증에는 웹 검색 결과만 사용할 수 있습니다.")
        return [Evidence(evidence_id=item.source_id, claim=item.excerpt,
                         source_title=item.title, source_url=item.source, page=None)
                for item in found]
    return search


class FounderInsightAgent:
    """FounderAgentInput -> AgentResult. 모델·검색기는 주입하며 호출별 상태를 분리한다.

    입력/설정 오류는 예외로 알린다. 검색·LLM·출처 검증 실패는 평가 보류 결과와
    타입만 포함한 로그로 남긴다. 회사별 재평가 루프는 호출하는 그래프가 담당한다.
    """

    def __init__(self, model: Any, web_search: FounderSearch, *, max_searches: int = 8,
                 recursion_limit: int = 30):
        if max_searches < 2 or recursion_limit < 1:
            raise ValueError("max_searches >= 2, recursion_limit >= 1이어야 합니다.")
        self.model = model
        self.web_search = web_search
        self.max_searches = max_searches
        self.recursion_limit = recursion_limit

    def invoke(self, data: FounderAgentInput | Mapping[str, Any],
               config: RunnableConfig | None = None) -> AgentResult:
        request = FounderAgentInput.model_validate(data)
        registry: dict[str, Evidence] = {}
        issues: list[str] = []
        queries: list[str] = []
        # 한 응답의 여러 tool call은 병렬 실행될 수 있다.
        search_lock = Lock()

        @tool
        def search_web(query: str) -> str:
            """회사와 창업자의 공개 경력·사업화 이력을 웹 검색한다. 출처 ID와 원문을 반환한다."""
            with search_lock:
                if len(queries) >= self.max_searches:
                    return json.dumps({"error": "검색 한도 도달. 확인한 근거로 평가를 종료하세요."}, ensure_ascii=False)
                queries.append(query)
                try:
                    # 검색기가 반환한 근거를 복사하여 이후 외부 변경으로부터 보호한다.
                    found = [Evidence.model_validate(
                        item.model_dump() if isinstance(item, Evidence) else item
                    ) for item in islice(self.web_search(query), 5)]
                    batch: dict[str, Evidence] = {}
                    for item in found:
                        url = urlsplit(item.source_url or "")
                        if url.scheme not in {"http", "https"} or not url.hostname or item.page is not None:
                            raise ValueError("웹 근거에는 HTTP(S) URL과 page=None이 필요합니다.")
                        previous = batch.get(item.evidence_id) or registry.get(item.evidence_id)
                        if previous is not None and previous != item:
                            raise ValueError("근거 ID 충돌")
                        batch[item.evidence_id] = item
                    registry.update(batch)
                    return json.dumps([item.model_dump() for item in batch.values()], ensure_ascii=False)
                except Exception as exc:
                    logger.warning("founder.search_failed: %s", type(exc).__name__)
                    issues.append("일부 웹 검색이 실패하여 경력·사업화 이력의 추가 확인이 필요합니다.")
                    return json.dumps({"error": "검색 실패. 다른 검색어로 재시도하세요."}, ensure_ascii=False)

        identity = f"{request.company_name} {' '.join(request.founder_names)}".strip()
        initial = [search_web.invoke({"query": f"{identity} 창업자 핵심 팀 경력 에너지 전문성"})]
        initial.append(search_web.invoke({"query": f"{identity} 인터뷰 창업 사업화 투자유치 이력"}))
        if not registry:
            return self._unavailable(request, [*issues, "창업자와 핵심 팀을 확인할 웹 근거가 없습니다."])

        try:
            agent = create_agent(model=self.model, tools=[search_web], system_prompt=FOUNDER_PROMPT,
                                 response_format=ToolStrategy(FounderAssessment))
            payload = {**request.model_dump(), "as_of": date.today().isoformat(),
                       "initial_evidence": [json.loads(value) for value in initial]}
            child_config = dict(config or {})
            child_config["recursion_limit"] = self.recursion_limit
            response = agent.invoke({"messages": [("user", json.dumps(payload, ensure_ascii=False))]},
                                    config=child_config)
            assessment = FounderAssessment.model_validate(response["structured_response"])
            evaluation = assessment.evaluations[0]
            if evaluation.criterion != "founder_team":
                raise ValueError("창업자 Agent의 평가 항목은 founder_team입니다.")
            # 미등록 ID는 KeyError로 실패한다. LLM은 URL/제목/원문을 생성할 수 없다.
            used = [registry[key] for key in dict.fromkeys(evaluation.evidence_ids)]
            missing = list(dict.fromkeys([*assessment.missing_information, *issues]))
            if evaluation.score is None:
                missing.append("창업자·팀 역량 점수를 확정할 근거가 부족합니다.")
            if not used:
                missing.append("평가를 뒷받침하는 웹 근거가 인용되지 않았습니다.")
            confidence = assessment.confidence
            if issues or not used or evaluation.score is None:
                confidence = "low"
            # 서로 다른 도메인도 독립 검증의 충분조건은 아니다. 독립성은 프롬프트로 검토한다.
            domains = {urlsplit(item.source_url).hostname.removeprefix("www.") for item in used}
            if confidence == "high" and len(domains) < 2:
                confidence = "medium"
                missing.append("주요 경력을 뒷받침하는 독립 출처의 추가 교차검증이 필요합니다.")
            if assessment.needs_more_information and not missing:
                missing.append("창업자·팀의 주요 경력과 사업화 이력에 대한 추가 확인이 필요합니다.")
            return AgentResult(agent_name="founder", company_name=request.company_name,
                               evaluations=assessment.evaluations, evidence=used,
                               risks=assessment.risks, missing_information=list(dict.fromkeys(missing)),
                               confidence=confidence,
                               needs_more_information=assessment.needs_more_information or bool(missing))
        except Exception as exc:
            logger.warning("founder.evaluation_failed: %s", type(exc).__name__)
            return self._unavailable(request, [*issues, "모델 응답 또는 출처 검증에 실패하여 재평가가 필요합니다."])

    @staticmethod
    def _unavailable(request: FounderAgentInput, reasons: list[str]) -> AgentResult:
        return AgentResult(agent_name="founder", company_name=request.company_name,
                           evaluations=[CriterionEvaluation(criterion="founder_team", score=None,
                                       reason="확인 가능한 근거가 부족하거나 평가 실행에 실패했습니다.")],
                           evidence=[], missing_information=list(dict.fromkeys(reasons)),
                           confidence="low", needs_more_information=True)


def create_founder_agent(model: Any, web_search: FounderSearch, **kwargs) -> FounderInsightAgent:
    """공개 계약용 팩토리. 반환 객체의 invoke(FounderAgentInput)를 호출한다."""
    return FounderInsightAgent(model, web_search, **kwargs)
