"""창업자 검증 Agent: 웹 검색만 사용, 창업자·팀 역량 25점 담당."""
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from itertools import islice
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

# agents/ 안에서 python founder_insight.py로 실행한 경우에도 패키지 경로를 설정한다.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "agents"

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import Field, field_validator

from prompts import founder_insight_prompt

from .evaluation_support import AgentResult, ContractModel, CriterionEvaluation, Evidence

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

# Graph 호출 시 모델을 별도로 주입하지 않으면 이 설정으로 생성한다.
MODEL_NAME = os.environ["FOUNDER_MODEL"]
MODEL_TEMPERATURE = 0
TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 5
FOUNDER_CRITERION = "창업자·팀 역량"
FOUNDER_MAX_SCORE = float(os.environ["FOUNDER_WEIGHT"])

# common.py가 없는 팀 저장소에서도 공개 Agent를 독립적으로 import할 수 있다.
Search = Callable[[str], Sequence[Any]]


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


class FounderEvidence(ContractModel):
    """프롬프트의 근거 형식. checked_at은 발행일이 아닌 이번 검색 확인일이다."""

    claim: str = Field(min_length=1)
    source: str = Field(min_length=1, description="검색 결과의 source_title을 그대로 사용")
    url: str = Field(min_length=1, description="검색 결과의 source_url을 그대로 사용")
    checked_at: date = Field(description="검색 결과에 제공된 확인 날짜, YYYY-MM-DD")


class FounderAssessment(ContractModel):
    """FOUNDER_INSIGHT_PROMPT의 반환 형식. 외부 반환 시 AgentResult로 변환한다."""

    score: float | None = Field(ge=0, le=100, description="0~100 원점수. 평가 불가 시 null")
    conclusion: str = Field(min_length=1)
    evidence: list[FounderEvidence]
    risks: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)


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

    def __init__(self, model: Any, web_search: FounderSearch, *,
                 system_prompt: str | None = None, max_searches: int = 8,
                 recursion_limit: int = 30):
        if max_searches < 2 or recursion_limit < 1:
            raise ValueError("max_searches >= 2, recursion_limit >= 1이어야 합니다.")
        self.system_prompt = founder_insight_prompt.FOUNDER_INSIGHT_PROMPT if system_prompt is None else system_prompt
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt는 비어 있지 않은 문자열이어야 합니다.")
        # 템플릿의 이스케이프된 JSON 중괄호만 복원한다. 사용자 입력은 치환하지 않는다.
        if "{{" in self.system_prompt:
            self.system_prompt = self.system_prompt.replace("{{", "{").replace("}}", "}")
        self.model = model
        self.web_search = web_search
        self.max_searches = max_searches
        self.recursion_limit = recursion_limit

    def invoke(self, data: FounderAgentInput | Mapping[str, Any],
               config: RunnableConfig | None = None) -> AgentResult:
        request = FounderAgentInput.model_validate(data)
        checked_at = date.today()
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
                    return json.dumps([{**item.model_dump(), "checked_at": checked_at.isoformat()}
                                       for item in batch.values()], ensure_ascii=False)
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
            agent = create_agent(model=self.model, tools=[search_web], system_prompt=self.system_prompt,
                                 response_format=ToolStrategy(FounderAssessment))
            payload = {**request.model_dump(), "as_of": checked_at.isoformat(),
                       "initial_evidence": [json.loads(value) for value in initial]}
            child_config = dict(config or {})
            child_config["recursion_limit"] = self.recursion_limit
            response = agent.invoke({"messages": [("user", json.dumps(payload, ensure_ascii=False))]},
                                    config=child_config)
            assessment = FounderAssessment.model_validate(response["structured_response"])
            # URL과 제목이 실제 검색 원장에 존재해야 한다. 같은 URL의 청크는 모두 보존한다.
            used: dict[str, Evidence] = {}
            for citation in assessment.evidence:
                if citation.checked_at != checked_at:
                    raise ValueError("근거 확인 날짜가 이번 검색 날짜와 다릅니다.")
                matches = [item for item in registry.values()
                           if item.source_url == citation.url and item.source_title == citation.source]
                if not matches:
                    raise ValueError("검색 결과에 없는 URL 또는 출처명입니다.")
                # LLM이 작성한 claim 대신 실제 검색 원문을 공통 Evidence에 보존한다.
                used.update((item.evidence_id, item) for item in matches)
            missing = list(dict.fromkeys([*assessment.missing_items, *issues]))
            score = assessment.score if used else None
            if score is None:
                missing.append("창업자·팀 역량 점수를 확정할 근거가 부족합니다.")
            if not used:
                missing.append("평가를 뒷받침하는 웹 근거가 인용되지 않았습니다.")
            domains = {urlsplit(item.source_url).hostname.removeprefix("www.") for item in used.values()}
            if used and len(domains) < 2:
                missing.append("주요 경력을 뒷받침하는 독립 출처의 추가 교차검증이 필요합니다.")
            # 프롬프트에는 confidence가 없다. 출처 개수만으로 high를 부여하지 않는다.
            needs_more = bool(missing or assessment.risks or score is None)
            confidence = "low" if needs_more else "medium"
            reason = assessment.conclusion
            if used:
                # 공통 Evidence에는 날짜 필드가 없어 평가 사유에 ID별 확인일을 남긴다.
                reason += "\n근거 확인 날짜: " + "; ".join(
                    f"{key}={checked_at.isoformat()}" for key in used)
            return AgentResult(agent_name="founder", company_name=request.company_name,
                               evaluations=[CriterionEvaluation(criterion="founder_team", score=score,
                                            reason=reason, evidence_ids=list(used))],
                               evidence=list(used.values()), risks=assessment.risks,
                               missing_information=list(dict.fromkeys(missing)),
                               confidence=confidence, needs_more_information=needs_more)
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


def create_founder_agent(model: Any, web_search: FounderSearch, *,
                         system_prompt: str | None = None, **kwargs) -> FounderInsightAgent:
    """prompts 파일의 기본 프롬프트 또는 전달받은 system_prompt를 적용한다."""
    return FounderInsightAgent(model, web_search, system_prompt=system_prompt, **kwargs)


# 팀 저장소에서는 두 팩토리 이름 모두 공통 AgentResult를 반환하는 Agent를 만든다.
create_founder_insight = create_founder_agent


def _default_web_search(query: str) -> list[Evidence]:
    """Graph가 검색기를 주입하지 않을 때 Tavily 웹 검색 결과를 공통 근거로 변환한다."""
    api_key = os.environ["TAVILY_API_KEY"]
    payload = json.dumps({"api_key": api_key, "query": query,
                          "max_results": TAVILY_MAX_RESULTS}).encode("utf-8")
    request = Request(TAVILY_API_URL, data=payload,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        results = json.load(response)["results"]
    return [Evidence(evidence_id=f"web-{item['url']}", claim=item["content"],
                     source_title=item.get("title") or item["url"],
                     source_url=item["url"], page=None) for item in results]


def run(state: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Graph의 호출·결과 계약과 기존 FounderInsightAgent 사이의 어댑터."""
    company_name = state["company_context"]["company_name"]
    round_no = state["retry_state"]["retry_count"] + 1
    request = FounderAgentInput.model_validate({
        "company_name": company_name,
        "founder_names": state.get("founder_names", []),
        "evaluation_request": state["evaluation_request"],
    })
    model = kwargs.get("model")
    if model is None:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)
    web_search = kwargs.get("web_search", _default_web_search)
    result = create_founder_agent(model, web_search).invoke(request, config=kwargs.get("config"))
    evidence = [{"evidence_id": item.evidence_id, "claim": item.claim,
                 "excerpt": item.claim, "source_type": "web", "source": item.source_url}
                for item in result.evidence]
    missing = list(dict.fromkeys([*result.missing_information, *result.risks]))
    evaluation = result.evaluations[0]
    summary = evaluation.reason
    scores = []
    if evaluation.score is not None and evaluation.evidence_ids:
        criteria = kwargs.get("criteria", [(FOUNDER_CRITERION, FOUNDER_MAX_SCORE)])
        criterion, maximum = criteria[0]
        scores.append({"criterion": criterion, "score": evaluation.score / 100 * maximum,
                       "max_score": maximum, "reason": evaluation.reason,
                       "evidence_ids": evaluation.evidence_ids})
    output = {"company_name": company_name, "round_no": round_no,
              "agent": "founder_insight", "status": "partial" if missing or not scores else "success",
              "summary": summary, "evidence": evidence, "missing_items": missing,
              "scores": scores}
    if scores:
        output["score"] = scores[0]["score"]
    return output


if __name__ == "__main__":
    print("창업자 Agent 모듈을 정상적으로 불러왔습니다.\n"
          "기업 평가는 create_founder_agent(model, web_search).invoke(입력값)으로 실행하세요.\n"
          "이 파일을 단독 실행하는 것만으로 검색이나 LLM 호출은 시작되지 않습니다.")
