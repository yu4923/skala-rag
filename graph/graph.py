import math
import os
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from dotenv import dotenv_values, load_dotenv
from .state import InvestmentState

from langgraph.graph import END, START, StateGraph


# ──────────────────────────────────────
# 평가 기준 및 실행 설정: .env에서 관리
# ──────────────────────────────────────

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
ENV = dotenv_values(ENV_PATH)
load_dotenv(dotenv_path=ENV_PATH, override=True)


def setting(name, convert=str):
    value = ENV.get(name)
    if value is None or not value.strip():
        raise ValueError(f".env에 {name} 값을 설정하세요.")
    try:
        return convert(value)
    except ValueError as exc:
        raise ValueError(f".env의 {name} 값이 올바르지 않습니다: {value!r}") from exc


def boolean_setting(name):
    value = setting(name).lower()
    if value not in {"true", "false"}:
        raise ValueError(f".env의 {name} 값은 true 또는 false여야 합니다.")
    return value == "true"


MAX_RETRIES = setting("MAX_RETRIES", int)
MIN_EVIDENCE_COUNT = setting("MIN_EVIDENCE_COUNT", int)
STOP_ON_FIRST_RECOMMENDATION = boolean_setting("STOP_ON_FIRST_RECOMMENDATION")
GRAPH_RECURSION_LIMIT = setting("GRAPH_RECURSION_LIMIT", int)
SCORE_TOLERANCE = setting("SCORE_TOLERANCE", float)
ENABLE_LANGSMITH = boolean_setting("LANGSMITH_TRACING")

CRITERIA = {
    "founder_insight": [
        ("창업자·팀 역량", setting("FOUNDER_WEIGHT", float)),
    ],
    "market_scout": [
        ("시장 규모·성장 가능성", setting("MARKET_GROWTH_WEIGHT", float)),
        ("실제 고객 수요", setting("CUSTOMER_DEMAND_WEIGHT", float)),
    ],
    "tech_brief": [
        ("제품·기술 정보의 구체성", setting("TECH_SPECIFICITY_WEIGHT", float)),
        ("제품 차별성·성능 정보의 명확성", setting("TECH_DIFFERENTIATION_WEIGHT", float)),
    ],
}

AGENT_RESULT_FIELDS = {
    "founder_insight": "founder_result",
    "market_scout": "market_result",
    "tech_brief": "tech_result",
}
AGENT_MODULES = {
    "founder_insight": "agents.founder_insight",
    "market_scout": "agents.market_scout",
    "tech_brief": "agents.tech_brief",
    "investment_evaluator": "agents.investment_evaluator",
    "report_generator": "agents.report_generator",
}

NO_DATA_SCORE_REASON = "[잠정 추정] 입력된 기업 정보와 평가 요청을 바탕으로 임시 점수를 산정함"
NO_DATA_SCORE_RATIO = 0.5


def estimate_missing_scores(state, agent, criteria, result):
    """사용자 입력과 이미 생성된 평가 내용으로 근거 없는 잠정 점수를 산정한다."""
    import json
    from pydantic import BaseModel, Field
    from agents.evaluation_support import create_model_from_env

    class Estimate(BaseModel):
        criterion: str
        score: float
        reason: str = Field(min_length=1)
        evidence_ids: list[str] = Field(default_factory=list)

    class Estimates(BaseModel):
        scores: list[Estimate]

    payload = {
        "company_name": state["company_context"]["company_name"],
        "company_context": state["company_context"],
        "evaluation_request": state["evaluation_request"],
        "founder_names": state.get("founder_names", []),
        "product_names": state.get("product_names", []),
        "market_input": state.get("market_input", {}),
        "agent": agent,
        "existing_summary": result.get("summary", ""),
        "existing_scores": result.get("scores", []),
        "available_evidence": [
            {"evidence_id": item["evidence_id"], "claim": item["claim"][:1200],
             "source": item["source"]}
            for item in result.get("evidence", [])[:20]
        ],
        "assessment_limitations": result.get("missing_items", []),
        "previous_run_notes": [
            {"node": item.get("node"), "code": item.get("code"), "message": item.get("message")}
            for item in state.get("errors", [])
            if item.get("company_name") == state["company_context"]["company_name"]
        ][-5:],
        "criteria": [{"criterion": name, "max_score": maximum} for name, maximum in criteria],
    }
    model_name = {"founder_insight": "FOUNDER_MODEL", "market_scout": "MARKET_MODEL",
                  "tech_brief": "TECH_MODEL"}[agent]
    model = create_model_from_env(model_env_var=model_name, timeout=20, max_retries=0).with_structured_output(Estimates)
    response = model.invoke([
        ("system", "제공된 모든 자료와 실행 한계를 함께 검토해 각 항목의 잠정 점수를 추정하라. 해당 평가 항목과 직접 관련된 자료만 인용한다. 근거가 없어도 0~max_score 사이 점수를 부여한다. "
         "정보가 거의 없으면 중립 점수를 사용한다. 검증되지 않은 사실을 단정하지 말고 reason에 추정의 한계를 명시한다. "
         "자료를 사용했다면 해당 evidence_id를 evidence_ids에 넣는다. 요청된 항목만 각각 한 번 반환한다."),
        ("user", json.dumps(payload, ensure_ascii=False)),
    ])
    estimates = Estimates.model_validate(response)
    expected = dict(criteria)
    if {item.criterion for item in estimates.scores} != set(expected) or len(estimates.scores) != len(expected):
        raise ValueError("잠정 점수 항목이 요청과 다릅니다.")
    if any(not is_number(item.score) or not 0 <= item.score <= expected[item.criterion]
           for item in estimates.scores):
        raise ValueError("잠정 점수가 배점 범위를 벗어났습니다.")
    available_ids = {item["evidence_id"] for item in result.get("evidence", [])}
    if any(set(item.evidence_ids) - available_ids for item in estimates.scores):
        raise ValueError("잠정 점수에 없는 자료 ID가 포함되었습니다.")
    return {item.criterion: (item.score, item.reason, item.evidence_ids) for item in estimates.scores}


def fill_missing_scores(result, agent, state=None):
    """누락 항목에 입력 기반 잠정 점수를 부여한다. 모델 실패 시 중립 점수를 쓴다."""
    result = deepcopy(result)
    available_ids = {item.get("evidence_id") for item in result.get("evidence", [])}
    expected = dict(CRITERIA[agent])
    retained_scores = []
    for item in result.get("scores", []):
        criterion = item.get("criterion")
        if criterion not in expected or item.get("score") is None:
            continue
        item = deepcopy(item)
        item["evidence_ids"] = [eid for eid in item.get("evidence_ids", []) if eid in available_ids]
        if not item["evidence_ids"] and not item.get("reason", "").startswith("[잠정 추정]"):
            item["reason"] = f"[잠정 추정] {item.get('reason') or '연결된 자료 없이 산정한 점수'}"
            result["status"] = "partial"
        retained_scores.append(item)
    result["scores"] = retained_scores
    existing = {item.get("criterion") for item in result.get("scores", [])}
    pending = [(name, maximum) for name, maximum in CRITERIA[agent] if name not in existing]
    estimates = {}
    if pending and state is not None:
        try:
            estimates = estimate_missing_scores(state, agent, pending, result)
        except Exception:
            pass
    filled = []
    for criterion, maximum in pending:
        filled.append(criterion)
        estimate = estimates.get(criterion, (maximum * NO_DATA_SCORE_RATIO, NO_DATA_SCORE_REASON, []))
        score, reason, ids = (*estimate, []) if len(estimate) == 2 else estimate
        if criterion not in estimates:
            claims = [item.get("claim", "").strip()[:180] for item in result.get("evidence", [])
                      if item.get("claim", "").strip()]
            if claims:
                reason = f"[잠정 추정] 확보 자료({'; '.join(claims[:2])})를 참고해 임시 점수를 산정함"
                ids = [item["evidence_id"] for item in result.get("evidence", [])[:2]]
            elif state is not None:
                company = state["company_context"]["company_name"]
                request = state["evaluation_request"].strip()[:100]
                reason = f"[잠정 추정] {company}에 대한 입력 요청({request})을 바탕으로 임시 점수를 산정함"
        if not reason.startswith("[잠정 추정]"):
            reason = f"[잠정 추정] {reason}"
        result.setdefault("scores", []).append({
            "criterion": criterion, "score": score, "max_score": maximum,
            "reason": reason, "evidence_ids": ids,
        })
    if filled:
        result.setdefault("missing_items", []).extend(f"{name}: 평가 자료 없음" for name in filled)
    if result.get("status") != "success" or filled:
        result["status"] = "partial"
        result["score"] = sum(item["score"] for item in result["scores"])
    return result

# ──────────────────────────────────────
# 공통 처리 및 출력 검증
# ──────────────────────────────────────

def identity(state):
    return state["company_context"]["company_name"], state["retry_state"]["retry_count"] + 1


def is_current(result, state):
    company, round_no = identity(state)
    return isinstance(result, dict) and result.get("company_name") == company and result.get("round_no") == round_no


def total_possible_score():
    return sum(weight for items in CRITERIA.values() for _, weight in items)


def validate_settings():
    if set(CRITERIA) != set(AGENT_RESULT_FIELDS):
        raise ValueError("CRITERIA와 전문 Agent 구성이 일치해야 합니다.")
    for items in CRITERIA.values():
        if not items or len({name for name, _ in items}) != len(items):
            raise ValueError("평가 항목은 비어 있거나 중복될 수 없습니다.")
        if not all(is_number(weight) and weight > 0 for _, weight in items):
            raise ValueError("배점은 양의 유한한 수여야 합니다.")
    if type(MAX_RETRIES) is not int or MAX_RETRIES < 0:
        raise ValueError("MAX_RETRIES는 0 이상의 정수여야 합니다.")
    if type(MIN_EVIDENCE_COUNT) is not int or MIN_EVIDENCE_COUNT < 1:
        raise ValueError("MIN_EVIDENCE_COUNT는 1 이상의 정수여야 합니다.")
    if type(GRAPH_RECURSION_LIMIT) is not int or GRAPH_RECURSION_LIMIT < 1:
        raise ValueError("GRAPH_RECURSION_LIMIT는 양의 정수여야 합니다.")


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}에 비어 있지 않은 문자열이 필요합니다.")


def require_text_list(value, name):
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name}에 문자열 목록이 필요합니다.")


def error_record(state, node, code, message):
    company, round_no = identity(state)
    return {
        "company_name": company, "round_no": round_no,
        "node": node, "code": code, "message": message,
    }


def call_agent(agent, state, function="run", **kwargs):
    module = import_module(AGENT_MODULES[agent])
    handler = getattr(module, function, None)
    if not callable(handler):
        raise NotImplementedError(f"{AGENT_MODULES[agent]}.{function} 함수를 구현해야 합니다.")
    # 병렬 노드와 원본 State를 변경하지 않도록 별도 값을 전달한다.
    return handler(state=deepcopy(state), **deepcopy(kwargs))


def validate_agent_result(result, state, agent):
    if not is_current(result, state) or result.get("agent") != agent:
        raise ValueError("평가 기업·회차 또는 Agent 이름이 일치하지 않습니다.")
    if result.get("status") not in {"success", "partial", "error"}:
        raise ValueError("Agent 상태 값이 올바르지 않습니다.")
    require_text(result.get("summary"), "summary")
    require_text_list(result.get("missing_items"), "missing_items")
    if not isinstance(result.get("evidence"), list) or not isinstance(result.get("scores"), list):
        raise ValueError("evidence와 scores는 목록이어야 합니다.")

    evidence_ids = set()
    for evidence in result["evidence"]:
        if not isinstance(evidence, dict):
            raise ValueError("근거는 딕셔너리여야 합니다.")
        for key in ("evidence_id", "claim", "excerpt", "source"):
            require_text(evidence.get(key), key)
        if evidence["evidence_id"] in evidence_ids:
            raise ValueError("근거 ID가 중복됩니다.")
        evidence_ids.add(evidence["evidence_id"])
        if evidence.get("source_type") not in {"pdf", "web", "input"}:
            raise ValueError("지원하지 않는 출처 유형입니다.")
        if "page" in evidence and (type(evidence["page"]) is not int or evidence["page"] < 1):
            raise ValueError("페이지 번호는 양의 정수여야 합니다.")

    expected = dict(CRITERIA[agent])
    seen = set()
    for item in result["scores"]:
        if not isinstance(item, dict):
            raise ValueError("점수 항목은 딕셔너리여야 합니다.")
        name = item.get("criterion")
        if not isinstance(name, str) or name not in expected or name in seen:
            raise ValueError("평가 항목이 누락되거나 중복되었거나 담당 범위를 벗어났습니다.")
        seen.add(name)
        maximum = item.get("max_score")
        score = item.get("score")
        if not is_number(maximum) or maximum != expected[name]:
            raise ValueError("평가 항목의 배점이 설정과 다릅니다.")
        if not is_number(score) or not 0 <= score <= maximum:
            raise ValueError("평가 점수가 범위를 벗어났습니다.")
        require_text(item.get("reason"), "reason")
        require_text_list(item.get("evidence_ids"), "evidence_ids")
        provisional = result["status"] != "success" and item["reason"].startswith("[잠정 추정]")
        if not provisional and (not item["evidence_ids"] or not set(item["evidence_ids"]).issubset(evidence_ids)):
            raise ValueError("점수가 실제 제공된 근거와 연결되지 않았습니다.")

    if result["status"] == "success":
        if seen != set(expected) or len(evidence_ids) < MIN_EVIDENCE_COUNT or result["missing_items"]:
            raise ValueError("완료된 평가에 필수 항목 또는 근거가 부족합니다.")
        if "score" not in result:
            raise ValueError("완료된 평가에 합계 점수가 없습니다.")
    if "score" in result:
        if not is_number(result["score"]) or not math.isclose(
            result["score"], sum(item["score"] for item in result["scores"]),
            rel_tol=0, abs_tol=SCORE_TOLERANCE,
        ):
            raise ValueError("Agent 합계와 항목별 점수 합계가 일치하지 않습니다.")


def run_specialist(state, agent, **agent_kwargs):
    field = AGENT_RESULT_FIELDS[agent]
    try:
        result = fill_missing_scores(call_agent(agent, state, criteria=CRITERIA[agent], **agent_kwargs), agent, state)
        validate_agent_result(result, state, agent)
        return {field: result}
    except Exception as exc:
        company, round_no = identity(state)
        detail = str(exc)[:200] if isinstance(exc, (ValueError, TypeError, KeyError)) else type(exc).__name__
        return {
            field: fill_missing_scores({
                "company_name": company, "round_no": round_no, "agent": agent,
                "status": "partial", "summary": f"실행 한계({detail})를 반영한 잠정 평가입니다.",
                "evidence": [], "missing_items": [f"실행 참고: {detail}"], "scores": [],
            }, agent, state),
        }


# ──────────────────────────────────────
# 노드 함수: 변경할 State 필드만 반환
# ──────────────────────────────────────

def start_evaluation(state: InvestmentState):
    return {"evaluation_status": {"decision": "pending", "stop_reason": "in_progress"}}


def founder_insight(state: InvestmentState):
    return run_specialist(state, "founder_insight")


def market_scout(state: InvestmentState):
    from rag.market_scout import rag_search

    return run_specialist(state, "market_scout", rag_search=rag_search)


def tech_brief(state: InvestmentState):
    return run_specialist(state, "tech_brief")


def integrate_results(state: InvestmentState):
    company, round_no = identity(state)
    results, missing, updates = [], [], {}
    for agent, key in AGENT_RESULT_FIELDS.items():
        result = state.get(key, {})
        try:
            validate_agent_result(result, state, agent)
            results.append(result)
        except (ValueError, TypeError, KeyError) as exc:
            missing.append(f"{agent}: {exc}")

    # 병렬 Agent가 각각 확보한 자료를 모은 뒤, 출처 연결이 없는 잠정 항목을 한 번 더 검토한다.
    all_evidence = {item["evidence_id"]: item for result in results for item in result["evidence"]}
    for agent, field in AGENT_RESULT_FIELDS.items():
        original = state.get(field)
        if not isinstance(original, dict) or original not in results:
            continue
        unlinked = [item for item in original["scores"]
                    if item["reason"].startswith("[잠정 추정]") and not item["evidence_ids"]]
        if not unlinked or not all_evidence:
            continue
        candidate = deepcopy(original)
        owned = {item["evidence_id"] for item in candidate["evidence"]}
        candidate["evidence"].extend(item for eid, item in all_evidence.items() if eid not in owned)
        criteria = [(item["criterion"], item["max_score"]) for item in unlinked]
        try:
            estimates = estimate_missing_scores(state, agent, criteria, candidate)
        except Exception:
            continue
        for item in unlinked:
            score, reason, ids = estimates[item["criterion"]]
            if not ids:
                continue
            target = next(score_item for score_item in candidate["scores"]
                          if score_item["criterion"] == item["criterion"])
            target.update(score=score,
                          reason=reason if reason.startswith("[잠정 추정]") else f"[잠정 추정] {reason}",
                          evidence_ids=ids)
            candidate["missing_items"] = [value for value in candidate["missing_items"]
                                          if value != f"{item['criterion']}: 평가 자료 없음"]
            candidate["missing_items"].append(f"{item['criterion']}: 타 Agent 자료를 활용한 잠정 평가")
        selected_ids = {eid for item in candidate["scores"] for eid in item["evidence_ids"]}
        candidate["evidence"] = [item for item in candidate["evidence"]
                                 if item["evidence_id"] in owned | selected_ids]
        candidate["score"] = sum(item["score"] for item in candidate["scores"])
        validate_agent_result(candidate, state, agent)
        updates[field] = candidate
        results = [candidate if result is original else result for result in results]

    for result in results:
        missing.extend(result["missing_items"])
        if result["status"] != "success":
            missing.append(f"{result['agent']}: 평가 미완료")

    connected_scores = sum(bool(item["evidence_ids"])
                          for result in results for item in result["scores"])
    review = {
        "company_name": company, "round_no": round_no,
        "integrated_summary": "\n".join(r["summary"] for r in results),
        "evidence_sufficient": bool(connected_scores),
        "reason": (f"자료 연결 점수 {connected_scores}개와 잠정 점수를 함께 평가합니다."
                   if connected_scores else "연결된 자료가 없어 잠정 평가만 가능합니다."),
        "missing_items": list(dict.fromkeys(missing)), "conflicts": [],
    }
    return {"evidence_review": review, **updates}


def gate_update(state, gate, outcome, next_node, reason):
    company, round_no = identity(state)
    update = {"route_history": [{
        "company_name": company, "round_no": round_no, "gate": gate,
        "outcome": outcome, "next_node": next_node, "reason": reason,
    }]}
    if not outcome:
        retry_reason = "insufficient_evidence" if gate == "evidence" else "criteria_not_met"
        update["retry_state"] = {**state["retry_state"], "retry_reason": retry_reason}
        update["evaluation_status"] = {
            "decision": "hold" if gate == "evidence" else "reject",
            "stop_reason": retry_reason,
        }
    elif gate == "investment":
        update["evaluation_status"] = {"decision": "recommend", "stop_reason": "criteria_met"}
    return update


def evidence_gate(state: InvestmentState):
    review = state["evidence_review"]
    return gate_update(state, "evidence", review["evidence_sufficient"], choose_evidence_route(state), review["reason"])


def investment_evaluator(state: InvestmentState):
    try:
        if not is_current(state.get("evidence_review"), state):
            raise ValueError("현재 회차의 근거 검토 결과가 필요합니다.")
        for agent, field in AGENT_RESULT_FIELDS.items():
            validate_agent_result(state[field], state, agent)
        result = call_agent("investment_evaluator", state, function="assess")
        if not is_current(result, state) or not is_number(result.get("total_score")) or not 0 <= result["total_score"] <= 100:
            raise ValueError("종합 평가의 기업·회차 또는 점수가 올바르지 않습니다.")
        if type(result.get("criteria_met")) is not bool or type(result.get("provisional")) is not bool:
            raise ValueError("종합 평가 판단 값이 올바르지 않습니다.")
        require_text(result.get("decision_reason"), "투자 판단 설명")
        return {"investment_result": result}
    except Exception as exc:
        return {"errors": [error_record(state, "investment_evaluator", "INVESTMENT_EVALUATION_FAILED", f"{type(exc).__name__}: {exc}")]}


def investment_gate(state: InvestmentState):
    result = state.get("investment_result", {})
    if not is_current(result, state):
        update = gate_update(state, "investment", False, "prepare_retry", "투자 판단을 완료하지 못했습니다.")
        update["evaluation_status"] = {"decision": "hold", "stop_reason": "insufficient_evidence"}
        update["retry_state"] = {**state["retry_state"], "retry_reason": "insufficient_evidence"}
        return update
    next_node = "save_company_result" if result["criteria_met"] else "prepare_retry"
    return gate_update(state, "investment", result["criteria_met"], next_node, result["decision_reason"])


def prepare_retry(state: InvestmentState):
    retry = state["retry_state"]

    # 이미 1회 재탐색했다면 더 이상 증가시키지 않음
    if retry["retry_count"] >= MAX_RETRIES:
        return {
            "evaluation_status": {
                **state["evaluation_status"],
                "stop_reason": "max_retry_reached",
            }
        }

    return {
        "retry_state": {
            **retry,
            "retry_count": retry["retry_count"] + 1,
        },
        "evaluation_status": {
            "decision": "pending",
            "stop_reason": "retry",
        },
    }


def save_company_result(state: InvestmentState):
    company, round_no = identity(state)
    record = {
        "company_name": company, "round_no": round_no,
        "evaluation_status": deepcopy(state["evaluation_status"]),
    }
    for key in (*AGENT_RESULT_FIELDS.values(), "evidence_review", "investment_result"):
        result = state.get(key, {})
        if is_current(result, state):
            record[key] = deepcopy(result)
    record["errors"] = [deepcopy(item) for item in state.get("errors", []) if is_current(item, state)]
    return {"company_results": [record]}


def next_company(state: InvestmentState):
    context = state["company_context"]
    index = context["company_index"] + 1
    return {
        "company_context": {**context, "company_index": index, "company_name": context["company_candidates"][index]},
        "retry_state": {**state["retry_state"], "retry_count": 0, "retry_reason": "initial"},
        "evaluation_status": {"decision": "pending", "stop_reason": "in_progress"},
    }


def report_generator(state: InvestmentState):
    try:
        report = call_agent("report_generator", state)
        require_text(report, "보고서")
        return {"report": report}
    except Exception as exc:
        return {"errors": [error_record(state, "report_generator", "REPORT_FAILED", f"{type(exc).__name__}: {exc}")]}


# ──────────────────────────────────────
# 조건부 Edge: State를 변경하지 않고 다음 노드 이름 반환
# ──────────────────────────────────────

def choose_evidence_route(state):
    return "investment_evaluator"


def choose_investment_route(state):
    return "save_company_result" if state["evaluation_status"]["decision"] == "recommend" else "prepare_retry"


def choose_retry_route(state):
    if (
        state["retry_state"]["retry_count"] == 1
        and state["evaluation_status"]["decision"] == "pending"
    ):
        return "start_evaluation"

    return "save_company_result"


def choose_company_route(state):
    context = state["company_context"]
    if STOP_ON_FIRST_RECOMMENDATION and state["evaluation_status"]["decision"] == "recommend":
        return "report_generator"
    if context["company_index"] + 1 < len(context["company_candidates"]):
        return "next_company"
    return "report_generator"


# ──────────────────────────────────────
# 그래프 생성
# ──────────────────────────────────────

def build_graph():
    validate_settings()
    builder = StateGraph(InvestmentState)

    builder.add_node("start_evaluation", start_evaluation)
    builder.add_node("founder_insight", founder_insight)
    builder.add_node("market_scout", market_scout)
    builder.add_node("tech_brief", tech_brief)
    builder.add_node("integrate_results", integrate_results)
    builder.add_node("evidence_gate", evidence_gate)
    builder.add_node("investment_evaluator", investment_evaluator)
    builder.add_node("investment_gate", investment_gate)
    builder.add_node("prepare_retry", prepare_retry)
    builder.add_node("save_company_result", save_company_result)
    builder.add_node("next_company", next_company)
    builder.add_node("report_generator", report_generator)

    builder.add_edge(START, "start_evaluation")
    for agent in AGENT_RESULT_FIELDS:
        builder.add_edge("start_evaluation", agent)
    # 모든 전문 Agent가 끝난 뒤 통합 노드를 한 번 실행한다.
    builder.add_edge(list(AGENT_RESULT_FIELDS), "integrate_results")
    builder.add_edge("integrate_results", "evidence_gate")
    builder.add_conditional_edges("evidence_gate", choose_evidence_route, {
        "investment_evaluator": "investment_evaluator", "prepare_retry": "prepare_retry",
    })
    builder.add_edge("investment_evaluator", "investment_gate")
    builder.add_conditional_edges("investment_gate", choose_investment_route, {
        "save_company_result": "save_company_result", "prepare_retry": "prepare_retry",
    })
    builder.add_conditional_edges("prepare_retry", choose_retry_route, {
        "start_evaluation": "start_evaluation", "save_company_result": "save_company_result",
    })
    builder.add_conditional_edges("save_company_result", choose_company_route, {
        "next_company": "next_company", "report_generator": "report_generator",
    })
    builder.add_edge("next_company", "start_evaluation")
    builder.add_edge("report_generator", END)
    return builder.compile()


def make_initial_state(companies: list[str], request: str) -> InvestmentState:
    validate_settings()
    require_text_list(companies, "기업 목록")
    require_text(request, "평가 요청")
    names = list(dict.fromkeys(name.strip() for name in companies))
    if not names:
        raise ValueError("평가할 기업을 한 개 이상 입력하세요.")
    return {
        "company_context": {"company_candidates": names, "company_index": 0, "company_name": names[0]},
        "evaluation_request": request,
        "retry_state": {"max_retries": MAX_RETRIES, "retry_count": 0, "retry_reason": "initial"},
        "evaluation_status": {"decision": "pending", "stop_reason": "in_progress"},
        "errors": [], "route_history": [], "company_results": [],
    }


def configure_logging(enable_tracing: bool):
    if not enable_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    project_name = setting("LANGSMITH_PROJECT")
    if not ENV.get("LANGSMITH_API_KEY"):
        raise ValueError(".env에 LangSmith API 키를 설정하세요.")
    from langchain_teddynote import logging
    logging.langsmith(project_name=project_name)
