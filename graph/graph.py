import math
import os
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from .state import InvestmentState

from langgraph.graph import END, START, StateGraph


# ──────────────────────────────────────
# 평가 기준 및 실행 설정: 변경할 값은 이곳에서 관리
# ──────────────────────────────────────

RECOMMEND_SCORE = 75.0
RAW_SCORE_MAX = 5.0
MIN_CRITERION_SCORE = 3.0
MAX_RETRIES = 1
MIN_EVIDENCE_COUNT = 1
STOP_ON_FIRST_RECOMMENDATION = True
GRAPH_RECURSION_LIMIT = 500
SCORE_TOLERANCE = 1e-6
ENABLE_LANGSMITH = False

CRITERIA = {
    "founder_insight": [
        ("창업자·팀 역량", 25.0),
    ],
    "market_scout": [
        ("시장 규모·성장 가능성", 25.0),
        ("실제 고객 수요", 25.0),
    ],
    "tech_brief": [
        ("제품·기술 정보의 구체성", 15.0),
        ("제품 차별성·성능 정보의 명확성", 10.0),
    ],
}

AGENT_RESULT_FIELDS = {
    "founder_insight": "founder_result",
    "market_scout": "market_result",
    "tech_brief": "tech_result",
}
AGENT_MODULES = {
    "founder_insight": "agents.founder_insight_agents",
    "market_scout": "agents.market_scout_agents",
    "tech_brief": "agents.tech_brief_agents",
    "investment_evaluator": "agents.investment_evaluator_agents",
    "report_generator": "agents.report_generator_agents",
}

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
    if not is_number(RECOMMEND_SCORE) or not 0 <= RECOMMEND_SCORE <= total_possible_score():
        raise ValueError("추천 점수는 총 배점 범위 안이어야 합니다.")
    if not is_number(RAW_SCORE_MAX) or RAW_SCORE_MAX <= 0:
        raise ValueError("원점수 만점은 양수여야 합니다.")
    if not is_number(MIN_CRITERION_SCORE) or not 0 <= MIN_CRITERION_SCORE <= RAW_SCORE_MAX:
        raise ValueError("항목별 최소 원점수가 범위를 벗어났습니다.")
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
        if not item["evidence_ids"] or not set(item["evidence_ids"]).issubset(evidence_ids):
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


def run_specialist(state, agent):
    field = AGENT_RESULT_FIELDS[agent]
    try:
        result = call_agent(agent, state, criteria=CRITERIA[agent])
        validate_agent_result(result, state, agent)
        update = {field: result}
        if result["status"] != "success":
            update["errors"] = [error_record(
                state, agent, "AGENT_INCOMPLETE", result["summary"],
            )]
        return update
    except Exception as exc:
        company, round_no = identity(state)
        message = f"{type(exc).__name__}: {exc}"
        return {
            field: {
                "company_name": company, "round_no": round_no, "agent": agent,
                "status": "error", "summary": "평가를 완료하지 못했습니다.",
                "evidence": [], "missing_items": [message], "scores": [],
            },
            "errors": [error_record(state, agent, "AGENT_FAILED", message)],
        }


# ──────────────────────────────────────
# 노드 함수: 변경할 State 필드만 반환
# ──────────────────────────────────────

def start_evaluation(state: InvestmentState):
    return {"evaluation_status": {"decision": "pending", "stop_reason": "in_progress"}}


def founder_insight(state: InvestmentState):
    return run_specialist(state, "founder_insight")


def market_scout(state: InvestmentState):
    return run_specialist(state, "market_scout")


def tech_brief(state: InvestmentState):
    return run_specialist(state, "tech_brief")


def integrate_results(state: InvestmentState):
    company, round_no = identity(state)
    results, missing, errors = [], [], []
    for agent, key in AGENT_RESULT_FIELDS.items():
        result = state.get(key, {})
        try:
            validate_agent_result(result, state, agent)
            results.append(result)
            missing.extend(result["missing_items"])
            if result["status"] != "success":
                missing.append(f"{agent}: 평가 미완료")
        except (ValueError, TypeError, KeyError) as exc:
            missing.append(f"{agent}: {exc}")

    review = {
        "company_name": company, "round_no": round_no,
        "integrated_summary": "\n".join(r["summary"] for r in results),
        "evidence_sufficient": False, "reason": "필수 평가 결과가 부족합니다.",
        "missing_items": list(dict.fromkeys(missing)), "conflicts": [],
    }
    if not missing:
        try:
            checked = call_agent("investment_evaluator", state, function="review_evidence")
            if not is_current(checked, state):
                raise ValueError("근거 검토의 기업·회차가 일치하지 않습니다.")
            for key in ("integrated_summary", "reason"):
                require_text(checked.get(key), key)
            for key in ("missing_items", "conflicts"):
                require_text_list(checked.get(key), key)
            if type(checked.get("evidence_sufficient")) is not bool:
                raise ValueError("근거 충분 여부는 bool이어야 합니다.")
            if checked["evidence_sufficient"] and (checked["missing_items"] or checked["conflicts"]):
                raise ValueError("미해결 보완 항목 또는 모순이 있는 결과를 충분으로 판단할 수 없습니다.")
            review = checked
        except Exception as exc:
            review["reason"] = "근거 내용 검토를 완료하지 못했습니다."
            review["missing_items"] = [review["reason"]]
            errors.append(error_record(state, "integrate_results", "EVIDENCE_REVIEW_FAILED", f"{type(exc).__name__}: {exc}"))
    if not review["evidence_sufficient"]:
        errors.append(error_record(state, "integrate_results", "INSUFFICIENT_EVIDENCE", review["reason"]))
    return {"evidence_review": review, "errors": errors}


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
    company, round_no = identity(state)
    try:
        if not is_current(state.get("evidence_review"), state) or not state["evidence_review"]["evidence_sufficient"]:
            raise ValueError("현재 회차의 충분한 근거 검토가 필요합니다.")
        results = []
        for agent, field in AGENT_RESULT_FIELDS.items():
            result = state[field]
            validate_agent_result(result, state, agent)
            if result["status"] != "success":
                raise ValueError("완료되지 않은 전문 평가가 있습니다.")
            results.append(result)
        total = sum(result["score"] for result in results)
        meets = total >= RECOMMEND_SCORE and all(
            item["score"] / item["max_score"] * RAW_SCORE_MAX >= MIN_CRITERION_SCORE
            for result in results for item in result["scores"]
        )
        reason = call_agent("investment_evaluator", state, total_score=total, criteria_met=meets)
        require_text(reason, "투자 판단 설명")
        return {"investment_result": {
            "company_name": company, "round_no": round_no, "total_score": total,
            "criteria_met": meets, "decision_reason": reason,
        }}
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
    return "investment_evaluator" if state["evidence_review"]["evidence_sufficient"] else "prepare_retry"


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
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
    if not enable_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return
    project_name = os.environ.get("LANGSMITH_PROJECT")
    if not project_name:
        raise ValueError(".env에 LANGSMITH_PROJECT를 설정하세요.")
    if not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):
        raise ValueError(".env에 LangSmith API 키를 설정하세요.")
    from langchain_teddynote import logging
    logging.langsmith(project_name=project_name)

