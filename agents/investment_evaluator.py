"""전문 평가 점수와 근거를 종합해 투자 점수와 판단을 생성한다."""

import json
from pydantic import BaseModel, Field

from prompts.investment_evaluator_prompt import INVESTMENT_EVALUATOR_PROMPT as INVESTMENT_SYSTEM_PROMPT

from .evaluation_support import (
    CRITERIA, InvestmentAgentInput, InvestmentResult, calculate_total,
    collect_evidence, mapping, unique,
    graph_agent_input, resolve_model,
    create_model_from_env,
)

# 실행 설정: 외부에서 model을 주입하면 이 기본값보다 우선한다.
MODEL = None
MODEL_FACTORY = create_model_from_env
MODEL_SETTINGS = {"model_env_var": "INVESTMENT_MODEL", "timeout": 60, "max_retries": 0}

# 프롬프트 수정 위치: prompts/investment_evaluator_prompt.py의 INVESTMENT_EVALUATOR_PROMPT.
# 위 import로 직접 불러오므로 Agent 코드에 본문을 복사하지 않는다.
# Graph의 assess()는 항목 합계를 기준으로 종합 점수와 추천 여부를 모델에 요청한다.


class InvestmentEvaluator:
    def __init__(self, model=None):
        self.model = model

    def invoke(self, agent_input: InvestmentAgentInput | dict) -> InvestmentResult:
        # RAG 결과 수신 지점: Graph가 시장성 결과를 market_result,
        # 제품·기술 결과를 tech_result에 넣어 전달한다. 이 Agent는 RAG를 직접 호출하지 않는다.
        # 모델 생성 후 변경된 값도 매 호출마다 재검증한다.
        data = InvestmentAgentInput(**mapping(agent_input, "investment input"))
        collect_evidence(data.results)
        risks, missing, reasons = [], [], []
        needs_more = False
        for result in data.results:
            role = result["agent_name"]
            risks.extend(result["risks"])
            missing.extend(result["missing_information"])
            needs_more |= result["needs_more_information"]
            evaluations = {item["criterion"]: item for item in result["evaluations"]}
            for criterion in CRITERIA[role]:
                item = evaluations.get(criterion)
                if item is None:
                    missing.append(f"{criterion}: 평가 결과 누락")
                    continue
                reasons.append(f"{criterion}: {item['reason']}")
                if item["score"] is None:
                    missing.append(f"{criterion}: 점수 미제공")
                if not item["evidence_ids"]:
                    missing.append(f"{criterion}: 연결된 근거 없음")
            if result["needs_more_information"] and not result["missing_information"]:
                missing.append(f"{role}: 추가 정보 요청의 구체적인 내용 확인 필요")
        needs_more = bool(needs_more or missing)
        base_score = calculate_total(data.results)
        if base_score is None:
            base_score = sum(item["score"] for result in data.results
                             for item in result["evaluations"] if item["score"] is not None)
        score, decision = base_score, "pending"
        if self.model is not None:
            try:
                judgement = _invoke_judgement(self.model, {
                    "company_name": data.company_name,
                    "company_context": data.company_context,
                    "specialist_score_sum": base_score,
                    "specialist_results": data.results,
                    "evidence_review": data.evidence_review,
                })
                score = judgement.total_score
                decision = "recommend" if judgement.recommend else "reject"
                reasons.append(f"전문 항목 합계 {base_score:g}/100점, 종합 판단 {score:g}/100점. {judgement.reason}")
            except Exception as exc:
                reasons.append(f"종합 모델 판단 실패로 잠정 점수만 표시: {type(exc).__name__}: {exc}")
        else:
            reasons.append("종합 모델 판단이 없어 추천 여부는 미정입니다.")
        result = InvestmentResult(
            company_name=data.company_name,
            total_score=score,
            decision=decision,
            key_reasons=unique(reasons), risks=unique(risks),
            missing_information=unique(missing), needs_more_information=needs_more,
            provisional=needs_more,
        )
        return result


class HolisticAssessment(BaseModel):
    total_score: float = Field(ge=0, le=100)
    recommend: bool
    reason: str = Field(min_length=1)


def _invoke_judgement(model, payload):
    response = model.with_structured_output(HolisticAssessment).invoke([
        ("system", INVESTMENT_SYSTEM_PROMPT),
        ("user", json.dumps(payload, ensure_ascii=False, allow_nan=False)),
    ])
    return HolisticAssessment.model_validate(response)


def assess(state, *, model=None):
    """항목 점수를 기준점으로 삼되 종합 점수와 추천 여부는 모델이 판단한다."""
    values = graph_agent_input(state)
    data = InvestmentAgentInput(**values)
    calculated = calculate_total(data.results)
    base_score = calculated if calculated is not None else sum(
        item["score"] for result in data.results for item in result["evaluations"]
        if item["score"] is not None
    )
    company = data.company_name
    round_no = state["retry_state"]["retry_count"] + 1
    provisional = calculated is None or bool(data.evidence_review.get("missing_items")) or any(
        result["needs_more_information"] or any(
            item["reason"].startswith("[잠정 추정]") or not item["evidence_ids"]
            for item in result["evaluations"]
        ) for result in data.results
    )
    payload = {
        "company_name": company,
        "company_context": data.company_context,
        "evaluation_request": state["evaluation_request"],
        "specialist_score_sum": base_score,
        "specialist_results": data.results,
        "evidence_review": data.evidence_review,
        "previous_errors": [
            {"node": item.get("node"), "code": item.get("code"), "message": item.get("message")}
            for item in state.get("errors", []) if item.get("company_name") == company
        ][-10:],
    }
    try:
        llm = resolve_model(model, MODEL, MODEL_FACTORY, MODEL_SETTINGS)
        judgement = _invoke_judgement(llm, payload)
        score = judgement.total_score
        recommend = judgement.recommend
        reason = (f"전문 항목 합계 {base_score:g}/100점, 종합 판단 {score:g}/100점. "
                  f"{judgement.reason}")
    except Exception as exc:
        score = base_score
        recommend = False
        provisional = True
        reason = (f"전문 항목 합계 {base_score:g}/100점을 잠정 종합 점수로 사용했습니다. "
                  f"종합 모델 판단을 완료하지 못해 추천은 보류합니다: {type(exc).__name__}: {exc}")
    return {
        "company_name": company, "round_no": round_no, "total_score": score,
        "criteria_met": recommend, "provisional": provisional,
        "decision_reason": reason,
    }


def run(state, *, model=None, **kwargs):
    """기존 문자열 반환 인터페이스와의 호환성을 유지한다."""
    return assess(state, model=model)["decision_reason"]
