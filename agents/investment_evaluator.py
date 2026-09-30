"""외부 검색/LLM 없이 전문 평가를 통합하는 종합 투자 판단 Agent."""

from prompts.investment_evaluator_prompt import INVESTMENT_EVALUATOR_PROMPT as INVESTMENT_SYSTEM_PROMPT

from .evaluation_support import (
    CRITERIA, InvestmentAgentInput, InvestmentResult, calculate_total,
    collect_evidence, mapping, unique,
)

# 프롬프트 수정 위치: prompts/investment_evaluator_prompt.py의 INVESTMENT_EVALUATOR_PROMPT.
# 위 import로 직접 불러오므로 Agent 코드에 본문을 복사하지 않는다.
# 현재 이 상수는 사용하지 않는다. LLM/API 확정 후 호출부를 별도로 연결해야 한다.
# TODO(입출력 연동): 프롬프트가 요구하는 criteria_met, evidence_review, 판단 설정값을
# Graph에서 전달받는 계약을 확정한다. 현재 InvestmentAgentInput에는 이 필드가 없다.
# 프롬프트의 반환값은 판단 근거 문자열이다. InvestmentResult 전체를 대체하지 않고,
# 향후 설명 필드에 연결하되 코드에서 계산한 점수와 판단을 유지해야 한다.


class InvestmentEvaluator:
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
        # TODO(프롬프트/LLM 연동): 이 파일의 InvestmentEvaluator.invoke()를 수정해
        # 통합된 평가·근거·위험·부족 정보와 계산된 총점을
        # 최종 프롬프트에 전달하는 호출부를 연결한다. 합산은 calculate_total()에 유지한다.
        # 통과·보류 기준도 팀에서 받아야 하며, 프롬프트만 추가해서 임계값을 만들지 않는다.
        reasons.append("투자 통과·보류 기준 미확정으로 최종 판단 대기")
        return InvestmentResult(
            company_name=data.company_name,
            total_score=calculate_total(data.results),
            decision="additional_research" if needs_more else "pending",
            key_reasons=unique(reasons), risks=unique(risks),
            missing_information=unique(missing), needs_more_information=needs_more,
        )
